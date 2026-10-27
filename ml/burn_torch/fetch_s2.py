"""Cas d'étude réel : télécharge un couple Sentinel-2 L2A avant/après incendie.

    pip install -r requirements/geo.txt
    python -m ml.burn_torch.fetch_s2                 # préréglage Melloula (Tabarka, 2023)
    python -m ml.burn_torch.fetch_s2 --center 8.70,36.93   # fenêtre imposée (lon,lat)

Source : catalogue STAC Earth Search (Element 84), images Copernicus Sentinel-2 L2A
hébergées en COG sur AWS ; aucun compte n'est nécessaire. Seules les fenêtres utiles
sont lues (≈ 30 à 50 Mo au total).

Étapes :
1. recherche des scènes avant et après l'incendie sur la zone ;
2. choix, sur une même tuile MGRS, des deux scènes les moins nuageuses *sur la zone*
   (masque SCL), et non sur la tuile entière ;
3. calcul du dNBR sur toute la zone, puis choix automatique de la fenêtre 512 × 512
   (10,24 km) qui contient le plus de pixels brûlés ;
4. lecture des 12 bandes à 20 m sur cette fenêtre, au format CaBuAr, et ajout de
   l'échantillon en tête de data/scenarios/burn/index.json.

Les scènes traitées avec la version de traitement 04.00 ou plus (2022+) ont un
décalage radiométrique de +1000 ; on le retire pour retrouver la convention des
données d'entraînement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from ml.burn_torch.constants import PIXEL_AREA_HA, S2_BANDS

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
# Noms des bandes dans Earth Search (les noms « B01 »… servent de repli).
ASSETS = {"B01": "coastal", "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1",
          "B06": "rededge2", "B07": "rededge3", "B08": "nir", "B8A": "nir08", "B09": "nir09",
          "B11": "swir16", "B12": "swir22"}
SCL_CLOUD = (3, 8, 9, 10)  # ombre de nuage, nuages moyens et denses, cirrus
SCL_NODATA = (0, 1)        # absence de données, pixel saturé
SCL_WATER = 6
BOA_OFFSET = 1000
DNBR_THRESHOLD = 0.24      # seuil de la baseline réglé sur l'entraînement CaBuAr

PRESETS = {
    "melloula": {
        "id": "melloula-2023",
        "label": "Melloula (Tabarka), juillet 2023",
        # Délégation de Tabarka, du littoral à la frontière algérienne.
        "bbox": (8.50, 36.78, 8.95, 36.99),
        "pre": "2023-06-10/2023-07-11",
        "post": "2023-07-29/2023-08-25",  # après la maîtrise des feux (fin juillet)
        "reference_ha": 450,
        "reference": "≈ 450 ha de pin maritime selon l'arrondissement forestier de Tabarka "
                     "(ensemble des foyers de Melloula et d'Aïn Sobh, juillet 2023)",
    },
}

GDAL_ENV = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "8",
    "GDAL_HTTP_RETRY_DELAY": "3",
    "GDAL_HTTP_TIMEOUT": "180",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "VSI_CACHE": "TRUE",
}


# --- Fonctions pures (testées sans réseau) ----------------------------------------------

def harmonize(dn: np.ndarray, baseline: str | None, offset_applied: bool) -> np.ndarray:
    """Retire le décalage +1000 des traitements ≥ 04.00, sans créer de faux « no data »."""
    try:
        needs = baseline is not None and float(baseline) >= 4.0 and not offset_applied
    except ValueError:
        needs = False
    if not needs:
        return dn.astype(np.uint16)
    out = dn.astype(np.int32) - BOA_OFFSET
    return np.where(dn > 0, np.clip(out, 1, None), 0).astype(np.uint16)


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    a, b = nir.astype(np.float32), swir2.astype(np.float32)
    return np.where(a + b > 0, (a - b) / np.maximum(a + b, 1.0), 0.0)


def best_window(score: np.ndarray, size: int) -> tuple[int, int, int]:
    """(ligne, colonne, total) de la fenêtre size × size de somme maximale."""
    h, w = score.shape
    sh, sw = min(size, h), min(size, w)
    ii = np.pad(score.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = ii[sh:, sw:] - ii[:-sh, sw:] - ii[sh:, :-sw] + ii[:-sh, :-sw]
    r, c = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(r), int(c), int(sums[r, c])


def pick_scene(scored: list, when: str, tolerance: float = 0.03):
    """Scène quasi aussi dégagée que la meilleure, la plus proche de l'incendie.

    Plus l'écart de dates est court, moins la végétation change pour d'autres raisons
    (sécheresse estivale, moissons) : le dNBR ne garde alors que le feu.
    """
    best = max(c for c, _ in scored)
    ok = [(c, it) for c, it in scored if c >= best - tolerance]
    return (max if when == "pre" else min)(ok, key=lambda c: c[1].datetime)


def merge_index(index: dict, entry: dict) -> dict:
    """Place l'échantillon en tête de l'index (en remplaçant une version précédente)."""
    samples = [s for s in index.get("samples", []) if s.get("id") != entry["id"]]
    return {**index, "samples": [entry, *samples]}


# --- Accès aux données (réseau) ---------------------------------------------------------

def _href(item, band: str) -> str:
    for key in (ASSETS.get(band, band), band, band.lower()):
        if key in item.assets:
            return item.assets[key].href
    raise KeyError(f"Bande {band} absente de {item.id}")


def _tile(item) -> str:
    p = item.properties
    if p.get("s2:mgrs_tile"):
        return str(p["s2:mgrs_tile"])
    if p.get("mgrs:utm_zone"):
        return "".join(str(p.get(k, "")) for k in
                       ("mgrs:utm_zone", "mgrs:latitude_band", "mgrs:grid_square"))
    return item.id.split("_")[1]


def _baseline(item) -> tuple[str | None, bool]:
    p = item.properties
    return p.get("s2:processing_baseline"), bool(p.get("earthsearch:boa_offset_applied", False))


class Grid:
    """Grille cible à 20 m : la fenêtre de la zone dans la bande B12 d'une tuile."""

    def __init__(self, item, bbox):
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds

        with rasterio.open(_href(item, "B12")) as src:
            self.crs = src.crs
            b = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
            win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            self.transform = src.window_transform(win)
            self.shape = (int(win.height), int(win.width))

    def sub(self, row: int, col: int, h: int, w: int) -> Grid:
        from rasterio.windows import Window
        from rasterio.windows import transform as window_transform

        g = object.__new__(Grid)
        g.crs = self.crs
        g.transform = window_transform(Window(col, row, w, h), self.transform)
        g.shape = (h, w)
        return g

    def bounds(self):
        from rasterio.transform import array_bounds

        # array_bounds renvoie (ouest, sud, est, nord)
        return array_bounds(*self.shape, self.transform)

    def lonlat_bounds(self):
        from rasterio.warp import transform_bounds

        return transform_bounds(self.crs, "EPSG:4326", *self.bounds(), densify_pts=21)


def read_band(item, band: str, grid: Grid, categorical: bool = False) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(_href(item, band)) as src:
        if src.crs != grid.crs:
            raise ValueError(f"{item.id} : projection {src.crs} ≠ {grid.crs}")
        win = src.window(*grid.bounds())
        rs = Resampling.nearest if categorical else Resampling.average
        if src.res[0] > 20:  # B01 / B09 à 60 m : sur-échantillonnage
            rs = Resampling.nearest if categorical else Resampling.bilinear
        return src.read(1, window=win, out_shape=grid.shape, resampling=rs,
                        boundless=True, fill_value=0)


def clear_score(scl: np.ndarray) -> tuple[float, float]:
    valid = ~np.isin(scl, SCL_NODATA)
    clear = valid & ~np.isin(scl, SCL_CLOUD)
    return float(valid.mean()), float(clear.mean())


def search(bbox, period: str):
    from pystac_client import Client

    items = Client.open(STAC_URL).search(collections=[COLLECTION], bbox=bbox, datetime=period,
                                         query={"eo:cloud_cover": {"lt": 70}},
                                         max_items=60).item_collection()
    return list(items)


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="melloula", choices=sorted(PRESETS))
    p.add_argument("--center", help="lon,lat : impose le centre de la fenêtre au lieu du dNBR")
    p.add_argument("--size", type=int, default=512, help="côté de la fenêtre (multiple de 32)")
    p.add_argument("--no-harmonize", action="store_true",
                   help="garder le décalage +1000 des traitements récents")
    p.add_argument("--out", default="data/scenarios/burn")
    a = p.parse_args()
    cfg = PRESETS[a.preset]
    if a.size % 32:
        sys.exit("--size doit être un multiple de 32")

    import rasterio

    with rasterio.Env(**GDAL_ENV):
        log(f"1/4 Recherche des scènes Sentinel-2 ({STAC_URL})")
        pre_items, post_items = search(cfg["bbox"], cfg["pre"]), search(cfg["bbox"], cfg["post"])
        log(f"    {len(pre_items)} scènes avant, {len(post_items)} scènes après")
        if not pre_items or not post_items:
            sys.exit("Aucune scène trouvée : élargissez les périodes du préréglage.")

        log("2/4 Nuages sur la zone (masque SCL), par tuile")
        grids: dict[str, Grid] = {}
        scored: dict[str, dict[str, list]] = {}
        for when, items in (("pre", pre_items), ("post", post_items)):
            for it in items:
                tile = _tile(it)
                if tile not in grids:
                    grids[tile] = Grid(it, cfg["bbox"])
                valid, clear = clear_score(read_band(it, "scl", grids[tile], categorical=True))
                log(f"    {when:4s} {it.datetime:%Y-%m-%d} tuile {tile} : "
                    f"{valid:5.0%} couverte, {clear:5.0%} dégagée")
                scored.setdefault(tile, {"pre": [], "post": []})[when].append((clear, it))
        tile, best = max(
            ((t, {k: pick_scene(v[k], k) for k in v}) for t, v in scored.items()
             if v["pre"] and v["post"]),
            key=lambda tb: min(tb[1]["pre"][0], tb[1]["post"][0]))
        pre_it, post_it = best["pre"][1], best["post"][1]
        grid = grids[tile]
        log(f"    retenu : tuile {tile}, avant {pre_it.datetime:%Y-%m-%d} "
            f"({best['pre'][0]:.0%} dégagé), après {post_it.datetime:%Y-%m-%d} "
            f"({best['post'][0]:.0%} dégagé)")

        def bands(item, names, g):
            base, applied = _baseline(item)
            out = {}
            for n in names:
                log(f"    {item.datetime:%Y-%m-%d} {n} ({g.shape[0]} × {g.shape[1]} px)…")
                raw = read_band(item, n, g)
                out[n] = raw if a.no_harmonize else harmonize(raw, base, applied)
            return out

        log("3/4 dNBR sur la zone et choix de la fenêtre")
        size = a.size
        if a.center:
            from rasterio.transform import rowcol
            from rasterio.warp import transform as warp

            lon, lat = (float(v) for v in a.center.split(","))
            xs, ys = warp("EPSG:4326", grid.crs, [lon], [lat])
            r0, c0 = rowcol(grid.transform, xs[0], ys[0])
            row = int(np.clip(r0 - size // 2, 0, max(grid.shape[0] - size, 0)))
            col = int(np.clip(c0 - size // 2, 0, max(grid.shape[1] - size, 0)))
        else:
            pre_b = bands(pre_it, ("B08", "B12"), grid)
            post_b = bands(post_it, ("B08", "B12"), grid)
            scl_pre = read_band(pre_it, "scl", grid, categorical=True)
            scl_post = read_band(post_it, "scl", grid, categorical=True)
            ok = ~np.isin(scl_pre, (*SCL_NODATA, *SCL_CLOUD, SCL_WATER)) & \
                 ~np.isin(scl_post, (*SCL_NODATA, *SCL_CLOUD, SCL_WATER))
            dnbr = nbr(pre_b["B08"], pre_b["B12"]) - nbr(post_b["B08"], post_b["B12"])
            burned = ok & (dnbr > DNBR_THRESHOLD)
            row, col, n = best_window(burned, size)
            log(f"    {burned.sum() * PIXEL_AREA_HA:.0f} ha au-dessus du seuil dNBR sur la zone, "
                f"{n * PIXEL_AREA_HA:.0f} ha dans la meilleure fenêtre")
        h, w = min(size, grid.shape[0]), min(size, grid.shape[1])
        h, w = h - h % 32, w - w % 32
        win = grid.sub(row, col, h, w)

        log(f"4/4 Lecture des 12 bandes ({h} × {w} px à 20 m) avant et après")
        pre_d, post_d = bands(pre_it, S2_BANDS, win), bands(post_it, S2_BANDS, win)
        pre = np.stack([pre_d[b] for b in S2_BANDS], -1)
        post = np.stack([post_d[b] for b in S2_BANDS], -1)
        scl_w = read_band(pre_it, "scl", win, categorical=True), \
            read_band(post_it, "scl", win, categorical=True)

    ok = ~np.isin(scl_w[0], (*SCL_NODATA, *SCL_CLOUD, SCL_WATER)) & \
         ~np.isin(scl_w[1], (*SCL_NODATA, *SCL_CLOUD, SCL_WATER))
    dnbr_w = nbr(pre[..., 7], pre[..., 11]) - nbr(post[..., 7], post[..., 11])
    dnbr_ha = float((ok & (dnbr_w > DNBR_THRESHOLD)).sum()) * PIXEL_AREA_HA
    west, south, east, north = win.lonlat_bounds()

    os.makedirs(a.out, exist_ok=True)
    name = cfg["id"]
    np.savez_compressed(os.path.join(a.out, f"{name}.npz"), pre=pre, post=post)
    entry = {
        "id": name, "file": f"{name}.npz", "fire": name, "label": cfg["label"],
        "burned_area_ha": cfg["reference_ha"], "burned_fraction": None,
        "reference_ha": cfg["reference_ha"], "reference": cfg["reference"],
        "dnbr_area_ha": round(dnbr_ha, 1), "shape": [h, w],
        "pre_date": f"{pre_it.datetime:%Y-%m-%d}", "post_date": f"{post_it.datetime:%Y-%m-%d}",
        "center": [round((west + east) / 2, 5), round((south + north) / 2, 5)],
        "bounds": [round(v, 5) for v in (west, south, east, north)],
        "source": f"Copernicus Sentinel-2 L2A (ESA), tuile {tile}, via Earth Search",
        "stac_items": [pre_it.id, post_it.id],
    }
    path = os.path.join(a.out, "index.json")
    index = {"source": "CaBuAr — Sentinel-2 L2A, Californie (CAL FIRE)", "samples": []}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            index = json.load(fh)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(merge_index(index, entry), fh, indent=2, ensure_ascii=False)
    log(f"\nÉchantillon « {name} » ajouté dans {a.out} : fenêtre centrée sur "
        f"{entry['center'][1]:.4f} N, {entry['center'][0]:.4f} E "
        f"(seuillage dNBR : {dnbr_ha:.0f} ha). Rechargez l'onglet Surfaces brûlées.")


if __name__ == "__main__":
    main()
