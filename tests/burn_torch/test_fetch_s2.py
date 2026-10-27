"""Cas d'étude Sentinel-2 : fonctions pures, puis chaîne complète sur de faux COG locaux."""

import datetime as dt
import json
import sys

import numpy as np
import pytest

from ml.burn_torch import fetch_s2 as f


def test_harmonize_removes_offset_only_for_recent_baselines():
    dn = np.array([0, 500, 1500, 3000], np.uint16)
    np.testing.assert_array_equal(f.harmonize(dn, "05.09", False), [0, 1, 500, 2000])
    np.testing.assert_array_equal(f.harmonize(dn, "03.01", False), dn)
    np.testing.assert_array_equal(f.harmonize(dn, "05.09", True), dn)
    np.testing.assert_array_equal(f.harmonize(dn, None, False), dn)


def test_best_window_finds_the_burn_scar():
    score = np.zeros((100, 80), bool)
    score[60:75, 10:30] = True
    score[5, 70] = True
    r, c, n = f.best_window(score, 32)
    assert n == 15 * 20 and r <= 60 and r + 32 >= 75 and c <= 10 and c + 32 >= 30


def test_merge_index_puts_case_study_first_and_replaces_it():
    idx = {"source": "x", "samples": [{"id": "a"}, {"id": "melloula-2023", "old": 1}]}
    out = f.merge_index(idx, {"id": "melloula-2023"})
    assert [s["id"] for s in out["samples"]] == ["melloula-2023", "a"]
    assert "old" not in out["samples"][0]


def _fake_scene(tmp, name, bbox, scar_lonlat, burned):
    """Tuile UTM 32N aux résolutions Sentinel-2 (10, 20, 60 m), décalage +1000 inclus."""
    rasterio = pytest.importorskip("rasterio")
    pystac = pytest.importorskip("pystac")
    from rasterio.transform import from_origin
    from rasterio.warp import transform

    xs, ys = transform("EPSG:4326", "EPSG:32632", [bbox[0], bbox[2]], [bbox[1], bbox[3]])
    x0, y1 = np.floor(min(xs) / 60) * 60 - 1200, np.ceil(max(ys) / 60) * 60 + 1200
    width_m, height_m = 60 * np.ceil((max(xs) - x0 + 1200) / 60), 60 * np.ceil(
        (y1 - min(ys) + 1200) / 60)
    sx, sy = transform("EPSG:4326", "EPSG:32632", [scar_lonlat[0]], [scar_lonlat[1]])
    item = pystac.Item(name, None, None, dt.datetime(2023, 7, 1, tzinfo=dt.timezone.utc),
                       {"s2:mgrs_tile": "32SNF", "s2:processing_baseline": "05.09"})
    for band, key in [*f.ASSETS.items(), ("scl", "scl")]:
        res = 60 if band in ("B01", "B09") else 10 if band in ("B02", "B03", "B04", "B08") else 20
        h, w = int(height_m / res), int(width_m / res)
        yy, xx = np.mgrid[0:h, 0:w]
        px, py = x0 + (xx + 0.5) * res, y1 - (yy + 0.5) * res
        scar = (np.abs(px - sx[0]) < 1500) & (np.abs(py - sy[0]) < 1000) & burned
        val = {"B08": np.where(scar, 1500, 3000), "B12": np.where(scar, 2500, 1000)}.get(
            band, np.full((h, w), 800)) + 1000
        if band == "scl":
            val = np.full((h, w), 4)
        path = tmp / f"{name}_{key}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                           dtype="uint8" if band == "scl" else "uint16", crs="EPSG:32632",
                           transform=from_origin(x0, y1, res, res)) as dst:
            dst.write(val.astype(dst.dtypes[0]), 1)
        item.add_asset(key, pystac.Asset(str(path)))
    return item


def test_fetch_pipeline_on_local_rasters(tmp_path, monkeypatch, capsys):
    bbox = (8.70, 36.88, 8.80, 36.95)
    scar = (8.76, 36.90)
    pre = _fake_scene(tmp_path, "pre", bbox, scar, burned=False)
    post = _fake_scene(tmp_path, "post", bbox, scar, burned=True)
    monkeypatch.setitem(f.PRESETS, "test", {**f.PRESETS["melloula"], "bbox": bbox})
    monkeypatch.setattr(f, "search", lambda b, period: [pre] if period == f.PRESETS["test"][
        "pre"] else [post])
    out = tmp_path / "burn"
    out.mkdir()
    (out / "index.json").write_text(json.dumps({"source": "CaBuAr", "samples": [{"id": "a"}]}))
    monkeypatch.setattr(sys, "argv", ["fetch_s2", "--preset", "test", "--size", "128",
                                      "--out", str(out)])
    f.main()
    print(capsys.readouterr().out)

    index = json.loads((out / "index.json").read_text())
    entry = index["samples"][0]
    assert [s["id"] for s in index["samples"]] == ["melloula-2023", "a"]
    assert abs(entry["center"][0] - scar[0]) < 0.02 and abs(entry["center"][1] - scar[1]) < 0.02
    # Cicatrice de 3 × 2 km, fenêtre de 2,56 km : environ 2,56 × 2 km = 512 ha dedans.
    assert 300 < entry["dnbr_area_ha"] <= 128 * 128 * 0.04
    arr = np.load(out / "melloula-2023.npz")
    assert arr["pre"].shape == arr["post"].shape == (128, 128, 12)
    assert int(np.median(arr["pre"][..., 7])) == 3000  # décalage +1000 retiré


def test_pick_scene_prefers_closest_date_among_clear_scenes():
    class It:
        def __init__(self, d):
            self.datetime = dt.datetime(2023, 7, d)

    scenes = [(1.0, It(1)), (0.99, It(7)), (0.60, It(10))]
    assert f.pick_scene(scenes, "pre")[1].datetime.day == 7
    assert f.pick_scene(scenes, "post")[1].datetime.day == 1
