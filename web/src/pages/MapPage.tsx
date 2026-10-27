import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { Mountain } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { ErrorNote, Toggle } from "../components/ui";

// Fond sombre CARTO (sans clé) + relief AWS Terrain Tiles (Terrarium, sans clé).
const STYLE_URL = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const DEM_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";

// Tabarka : coordonnées de la ville (NGA GEOnet Names Server). Melloula et Aïn Sobh sont
// dans la délégation de Tabarka : le repère est indicatif, pas la position exacte des feux.
const TABARKA: [number, number] = [8.75801, 36.954423];

const FACTS = [
  { value: "4,6 M ha", text: "de forêts, soit 34 % du territoire, surtout dans le Nord-Ouest et le Centre-Ouest." },
  { value: "88 feux", text: "et près de 3 000 ha brûlés entre le 1er juin et le 26 juillet 2022." },
  { value: "≈ 450 ha", text: "annoncés pendant les feux de Melloula (Tabarka), juillet 2023. La cicatrice finale mesurée par satellite est environ 3 fois plus grande." },
];

export default function MapPage() {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [terrain, setTerrain] = useState(true);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    if (!container.current) return;
    let m: maplibregl.Map;
    try {
      m = new maplibregl.Map({
      container: container.current,
      style: STYLE_URL,
      center: [8.95, 36.62],
      zoom: 8.4,
      pitch: 58,
      bearing: -18,
      maxPitch: 75,
      attributionControl: { compact: true },
      });
    } catch (e) {
      setFailed(`La carte 3D nécessite WebGL : ${(e as Error).message}`);
      return;
    }
    map.current = m;
    m.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    m.on("error", (e) => {
      const msg = String(e.error?.message ?? "");
      if (msg.match(/Failed to fetch|NetworkError|style/i)) {
        setFailed("Les tuiles viennent de CARTO et d'AWS : vérifiez la connexion Internet, puis rechargez la page.");
      }
    });
    m.on("load", () => {
      const dem = { type: "raster-dem" as const, tiles: [DEM_TILES], tileSize: 256,
                    encoding: "terrarium" as const, maxzoom: 13 };
      m.addSource("dem", { ...dem, attribution: "Relief : AWS Terrain Tiles (Mapzen)" });
      m.addSource("dem-shade", dem);
      m.addLayer({
        id: "hillshade",
        type: "hillshade",
        source: "dem-shade",
        paint: {
          "hillshade-shadow-color": "#07110e",
          "hillshade-highlight-color": "#3d5a4c",
          "hillshade-accent-color": "#1c3328",
          "hillshade-exaggeration": 0.55,
        },
      });
      m.setTerrain({ source: "dem", exaggeration: 1.6 });

      const el = document.createElement("div");
      el.className = "size-4 rounded-full border-2 border-paper bg-ember shadow-[0_0_0_6px_rgba(240,149,74,0.25)]";
      el.setAttribute("aria-label", "Délégation de Tabarka");
      new maplibregl.Marker({ element: el })
        .setLngLat(TABARKA)
        .setPopup(new maplibregl.Popup({ offset: 14, maxWidth: "260px" }).setHTML(
          `<strong>Délégation de Tabarka</strong><br/>
           Incendies de Melloula et d'Aïn Sobh, juillet 2023 : environ 450 ha de pin maritime.
           <br/><span style="color:#9db3a3;font-size:12px">Repère placé sur la ville de Tabarka, position indicative.</span>`))
        .addTo(m);

      // Cas d'étude Sentinel-2 installé : cadre de la fenêtre analysée et lien vers l'analyse.
      api.burnSamples().then(({ samples }) => {
        const cs = samples.find((s) => s.bounds && s.center);
        if (!cs || !m.getStyle()) return;
        const [w, so, e, n] = cs.bounds!;
        m.addSource("case-study", {
          type: "geojson",
          data: { type: "Feature", properties: {}, geometry: { type: "Polygon",
            coordinates: [[[w, so], [e, so], [e, n], [w, n], [w, so]]] } },
        });
        m.addLayer({ id: "case-study-fill", type: "fill", source: "case-study",
                     paint: { "fill-color": "#f0954a", "fill-opacity": 0.12 } });
        m.addLayer({ id: "case-study-line", type: "line", source: "case-study",
                     paint: { "line-color": "#f0954a", "line-width": 2, "line-dasharray": [2, 1.5] } });
        const box = document.createElement("button");
        box.type = "button";
        box.className = "cursor-pointer rounded-md border border-ember bg-night/90 px-2 py-1 text-xs font-semibold text-paper hover:bg-canopy-2";
        box.textContent = "Zone analysée : zoomer";
        const popup = new maplibregl.Popup({ offset: 10, maxWidth: "260px" }).setHTML(
          `<strong>${cs.label ?? cs.id}</strong><br/>
           Images Sentinel-2 du ${cs.pre_date} et du ${cs.post_date}.<br/>
           <a href="#/surfaces" style="color:#f0954a;font-weight:600">Mesurer la surface brûlée →</a>`);
        const marker = new maplibregl.Marker({ element: box, anchor: "bottom" })
          .setLngLat([cs.center![0], n])
          .addTo(m);
        box.addEventListener("click", (ev) => {
          ev.stopPropagation();
          m.fitBounds([[w, so], [e, n]], { padding: 120, pitch: 60, bearing: -20, duration: 2200 });
          m.once("moveend", () => popup.setLngLat([cs.center![0], n]).addTo(m));
        });
        void marker;
      }).catch(() => undefined);
    });
    return () => {
      m.remove();
      map.current = null;
    };
  }, []);

  useEffect(() => {
    const m = map.current;
    if (!m || !m.isStyleLoaded() || !m.getSource("dem")) return;
    m.setTerrain(terrain ? { source: "dem", exaggeration: 1.6 } : null);
    m.easeTo({ pitch: terrain ? 58 : 0, duration: 900 });
  }, [terrain]);

  return (
    <div className="relative h-full min-h-[560px]">
      {/* MapLibre impose position:relative sur son conteneur : c'est le parent qui est absolu. */}
      <div className="absolute inset-0">
        <div ref={container} className="h-full w-full" aria-label="Carte du Nord-Ouest tunisien" role="region" />
      </div>
      <aside className="pointer-events-none absolute inset-x-3 bottom-3 md:inset-x-auto md:left-6 md:top-6 md:bottom-auto md:w-[360px]">
        <div className="pointer-events-auto rounded-2xl border border-rule bg-night/90 p-5 backdrop-blur">
          <h1 className="display text-2xl">Kroumirie et Mogods</h1>
          <p className="mt-2 text-sm leading-relaxed text-lichen">
            Le massif forestier le plus exposé du pays. Chaque été, chaleur et sirocco y transforment
            un départ de feu en front de plusieurs kilomètres.
          </p>
          <dl className="mt-5 space-y-4">
            {FACTS.map((f) => (
              <div key={f.value} className="grid grid-cols-[88px_1fr] items-baseline gap-3">
                <dt className="tabular text-lg font-semibold text-ember">{f.value}</dt>
                <dd className="text-sm leading-snug">{f.text}</dd>
              </div>
            ))}
          </dl>
          <div className="mt-5 border-t border-rule pt-3">
            <Toggle checked={terrain} onChange={setTerrain}>
              <span className="flex items-center gap-2"><Mountain size={16} aria-hidden /> Relief en 3D</span>
            </Toggle>
          </div>
          <p className="mt-3 text-xs leading-relaxed text-moss">
            Sources : Direction générale des forêts, TAP, Kapitalis et L'Économiste Maghrébin (juillet 2023).
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <a href="#/propagation" className="rounded-lg bg-paper px-3 py-2 text-sm font-semibold text-night hover:bg-sage">
              Simuler une propagation
            </a>
            <a href="#/surfaces" className="rounded-lg border border-rule px-3 py-2 text-sm hover:bg-canopy-2">
              Mesurer une surface brûlée
            </a>
          </div>
        </div>
        {failed && (
          <div className="pointer-events-auto mt-3">
            <ErrorNote title="Le fond de carte n'a pas pu être chargé">{failed}</ErrorNote>
          </div>
        )}
      </aside>
    </div>
  );
}
