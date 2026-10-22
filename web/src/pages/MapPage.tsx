import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { Mountain } from "lucide-react";
import { useEffect, useRef, useState } from "react";
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
  { value: "≈ 450 ha", text: "de pin maritime perdus à Melloula (Tabarka) en juillet 2023, sous l'effet du sirocco." },
];

export default function MapPage() {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [terrain, setTerrain] = useState(true);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!container.current) return;
    const m = new maplibregl.Map({
      container: container.current,
      style: STYLE_URL,
      center: [8.95, 36.62],
      zoom: 8.4,
      pitch: 58,
      bearing: -18,
      maxPitch: 75,
      attributionControl: { compact: true },
    });
    map.current = m;
    m.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    m.on("error", (e) => {
      if (String(e.error?.message ?? "").match(/Failed to fetch|NetworkError|style/i)) setFailed(true);
    });
    m.on("load", () => {
      m.addSource("dem", {
        type: "raster-dem",
        tiles: [DEM_TILES],
        tileSize: 256,
        encoding: "terrarium",
        maxzoom: 13,
        attribution: "Relief : AWS Terrain Tiles (Mapzen)",
      });
      m.addLayer({
        id: "hillshade",
        type: "hillshade",
        source: "dem",
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
      <div ref={container} className="absolute inset-0" aria-label="Carte du Nord-Ouest tunisien" role="region" />
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
            <ErrorNote title="Le fond de carte n'a pas pu être chargé">
              Les tuiles viennent de CARTO et d'AWS : vérifiez la connexion Internet, puis rechargez la page.
            </ErrorNote>
          </div>
        )}
      </aside>
    </div>
  );
}
