import { FlaskConical, Flame, Layers, Map as MapIcon } from "lucide-react";
import { lazy, Suspense, useEffect, useState, type ComponentType } from "react";
import { api, type Health } from "./api";
import BurnPage from "./pages/BurnPage";
import ModelsPage from "./pages/ModelsPage";
import SpreadPage from "./pages/SpreadPage";
import { Loading } from "./components/ui";

// MapLibre est lourd : chargé seulement à l'ouverture de la carte.
const MapPage = lazy(() => import("./pages/MapPage"));

interface Route {
  path: string;
  label: string;
  icon: ComponentType<{ size?: number; strokeWidth?: number }>;
  page: ComponentType;
}

const ROUTES: Route[] = [
  { path: "carte", label: "Carte", icon: MapIcon, page: MapPage },
  { path: "propagation", label: "Propagation", icon: Flame, page: SpreadPage },
  { path: "surfaces", label: "Surfaces brûlées", icon: Layers, page: BurnPage },
  { path: "modeles", label: "Modèles", icon: FlaskConical, page: ModelsPage },
];

function useHashRoute(): string {
  const read = () => window.location.hash.replace(/^#\/?/, "") || "carte";
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const onChange = () => setRoute(read());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

function useHealth(): Health | null | "down" {
  const [health, setHealth] = useState<Health | null | "down">(null);
  useEffect(() => {
    let alive = true;
    const poll = () => api.health().then((h) => alive && setHealth(h)).catch(() => alive && setHealth("down"));
    poll();
    const id = window.setInterval(poll, 15000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);
  return health;
}

function ServiceDot({ label, ok }: { label: string; ok: boolean | null }) {
  const color = ok === null ? "bg-moss" : ok ? "bg-sage" : "bg-ember-deep";
  const state = ok === null ? "vérification…" : ok ? "modèle chargé" : "indisponible";
  return (
    <li className="flex items-center gap-2" title={`${label} : ${state}`}>
      <span className={`size-2 rounded-full ${color}`} aria-hidden />
      <span className="text-xs text-lichen">{label}</span>
      <span className="sr-only">{state}</span>
    </li>
  );
}

export default function App() {
  const route = useHashRoute();
  const health = useHealth();
  const current = ROUTES.find((r) => r.path === route) ?? ROUTES[0];
  const Page = current.page;
  const ok = (s?: { status: string; model_loaded?: boolean }) =>
    health === null ? null : health === "down" ? false : s?.status === "ok" && !!s.model_loaded;

  return (
    <div className="flex h-full flex-col md:flex-row">
      <nav
        aria-label="Navigation principale"
        className="flex shrink-0 items-center gap-1 border-b border-rule bg-canopy px-3 py-2 md:w-56 md:flex-col md:items-stretch md:gap-1 md:border-b-0 md:border-r md:px-3 md:py-5"
      >
        <a href="#/carte" className="mr-3 flex items-center gap-2.5 md:mb-8 md:mr-0 md:px-2">
          <svg viewBox="0 0 32 32" className="size-8 shrink-0" aria-hidden>
            <rect width="32" height="32" rx="8" fill="#0F1E1A" />
            <path d="M16 5c2 5 7 7 7 13a7 7 0 0 1-14 0c0-3 2-5 3-6 0 3 2 4 3 4-1-4 0-8 1-11z" fill="#F0954A" />
          </svg>
          <span className="hidden leading-tight md:block">
            <span className="display block text-base">FireWatch</span>
            <span className="block text-xs text-lichen">Nord-Ouest tunisien</span>
          </span>
        </a>
        {ROUTES.map(({ path, label, icon: Icon }) => {
          const active = path === current.path;
          return (
            <a
              key={path}
              href={`#/${path}`}
              aria-current={active ? "page" : undefined}
              className={`flex items-center gap-3 rounded-lg px-2.5 py-2 text-sm transition-colors ${
                active ? "bg-canopy-2 text-paper" : "text-lichen hover:bg-canopy-2/60 hover:text-paper"
              }`}
            >
              <Icon size={18} strokeWidth={1.8} />
              <span className="hidden sm:inline">{label}</span>
            </a>
          );
        })}
        <ul className="ml-auto hidden gap-3 sm:flex md:ml-0 md:mt-auto md:flex-col md:gap-2 md:px-2" aria-label="État des services">
          <ServiceDot label="Propagation (TensorFlow)" ok={ok(health && health !== "down" ? health.spread : undefined)} />
          <ServiceDot label="Surfaces brûlées (PyTorch)" ok={ok(health && health !== "down" ? health.burn : undefined)} />
        </ul>
      </nav>
      <main className="min-h-0 flex-1 overflow-auto">
        <Suspense fallback={<div className="p-10"><Loading>Chargement de la carte…</Loading></div>}>
          <Page />
        </Suspense>
      </main>
    </div>
  );
}
