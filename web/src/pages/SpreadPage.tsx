import { Pause, Play, Wind } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type SpreadPrediction, type SpreadScenario, type SpreadScenarioSummary } from "../api";
import SpreadStage, { type BaseLayer } from "../components/SpreadStage";
import { ErrorNote, Loading, Metric, PageHeader, Panel, Toggle } from "../components/ui";
import { meanDirection, pct, score } from "../lib/grid";

const MC_SAMPLES = 20;

export default function SpreadPage() {
  const [list, setList] = useState<SpreadScenarioSummary[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [scenario, setScenario] = useState<SpreadScenario | null>(null);
  const [pred, setPred] = useState<SpreadPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [base, setBase] = useState<BaseLayer>("relief");
  const [showTruth, setShowTruth] = useState(true);
  const [withUncertainty, setWithUncertainty] = useState(false);
  const [showUncertainty, setShowUncertainty] = useState(false);
  const [display, setDisplay] = useState(0.5);
  const [playing, setPlaying] = useState(false);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    api.spreadScenarios()
      .then((l) => {
        setList(l);
        if (l.length) setSelected(l[0].id);
      })
      .catch((e: Error) => setListError(e.message));
  }, []);

  const run = useCallback(async (id: string, mc: boolean) => {
    setBusy(true);
    setError(null);
    try {
      const sc = await api.spreadScenario(id);
      setScenario(sc);
      const p = await api.predict(sc.features, mc ? MC_SAMPLES : 0);
      setPred(p);
      setDisplay(p.threshold);
    } catch (e) {
      setError((e as Error).message);
      setPred(null);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    if (selected) run(selected, withUncertainty);
  }, [selected, withUncertainty, run]);

  useEffect(() => {
    if (!withUncertainty) setShowUncertainty(false);
  }, [withUncertainty]);

  // Lecture : le feu prédit « s'allume » des zones les plus sûres vers le seuil du modèle.
  const play = () => {
    if (!pred) return;
    if (playing) {
      setPlaying(false);
      if (raf.current) cancelAnimationFrame(raf.current);
      return;
    }
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      setDisplay(pred.threshold);
      return;
    }
    setPlaying(true);
    const start = performance.now();
    const from = 0.95;
    const to = pred.threshold;
    const step = (now: number) => {
      const t = Math.min((now - start) / 3200, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(from + (to - from) * eased);
      if (t < 1) raf.current = requestAnimationFrame(step);
      else setPlaying(false);
    };
    raf.current = requestAnimationFrame(step);
  };
  useEffect(() => () => { if (raf.current) cancelAnimationFrame(raf.current); }, []);

  const scores = useMemo(() => {
    if (!scenario || !pred) return null;
    const model = score((y, x) => pred.probability[y][x] >= pred.threshold, scenario.truth);
    const persistence = score((y, x) => scenario.features.PrevFireMask[y][x] === 1, scenario.truth);
    return { model, persistence };
  }, [scenario, pred]);

  const wind = useMemo(() => {
    if (!scenario) return { from: 0, speed: 0 };
    const vs = scenario.features.vs.flat();
    return { from: meanDirection(scenario.features.th), speed: vs.reduce((a, b) => a + b, 0) / vs.length };
  }, [scenario]);

  if (listError) {
    return (
      <div className="p-6 md:p-10">
        <PageHeader title="Où le feu sera-t-il demain ?" />
        <ErrorNote title="La passerelle API ne répond pas">
          Vérifiez que la plateforme tourne : <code>docker compose -f deploy/docker-compose.yml ps</code>.
          Détail : {listError}
        </ErrorNote>
      </div>
    );
  }

  return (
    <div className="p-6 md:p-10">
      <PageHeader title="Où le feu sera-t-il demain ?">
        Le réseau TensorFlow lit l'état d'un incendie au jour J (front de feu, vent, température,
        sécheresse, végétation, relief) sur une zone de 64 × 64 km et estime, pour chaque km², la
        probabilité qu'il brûle le lendemain.
      </PageHeader>

      {list && list.length === 0 && (
        <ErrorNote title="Aucun scénario installé">
          Générez des scénarios de démonstration avec <code>make scenarios-synthetic</code>, ou exportez
          de vraies tuiles depuis Kaggle (voir le README), puis rechargez la page.
        </ErrorNote>
      )}
      {!list && <Loading>Chargement des scénarios…</Loading>}

      {list && list.length > 0 && (
        <div className="grid gap-6 xl:grid-cols-[220px_minmax(0,640px)_1fr]">
          <Panel title="Incendies" className="order-2 max-h-[50vh] overflow-auto xl:order-none xl:max-h-[calc(100vh-220px)]">
            <ul className="space-y-1">
              {list.map((s) => {
                const active = s.id === selected;
                return (
                  <li key={s.id}>
                    <button
                      type="button"
                      onClick={() => setSelected(s.id)}
                      aria-pressed={active}
                      className={`w-full rounded-lg px-2.5 py-2 text-left text-sm transition-colors ${
                        active ? "bg-canopy-2 text-paper" : "text-lichen hover:bg-canopy-2/60"
                      }`}
                    >
                      <span className="block font-medium text-paper">Tuile {s.id}</span>
                      <span className="tabular block text-xs">
                        {s.stats.fire_today_px ?? 0} km² en feu, +{s.stats.new_fire_px ?? 0} le lendemain
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
            <p className="mt-3 border-t border-rule pt-3 text-xs text-moss">{list[0].source}</p>
          </Panel>

          <div className="order-1 xl:order-none">
            {scenario && pred ? (
              <SpreadStage
                features={scenario.features}
                probability={pred.probability}
                uncertainty={pred.uncertainty}
                truth={scenario.truth}
                displayThreshold={display}
                base={base}
                showTruth={showTruth}
                showUncertainty={showUncertainty}
                windFromDeg={wind.from}
                windSpeed={wind.speed}
              />
            ) : (
              <div className="grid aspect-square place-items-center rounded-xl border border-rule bg-canopy">
                {error ? <ErrorNote title="La prédiction a échoué">{error}</ErrorNote> : <Loading>Prédiction en cours…</Loading>}
              </div>
            )}

            {pred && (
              <div className="mt-4 flex items-center gap-4">
                <button
                  type="button"
                  onClick={play}
                  className="flex shrink-0 items-center gap-2 rounded-lg bg-paper px-3 py-2 text-sm font-semibold text-night hover:bg-sage"
                >
                  {playing ? <Pause size={16} aria-hidden /> : <Play size={16} aria-hidden />}
                  {playing ? "Pause" : "Lire la prédiction"}
                </button>
                <label className="flex-1 text-sm">
                  <span className="tabular text-lichen">
                    Zones où la probabilité de feu dépasse {pct(display)}
                  </span>
                  <input
                    type="range"
                    min={0.02}
                    max={0.95}
                    step={0.01}
                    value={display}
                    onChange={(e) => setDisplay(Number(e.target.value))}
                    className="mt-1 w-full"
                  />
                </label>
              </div>
            )}

            <ul className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-xs text-lichen" aria-label="Légende">
              <li className="flex items-center gap-2"><span className="size-3 rounded-sm bg-[#fff4d6]" /> Front de feu au jour J</li>
              <li className="flex items-center gap-2"><span className="h-3 w-8 rounded-sm bg-gradient-to-r from-[#6e1a16] via-[#e87034] to-[#f8a04e]" /> Probabilité de feu à J+1</li>
              <li className="flex items-center gap-2"><span className="size-3 rounded-sm border-2 border-sage" /> Feu réellement observé à J+1</li>
              {showUncertainty && <li className="flex items-center gap-2"><span className="h-3 w-8 rounded-sm bg-gradient-to-r from-[#3c7896] to-[#d2f0fa]" /> Incertitude du modèle</li>}
            </ul>
          </div>

          <div className="order-3 space-y-4 xl:order-none">
            <Panel title="Sur cette tuile">
              {scores ? (
                <>
                  <dl className="grid grid-cols-3 gap-3">
                    <Metric label="Rappel" value={pct(scores.model.recall)} tone="ember"
                      hint={`persistance ${pct(scores.persistence.recall)}`} />
                    <Metric label="Précision" value={pct(scores.model.precision)}
                      hint={`persistance ${pct(scores.persistence.precision)}`} />
                    <Metric label="IoU" value={pct(scores.model.iou)}
                      hint={`persistance ${pct(scores.persistence.iou)}`} />
                  </dl>
                  <p className="mt-3 text-xs leading-relaxed text-moss">
                    Rappel : part du feu réel de J+1 que le modèle a anticipée, au seuil de décision
                    {" "}{pct(pred!.threshold)} choisi sur la validation.
                  </p>
                </>
              ) : busy ? <Loading /> : <p className="text-sm text-lichen">Choisissez un incendie.</p>}
            </Panel>

            <Panel title="Affichage">
              <fieldset className="mb-2">
                <legend className="mb-1 text-xs text-lichen">Fond de carte</legend>
                <div className="flex gap-2">
                  {(["relief", "vegetation"] as const).map((b) => (
                    <label key={b} className={`cursor-pointer rounded-lg border px-3 py-1.5 text-sm ${base === b ? "border-sage bg-canopy-2" : "border-rule text-lichen"}`}>
                      <input type="radio" name="base" value={b} checked={base === b} onChange={() => setBase(b)} className="sr-only" />
                      {b === "relief" ? "Relief" : "Végétation (NDVI)"}
                    </label>
                  ))}
                </div>
              </fieldset>
              <Toggle checked={showTruth} onChange={setShowTruth}>Comparer au feu réel</Toggle>
              <Toggle checked={withUncertainty} onChange={setWithUncertainty}>
                Calculer l'incertitude ({MC_SAMPLES} passes MC Dropout)
              </Toggle>
              {withUncertainty && (
                <Toggle checked={showUncertainty} onChange={setShowUncertainty}>Afficher la carte d'incertitude</Toggle>
              )}
            </Panel>

            <Panel title="Conditions au jour J">
              <dl className="grid grid-cols-2 gap-3">
                <Metric label="Vent moyen" value={`${wind.speed.toFixed(1)} m/s`} />
                <Metric label="Direction d'origine" value={`${Math.round(wind.from)}°`} />
              </dl>
              <p className="mt-3 flex gap-2 text-xs leading-relaxed text-moss">
                <Wind size={14} className="mt-0.5 shrink-0" aria-hidden />
                La flèche sur la carte indique où le vent pousse le feu.
              </p>
              {pred?.model_run && <p className="mt-2 text-xs text-moss">Modèle : {pred.model_run}</p>}
            </Panel>
          </div>
        </div>
      )}
    </div>
  );
}
