import { useEffect, useState } from "react";
import { api } from "../api";
import { PageHeader, Panel } from "../components/ui";
import { BURN_RESULTS, SPREAD_RESULTS } from "../data/results";
import { pct } from "../lib/grid";

function Bar({ value, max, tone }: { value: number; max: number; tone: "ember" | "moss" | "sage" }) {
  const color = { ember: "bg-ember", moss: "bg-moss", sage: "bg-sage" }[tone];
  return (
    <div className="h-2 w-full rounded-full bg-rule/60" aria-hidden>
      <div className={`h-2 rounded-full ${color}`} style={{ width: `${(value / max) * 100}%` }} />
    </div>
  );
}

function Live({ info }: { info?: Record<string, unknown> }) {
  if (!info) return <span className="text-moss">non disponible</span>;
  if (info.loaded === false) return <span className="text-ember">aucun modèle chargé</span>;
  const label = info.run ?? `${info.arch ?? ""} ${info.encoder ?? ""}`;
  return <span className="text-sage">modèle chargé : {String(label).trim()}</span>;
}

export default function ModelsPage() {
  const [models, setModels] = useState<Record<string, Record<string, unknown>>>({});
  useEffect(() => {
    api.models().then(setModels).catch(() => setModels({}));
  }, []);

  return (
    <div className="p-6 md:p-10">
      <PageHeader title="Deux réseaux, deux frameworks">
        Chaque modèle est comparé à une méthode de référence sans apprentissage, sur des données de
        test jamais vues pendant l'entraînement.
      </PageHeader>

      <div className="grid gap-6 2xl:grid-cols-2">
        <Panel>
          <h2 className="display text-lg">Propagation à J+1</h2>
          <p className="mt-1 text-sm text-lichen">TensorFlow / Keras, <Live info={models.spread} /></p>
          <p className="mt-1 text-xs text-moss">{SPREAD_RESULTS.dataset}</p>
          <table className="mt-4 w-full text-sm">
            <caption className="sr-only">Résultats du module de propagation sur le jeu de test</caption>
            <thead className="text-left text-xs text-lichen">
              <tr>
                <th className="pb-2 font-normal">Méthode</th>
                <th className="w-40 pb-2 font-normal">AUC-PR</th>
                <th className="pb-2 text-right font-normal">F1</th>
                <th className="pb-2 text-right font-normal">Rappel</th>
              </tr>
            </thead>
            <tbody className="tabular">
              {SPREAD_RESULTS.rows.map((r) => (
                <tr key={r.name} className="border-t border-rule">
                  <td className={`py-2.5 pr-3 ${r.baseline ? "text-lichen" : ""}`}>
                    {r.name}
                    {r.production && <span className="ml-2 rounded bg-canopy-2 px-1.5 py-0.5 text-xs text-sage">retenu</span>}
                  </td>
                  <td className="py-2.5 pr-3">
                    <div className="flex items-center gap-2">
                      <span className="w-10">{r.aucpr.toFixed(3)}</span>
                      <Bar value={r.aucpr} max={0.3} tone={r.baseline ? "moss" : "ember"} />
                    </div>
                  </td>
                  <td className="py-2.5 text-right">{r.f1.toFixed(3)}</td>
                  <td className="py-2.5 text-right">{pct(r.recall)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <ul className="mt-4 list-disc space-y-1 pl-5 text-xs leading-relaxed text-lichen">
            {SPREAD_RESULTS.notes.map((n) => <li key={n}>{n}</li>)}
          </ul>
        </Panel>

        <Panel>
          <h2 className="display text-lg">Surfaces brûlées</h2>
          <p className="mt-1 text-sm text-lichen">PyTorch, <Live info={models.burn} /></p>
          <p className="mt-1 text-xs text-moss">{BURN_RESULTS.dataset}</p>
          <table className="mt-4 w-full text-sm">
            <caption className="sr-only">Résultats du module de surfaces brûlées sur le jeu de test</caption>
            <thead className="text-left text-xs text-lichen">
              <tr>
                <th className="pb-2 font-normal">Méthode</th>
                <th className="w-40 pb-2 font-normal">IoU</th>
                <th className="pb-2 text-right font-normal">F1</th>
                <th className="pb-2 text-right font-normal">Écart surface</th>
              </tr>
            </thead>
            <tbody className="tabular">
              {BURN_RESULTS.rows.map((r) => (
                <tr key={r.name} className="border-t border-rule">
                  <td className={`py-2.5 pr-3 ${r.baseline ? "text-lichen" : ""}`}>
                    {r.name}
                    {r.production && <span className="ml-2 rounded bg-canopy-2 px-1.5 py-0.5 text-xs text-sage">retenu</span>}
                  </td>
                  <td className="py-2.5 pr-3">
                    <div className="flex items-center gap-2">
                      <span className="w-10">{r.iou.toFixed(3)}</span>
                      <Bar value={r.iou} max={0.7} tone={r.baseline ? "moss" : "sage"} />
                    </div>
                  </td>
                  <td className="py-2.5 text-right">{r.f1.toFixed(3)}</td>
                  <td className="py-2.5 text-right">{pct(r.areaErr, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <ul className="mt-4 list-disc space-y-1 pl-5 text-xs leading-relaxed text-lichen">
            {BURN_RESULTS.notes.map((n) => <li key={n}>{n}</li>)}
          </ul>
        </Panel>
      </div>
    </div>
  );
}
