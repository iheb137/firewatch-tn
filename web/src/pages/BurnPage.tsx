import { Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, png, type BurnResult, type BurnSample } from "../api";
import { ErrorNote, Loading, Metric, PageHeader, Panel, Toggle } from "../components/ui";
import { fmt, pct } from "../lib/grid";

/** Transforme un masque PNG blanc/noir en calque : remplissage léger et contour opaque. */
function useTintedMask(b64: string | undefined, rgb: [number, number, number], fillAlpha: number) {
  const [url, setUrl] = useState<string>();
  useEffect(() => {
    if (!b64) {
      setUrl(undefined);
      return;
    }
    const img = new Image();
    img.onload = () => {
      const cv = document.createElement("canvas");
      const w = (cv.width = img.width);
      const h = (cv.height = img.height);
      const ctx = cv.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      const d = ctx.getImageData(0, 0, w, h);
      const on = new Uint8Array(w * h);
      for (let i = 0; i < w * h; i++) on[i] = d.data[i * 4] > 127 ? 1 : 0;
      const R = 2; // épaisseur du contour, en pixels
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          let edge = false;
          if (on[i]) {
            for (let k = 1; k <= R && !edge; k++) {
              edge = x < k || y < k || x >= w - k || y >= h - k ||
                !on[i - k] || !on[i + k] || !on[i - k * w] || !on[i + k * w];
            }
          }
          d.data[i * 4] = rgb[0];
          d.data[i * 4 + 1] = rgb[1];
          d.data[i * 4 + 2] = rgb[2];
          d.data[i * 4 + 3] = on[i] ? (edge ? 255 : fillAlpha) : 0;
        }
      }
      ctx.putImageData(d, 0, 0);
      setUrl(cv.toDataURL());
    };
    img.src = png(b64)!;
  }, [b64, rgb, fillAlpha]);
  return url;
}

/** « cabuar-f0-<incendie>_<n> » → « Incendie A », ou « Incendie A · zone n » si plusieurs zones. */
function sampleLabels(samples: BurnSample[]): Map<string, string> {
  const fireOf = (s: BurnSample) => s.fire ?? s.id.replace(/^cabuar-f\d+-/, "").replace(/_\d+$/, "");
  const letters = new Map<string, string>();
  const count = new Map<string, number>();
  for (const s of samples) {
    if (s.label) continue;
    const f = fireOf(s);
    if (!letters.has(f)) letters.set(f, String.fromCharCode(65 + (letters.size % 26)));
    count.set(f, (count.get(f) ?? 0) + 1);
  }
  return new Map(samples.map((s) => {
    if (s.label) return [s.id, s.label];
    const f = fireOf(s);
    const zone = s.id.match(/_(\d+)$/)?.[1];
    const base = `Incendie ${letters.get(f)}`;
    return [s.id, count.get(f)! > 1 && zone ? `${base} · zone ${zone}` : base];
  }));
}

const TRUTH_RGB: [number, number, number] = [184, 212, 168];

export default function BurnPage() {
  const [samples, setSamples] = useState<BurnSample[] | null>(null);
  const [source, setSource] = useState("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [result, setResult] = useState<BurnResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [swipe, setSwipe] = useState(50);
  const [showPred, setShowPred] = useState(true);
  const [showTruth, setShowTruth] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const truthLayer = useTintedMask(result?.truth?.mask_png_base64, TRUTH_RGB, 60);
  const labels = useMemo(() => sampleLabels(samples ?? []), [samples]);

  useEffect(() => {
    api.burnSamples()
      .then((r) => {
        setSamples(r.samples);
        setSource(r.source);
        if (r.samples.length) setSelected(r.samples[0].id);
      })
      .catch((e: Error) => setLoadError(e.message));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    api.burnSample(selected)
      .then(setResult)
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  }, [selected]);

  const upload = async (file: File) => {
    setBusy(true);
    setError(null);
    setSelected(null);
    try {
      setResult(await api.burnUpload(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const pre = png(result?.pre_rgb_png_base64);
  const post = png(result?.post_rgb_png_base64);
  const truth = result?.truth;
  const sample = result?.sample;
  const official = !truth && sample?.reference_ha ? sample.reference_ha : null;
  const areaError = truth ? Math.abs(result!.burned_area_ha - truth.burned_area_ha) / Math.max(truth.burned_area_ha, 1e-6) : null;

  return (
    <div className="p-6 md:p-10">
      <PageHeader title="Combien d'hectares ont brûlé ?">
        Le réseau PyTorch compare deux images satellite Sentinel-2, avant et après l'incendie, et
        délimite la zone brûlée pixel par pixel (20 m, soit 0,04 ha par pixel).
      </PageHeader>

      {loadError && (
        <ErrorNote title="La passerelle API ne répond pas">Détail : {loadError}</ErrorNote>
      )}

      <div className="grid gap-6 xl:grid-cols-[220px_minmax(0,640px)_1fr]">
        <Panel title="Images" className="order-2 xl:order-none">
          {!samples && !loadError && <Loading />}
          {samples?.length === 0 && (
            <p className="text-sm text-lichen">
              Aucun exemple installé. Lancez <code>make scenarios-synthetic</code> ou exportez des patchs
              CaBuAr depuis Kaggle.
            </p>
          )}
          <ul className="space-y-1">
            {samples?.map((s) => (
              <li key={s.id}>
                <button
                  type="button"
                  onClick={() => setSelected(s.id)}
                  aria-pressed={s.id === selected}
                  className={`w-full rounded-lg px-2.5 py-2 text-left text-sm ${s.id === selected ? "bg-canopy-2" : "text-lichen hover:bg-canopy-2/60"}`}
                >
                  <span className="block font-medium text-paper" title={s.id}>{labels.get(s.id)}</span>
                  <span className="tabular block text-xs">
                    {s.reference_ha
                      ? `≈ ${fmt(s.reference_ha)} ha annoncés · Tunisie`
                      : `${fmt(s.burned_area_ha)} ha brûlés (${pct(s.burned_fraction ?? 0)})`}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          {source && <p className="mt-3 border-t border-rule pt-3 text-xs text-moss">{source}</p>}
          <div className="mt-4">
            <input
              ref={fileInput}
              type="file"
              accept=".npz"
              className="sr-only"
              onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
            />
            <button
              type="button"
              onClick={() => fileInput.current?.click()}
              className="flex w-full items-center justify-center gap-2 rounded-lg border border-rule px-3 py-2 text-sm hover:bg-canopy-2"
            >
              <Upload size={16} aria-hidden /> Analyser un fichier .npz
            </button>
            <p className="mt-2 text-xs text-moss">Tableaux <code>pre</code> et <code>post</code> de forme (H, W, 12).</p>
          </div>
        </Panel>

        <div className="order-1 xl:order-none">
          <div className="relative aspect-square overflow-hidden rounded-xl border border-rule bg-canopy">
            {busy && <div className="absolute inset-0 z-10 grid place-items-center bg-night/60"><Loading>Segmentation en cours…</Loading></div>}
            {error && !busy && <div className="absolute inset-4 z-10"><ErrorNote title="L'analyse a échoué">{error}</ErrorNote></div>}
            {post && (
              <>
                <img src={post} alt="Image après incendie" className="absolute inset-0 size-full object-cover" />
                {pre && (
                  <img
                    src={pre}
                    alt="Image avant incendie"
                    className="absolute inset-0 size-full object-cover"
                    style={{ clipPath: `inset(0 ${100 - swipe}% 0 0)` }}
                  />
                )}
                {showPred && result?.overlay_png_base64 && (
                  <img src={png(result.overlay_png_base64)} alt="" className="absolute inset-0 size-full object-cover"
                    style={{ clipPath: `inset(0 0 0 ${pre ? swipe : 0}%)` }} />
                )}
                {showTruth && truthLayer && (
                  <img src={truthLayer} alt="" className="absolute inset-0 size-full object-cover"
                    style={{ clipPath: `inset(0 0 0 ${pre ? swipe : 0}%)` }} />
                )}
                {pre && (
                  <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-paper" style={{ left: `${swipe}%` }} aria-hidden>
                    <span className="absolute left-1/2 top-3 -translate-x-1/2 whitespace-nowrap rounded bg-night/80 px-2 py-0.5 text-xs">
                      avant | après
                    </span>
                  </div>
                )}
              </>
            )}
          </div>
          {pre && (
            <label className="mt-4 block text-sm text-lichen">
              Glissez pour comparer avant et après
              <input type="range" min={0} max={100} value={swipe} onChange={(e) => setSwipe(Number(e.target.value))} className="mt-1 w-full" />
            </label>
          )}
          <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-xs text-lichen" aria-label="Légende">
            <li className="flex items-center gap-2"><span className="size-3 rounded-sm border-2 border-[#ef4423] bg-[#ef4423]/30" /> Zone brûlée prédite</li>
            {truth && <li className="flex items-center gap-2"><span className="size-3 rounded-sm border-2 border-sage bg-sage/25" /> Zone brûlée réelle (CAL FIRE)</li>}
          </ul>
        </div>

        <div className="order-3 space-y-4 xl:order-none">
          <Panel title="Surface brûlée">
            {result ? (
              <dl className="grid grid-cols-2 gap-3">
                <Metric label="Estimée par le modèle" value={`${fmt(result.burned_area_ha)} ha`} tone="ember" />
                {truth && <Metric label="Mesurée sur le terrain" value={`${fmt(truth.burned_area_ha)} ha`} tone="sage" />}
                {areaError !== null && <Metric label="Écart de surface" value={pct(areaError, 1)} />}
                {truth && <Metric label="IoU" value={pct(truth.iou)} hint={`précision ${pct(truth.precision)}, rappel ${pct(truth.recall)}`} />}
                {official !== null && (
                  <>
                    {sample?.dnbr_area_ha !== undefined && (
                      <Metric label="Seuillage dNBR (sans IA)" value={`${fmt(sample.dnbr_area_ha)} ha`}
                        hint={`écart IA / dNBR : ${pct(Math.abs(result.burned_area_ha - sample.dnbr_area_ha) / sample.dnbr_area_ha, 0)}`} />
                    )}
                    <Metric label="Bilan officiel provisoire" value={`≈ ${fmt(official)} ha`} tone="sage"
                      hint="annoncé pendant les feux" />
                  </>
                )}
              </dl>
            ) : busy ? <Loading /> : <p className="text-sm text-lichen">Choisissez une image.</p>}
          </Panel>
          <Panel title="Affichage">
            <Toggle checked={showPred} onChange={setShowPred}>Zone brûlée prédite</Toggle>
            {truth && <Toggle checked={showTruth} onChange={setShowTruth}>Zone brûlée réelle</Toggle>}
          </Panel>
          {official !== null && sample ? (
            <Panel title="Cas d'étude">
              <p className="text-sm leading-relaxed">{sample.reference}</p>
              <dl className="mt-3 grid grid-cols-2 gap-3 text-xs">
                <div><dt className="text-lichen">Image avant</dt><dd className="tabular">{sample.pre_date}</dd></div>
                <div><dt className="text-lichen">Image après</dt><dd className="tabular">{sample.post_date}</dd></div>
              </dl>
              <p className="mt-3 text-xs leading-relaxed text-moss">
                Fenêtre de {fmt((sample.shape[0] * 20) / 1000, 1)} × {fmt((sample.shape[1] * 20) / 1000, 1)} km
                placée automatiquement sur la plus forte baisse de végétation (dNBR). Sans vérité
                terrain au pixel près, on compare l'IA à une méthode indépendante (dNBR) et au bilan
                officiel. {sample.source}.
              </p>
            </Panel>
          ) : (
            <p className="text-xs leading-relaxed text-moss">
              Choisissez « Melloula » pour voir le même calcul sur un incendie tunisien réel,
              comparé à une méthode indépendante et au bilan des services forestiers.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
