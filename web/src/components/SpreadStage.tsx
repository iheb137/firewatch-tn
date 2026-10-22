import { useEffect, useMemo, useRef } from "react";
import type { Grid } from "../api";
import { EMBER, FRONT, hillshade, minMax, ramp, RELIEF, VEGETATION, WATER, type RGB } from "../lib/grid";

export type BaseLayer = "relief" | "vegetation";

interface Props {
  features: Record<string, Grid>;
  probability: Grid | null;
  uncertainty: Grid | null;
  truth: Grid;
  displayThreshold: number;
  base: BaseLayer;
  showTruth: boolean;
  showUncertainty: boolean;
  windFromDeg: number;
  windSpeed: number;
}

const SCALE = 8; // 64 px -> 512 px

const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

export default function SpreadStage(p: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const size = p.truth.length;

  // Le fond ne dépend que de la tuile : calculé une fois.
  const baseColors = useMemo(() => {
    const elev = p.features.elevation;
    const [lo, hi] = minMax(elev);
    const shade = hillshade(elev);
    return elev.map((row, y) => row.map((v, x): RGB => {
      if (p.base === "vegetation") {
        const ndvi = (p.features.NDVI[y][x] + 1000) / 10000;
        return mix(ramp(VEGETATION, ndvi), [10, 20, 16], 0.25 * (1 - shade[y][x]));
      }
      return ramp(RELIEF, 0.45 * ((v - lo) / (hi - lo)) + 0.55 * shade[y][x]);
    }));
  }, [p.features, p.base]);

  const maxStd = useMemo(() => {
    if (!p.uncertainty) return 1;
    let m = 1e-6;
    for (const row of p.uncertainty) for (const v of row) m = Math.max(m, v);
    return m;
  }, [p.uncertainty]);

  useEffect(() => {
    const cv = canvas.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(size, size);
    const prev = p.features.PrevFireMask;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let c = baseColors[y][x];
        const prob = p.probability?.[y][x] ?? 0;
        if (p.showUncertainty && p.uncertainty) {
          const u = p.uncertainty[y][x] / maxStd;
          if (u > 0.05) c = mix(c, ramp(WATER, u), Math.min(0.9, 0.25 + u));
        } else if (prob >= p.displayThreshold && prev[y][x] !== 1) {
          c = mix(c, ramp(EMBER, 0.15 + 0.8 * prob), 0.82);
        }
        if (prev[y][x] === 1) c = FRONT; // front de feu au jour J
        else if (prev[y][x] === -1) c = mix(c, [0, 0, 0], 0.35);
        const i = (y * size + x) * 4;
        img.data[i] = c[0];
        img.data[i + 1] = c[1];
        img.data[i + 2] = c[2];
        img.data[i + 3] = 255;
      }
    }
    const tmp = document.createElement("canvas");
    tmp.width = tmp.height = size;
    tmp.getContext("2d")!.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(tmp, 0, 0, size * SCALE, size * SCALE);

    if (p.showTruth) {
      // Contour du feu réellement observé à J+1 (pixels -1 ignorés).
      ctx.strokeStyle = "#b8d4a8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      const on = (yy: number, xx: number) => yy >= 0 && xx >= 0 && yy < size && xx < size && p.truth[yy][xx] === 1;
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          if (!on(y, x)) continue;
          const X = x * SCALE;
          const Y = y * SCALE;
          if (!on(y - 1, x)) { ctx.moveTo(X, Y); ctx.lineTo(X + SCALE, Y); }
          if (!on(y + 1, x)) { ctx.moveTo(X, Y + SCALE); ctx.lineTo(X + SCALE, Y + SCALE); }
          if (!on(y, x - 1)) { ctx.moveTo(X, Y); ctx.lineTo(X, Y + SCALE); }
          if (!on(y, x + 1)) { ctx.moveTo(X + SCALE, Y); ctx.lineTo(X + SCALE, Y + SCALE); }
        }
      }
      ctx.stroke();
    }
  }, [baseColors, p.probability, p.uncertainty, p.truth, p.features, p.displayThreshold, p.showTruth,
      p.showUncertainty, maxStd, size]);

  // Le vent « vient de » windFromDeg : la flèche pointe vers où il pousse le feu.
  const toward = (p.windFromDeg + 180) % 360;

  return (
    <figure className="relative w-full">
      <canvas
        ref={canvas}
        width={size * SCALE}
        height={size * SCALE}
        className="pixelated block aspect-square w-full rounded-xl border border-rule"
        role="img"
        aria-label="Carte de la tuile : front de feu actuel, probabilité de feu à J+1 et contour réel"
      />
      <div className="pointer-events-none absolute right-3 top-3 flex items-center gap-2 rounded-lg bg-night/80 px-2.5 py-1.5 text-xs backdrop-blur">
        <svg viewBox="-12 -12 24 24" className="size-7" style={{ transform: `rotate(${toward}deg)` }} aria-hidden>
          <path d="M0 -10 L6 4 L0 1 L-6 4 Z" fill="#e6ece4" />
        </svg>
        <span className="tabular">
          Vent {p.windSpeed.toFixed(1)} m/s
          <span className="block text-lichen">venant de {Math.round(p.windFromDeg)}°</span>
        </span>
      </div>
      <figcaption className="sr-only">Tuile de 64 × 64 km, un pixel représente 1 km².</figcaption>
    </figure>
  );
}
