import type { Grid } from "../api";

export type RGB = [number, number, number];

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** Interpolation linéaire dans une palette de couleurs. */
export function ramp(stops: RGB[], t: number): RGB {
  const x = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  const f = x - i;
  return [lerp(stops[i][0], stops[i + 1][0], f), lerp(stops[i][1], stops[i + 1][1], f),
          lerp(stops[i][2], stops[i + 1][2], f)];
}

// Fond : relief en tons de forêt ; feu : braise ; incertitude : eau.
export const RELIEF: RGB[] = [[18, 38, 31], [44, 70, 56], [96, 120, 98], [170, 180, 150]];
export const VEGETATION: RGB[] = [[58, 52, 38], [62, 86, 52], [70, 130, 70], [150, 200, 120]];
export const EMBER: RGB[] = [[110, 26, 22], [186, 56, 34], [232, 112, 52], [248, 160, 78]];
// Front de feu au jour J : blanc chaud, nettement distinct des probabilités.
export const FRONT: RGB = [255, 244, 214];
export const WATER: RGB[] = [[16, 40, 48], [60, 120, 150], [111, 176, 200], [210, 240, 250]];

export function minMax(g: Grid): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const row of g) for (const v of row) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi === lo ? lo + 1 : hi];
}

/** Ombrage simple (lumière nord-ouest) pour donner du relief à l'altitude. */
export function hillshade(elev: Grid): Grid {
  const h = elev.length;
  const w = elev[0].length;
  const out: Grid = [];
  for (let y = 0; y < h; y++) {
    const row: number[] = [];
    for (let x = 0; x < w; x++) {
      const dx = elev[y][Math.min(x + 1, w - 1)] - elev[y][Math.max(x - 1, 0)];
      const dy = elev[Math.min(y + 1, h - 1)][x] - elev[Math.max(y - 1, 0)][x];
      row.push(Math.max(0, Math.min(1, 0.55 - (dx + dy) / 600)));
    }
    out.push(row);
  }
  return out;
}

export interface TileScore {
  precision: number;
  recall: number;
  iou: number;
  tp: number;
}

/** Score d'une prédiction binaire sur une tuile, pixels inconnus (-1) exclus. */
export function score(pred: (y: number, x: number) => boolean, truth: Grid): TileScore {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  truth.forEach((row, y) => row.forEach((t, x) => {
    if (t === -1) return;
    const p = pred(y, x);
    if (p && t === 1) tp++;
    else if (p) fp++;
    else if (t === 1) fn++;
  }));
  return {
    precision: tp + fp ? tp / (tp + fp) : 0,
    recall: tp + fn ? tp / (tp + fn) : 0,
    iou: tp + fp + fn ? tp / (tp + fp + fn) : 0,
    tp,
  };
}

/** Moyenne circulaire d'une grille de directions (degrés). */
export function meanDirection(g: Grid): number {
  let s = 0;
  let c = 0;
  for (const row of g) for (const v of row) {
    s += Math.sin((v * Math.PI) / 180);
    c += Math.cos((v * Math.PI) / 180);
  }
  return ((Math.atan2(s, c) * 180) / Math.PI + 360) % 360;
}

export const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)} %`;
export const fmt = (v: number, digits = 0) =>
  v.toLocaleString("fr-FR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
