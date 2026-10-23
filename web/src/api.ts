// Client de la passerelle FastAPI (servie sous /api, relayée par Vite ou nginx).

export type Grid = number[][];

export interface SpreadScenarioSummary {
  id: string;
  source: string;
  stats: {
    fire_today_px?: number;
    fire_tomorrow_px?: number;
    new_fire_px?: number;
    unknown_px?: number;
    wind_from_deg?: number;
    wind_speed_ms?: number;
  };
}

export interface SpreadScenario extends SpreadScenarioSummary {
  features: Record<string, Grid>;
  truth: Grid;
}

export interface SpreadPrediction {
  probability: Grid;
  mask: Grid;
  threshold: number;
  uncertainty: Grid | null;
  model_run: string | null;
}

export interface BurnSample {
  id: string;
  file: string;
  fire?: string;
  burned_fraction: number;
  burned_area_ha: number;
  shape: [number, number];
}

export interface BurnResult {
  burned_area_ha: number;
  burned_fraction: number;
  threshold: number;
  shape: [number, number];
  mask_png_base64: string;
  pre_rgb_png_base64?: string;
  post_rgb_png_base64?: string;
  overlay_png_base64?: string;
  truth?: {
    burned_area_ha: number;
    iou: number;
    precision: number;
    recall: number;
    mask_png_base64: string;
  };
  sample?: BurnSample;
}

export interface Health {
  api: string;
  spread: { status: string; model_loaded?: boolean };
  burn: { status: string; model_loaded?: boolean };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api/v1${path}`, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* corps non JSON */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),
  models: () => request<Record<string, Record<string, unknown>>>("/models"),
  spreadScenarios: () => request<SpreadScenarioSummary[]>("/scenarios/spread"),
  spreadScenario: (id: string) => request<SpreadScenario>(`/scenarios/spread/${encodeURIComponent(id)}`),
  predict: (features: Record<string, Grid>, mcSamples: number) =>
    request<SpreadPrediction>("/spread/predict", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ features, mc_samples: mcSamples }),
    }),
  burnSamples: () => request<{ source: string; samples: BurnSample[] }>("/scenarios/burn"),
  burnSample: (id: string) =>
    request<BurnResult>(`/scenarios/burn/${encodeURIComponent(id)}/segment`, { method: "POST" }),
  burnUpload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<BurnResult>("/burn/segment", { method: "POST", body: form });
  },
};

export const png = (b64?: string) => (b64 ? `data:image/png;base64,${b64}` : undefined);
