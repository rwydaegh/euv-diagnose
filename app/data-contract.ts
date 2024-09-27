export interface ModelSpectra {
  spectra: number[][][];
  phaseDeg: number;
}
export interface DemoData {
  schemaVersion: 1;
  generatedAt?: string;
  revision?: string;
  synthetic: {
    wavelengths: number[];
    angles: number[];
    nominal: ModelSpectra;
    alternative: ModelSpectra;
    sigma: [number, number];
    phaseDifferenceDeg: number;
    baselineDistance: number;
    distances: number[];
    metadata: {
      title?: string;
      description?: string;
      sourceUrl: string;
      assumptions: string[];
    };
  };
  measured?: {
    label: string;
    wavelengths: number[];
    angles: number[];
    observed: number[][][];
    predicted: number[][][];
    trainingAngles: number[];
    heldoutAngles: number[];
    rmse: [number, number];
    comparisons?: {
      id: string;
      label: string;
      predicted: number[][][];
      phaseDeg: number;
      rmse: [number, number];
      detail: string;
    }[];
    metadata: { sourceUrl: string; description: string; limitations: string[] };
  };
  learning?: {
    title: string;
    description: string;
    metrics: { label: string; value: string; detail?: string }[];
    limitations: string[];
    sourceUrl?: string;
    evaluations?: {
      name: string;
      truth: number[];
      prediction: number[];
      intervalHalfWidthDeg: number;
    }[];
  };
}
