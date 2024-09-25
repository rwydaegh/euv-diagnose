/** Every angle is in degrees from normal; wavelength is in nm; spectra are intensity fractions. */
export interface ModelSpectra {
  spectra: number[][][]; // [angle][wavelength][absorber, multilayer]
  phaseDeg: number; // arg(r_absorber/r_multilayer), common reference plane, at 13.5 nm / 6°
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
    baselineDistance: number; // full 2–8° joint whitened Euclidean distance
    distances: number[]; // one per angle; joint two-channel whitened Euclidean distance
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
