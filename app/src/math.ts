export function extent(values: number[]): [number, number] {
  if (!values.length || values.some((v) => !Number.isFinite(v)))
    throw new Error("Plot values must be finite and nonempty.");
  const low = Math.min(...values),
    high = Math.max(...values);
  const pad = (high - low || Math.abs(high) || 1) * 0.09;
  return [Math.min(0, low - pad), high + pad];
}
export function pathData(
  xs: number[],
  ys: number[],
  domain: [number, number],
  width = 640,
  height = 210,
): string {
  if (xs.length !== ys.length || xs.length < 2)
    throw new Error("A curve needs matching arrays with at least two points.");
  const x0 = xs[0],
    span = xs[xs.length - 1] - x0;
  return xs
    .map(
      (x, i) =>
        `${i ? "L" : "M"}${(52 + ((x - x0) / span) * (width - 72)).toFixed(2)},${(14 + ((domain[1] - ys[i]) / (domain[1] - domain[0])) * (height - 48)).toFixed(2)}`,
    )
    .join(" ");
}
export function csvRows(
  wavelengths: number[],
  angles: number[],
  a: number[][][],
  b: number[][][],
  labels: [string, string],
): string {
  const rows = [
    `wavelength_nm,angle_from_normal_deg,channel,${labels[0]},${labels[1]}`,
  ];
  angles.forEach((angle, ai) =>
    wavelengths.forEach((wavelength, wi) =>
      ["absorber", "multilayer"].forEach((channel, ci) =>
        rows.push(
          [wavelength, angle, channel, a[ai][wi][ci], b[ai][wi][ci]].join(","),
        ),
      ),
    ),
  );
  return rows.join("\n") + "\n";
}
