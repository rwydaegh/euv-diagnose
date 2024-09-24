import { describe, expect, it } from "vitest";
import { csvRows, extent, pathData } from "./math";
describe("scientific plot and export helpers", () => {
  it("preserves negative measured intensities in plot domain", () =>
    expect(extent([-0.02, 0.3])[0]).toBeLessThan(-0.02));
  it("rejects nonfinite data instead of hiding it", () =>
    expect(() => extent([NaN])).toThrow());
  it("maps a two-point line to the plot boundaries", () =>
    expect(pathData([12, 14], [0, 1], [0, 1])).toBe(
      "M52.00,176.00 L620.00,14.00",
    ));
  it("exports both channels, exact values and angle conventions", () => {
    const csv = csvRows(
      [13.5],
      [6],
      [[[-0.001, 0.6]]],
      [[[0.01, 0.61]]],
      ["observed", "predicted"],
    );
    expect(csv).toContain("angle_from_normal_deg");
    expect(csv).toContain("13.5,6,absorber,-0.001,0.01");
    expect(csv.split("\n")).toHaveLength(4);
  });
});
