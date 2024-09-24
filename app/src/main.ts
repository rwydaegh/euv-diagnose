import "@fontsource-variable/dm-sans";
import "@fontsource-variable/manrope";
import "./style.css";
import type { DemoData } from "../data-contract";
import { csvRows, extent, pathData } from "./math";

const root = document.querySelector<HTMLDivElement>("#app")!;
let data: DemoData;
let tab: "synthetic" | "measured" | "learning" = "synthetic";
let angle = 6;
let channel = 1;
let expanded = false;
let comparison = 0;
let evaluation = 0;
const escape = (s: string) =>
  s.replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
const number = (v: number, digits = 2) =>
  v.toLocaleString("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
const arrow = '<span aria-hidden="true">↗</span>';
const safeLink = (url: string) => (/^https:\/\//.test(url) ? escape(url) : "#");

function plot(
  xs: number[],
  curves: {
    ys: number[];
    color: string;
    dashed?: boolean;
    dots?: boolean;
    label: string;
  }[],
  label: string,
  unit: string,
): string {
  const domain = extent(curves.flatMap((c) => c.ys));
  const ticks = [0, 1, 2, 3, 4].map(
    (i) => domain[0] + ((domain[1] - domain[0]) * i) / 4,
  );
  const grid = ticks
    .map(
      (v, i) =>
        `<line x1="52" x2="620" y1="${176 - i * 40.5}" y2="${176 - i * 40.5}" stroke="#e4e9ed"/><text x="42" y="${180 - i * 40.5}" text-anchor="end">${Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(1) : number(v, Math.abs(domain[1]) < 0.1 ? 3 : 2)}</text>`,
    )
    .join("");
  const xticks = [xs[0], xs[Math.floor(xs.length / 2)], xs[xs.length - 1]]
    .map(
      (x) =>
        `<text x="${52 + ((x - xs[0]) / (xs[xs.length - 1] - xs[0])) * 568}" y="198" text-anchor="middle">${number(x, 1)}</text>`,
    )
    .join("");
  const paths = curves
    .map(
      (c) =>
        `<path d="${pathData(xs, c.ys, domain)}" fill="none" stroke="${c.color}" stroke-width="${c.dots ? 1 : 2.4}" ${c.dashed ? 'stroke-dasharray="7 5"' : ""} opacity="${c.dots ? ".65" : "1"}"/>${
          c.dots
            ? c.ys
                .filter((_, i) => i % 3 === 0)
                .map(
                  (y, i) =>
                    `<circle cx="${52 + ((xs[i * 3] - xs[0]) / (xs[xs.length - 1] - xs[0])) * 568}" cy="${14 + ((domain[1] - y) / (domain[1] - domain[0])) * 162}" r="2" fill="${c.color}"/>`,
                )
                .join("")
            : ""
        }`,
    )
    .join("");
  return `<svg class="plot" viewBox="0 0 640 214" role="img" aria-label="${escape(label)}"><title>${escape(label)}</title><g font-family="system-ui,sans-serif" font-size="10" fill="#647181">${grid}${xticks}<text x="336" y="213" text-anchor="middle">Wavelength / nm</text></g>${paths}</svg><div class="plot-key"><span>${escape(unit)}</span>${curves.map((c) => `<span><i style="background:${c.color}"></i>${escape(c.label)}</span>`).join("")}</div>`;
}

function stack(): string {
  return `<div class="stack-scene" role="img" aria-label="Schematic photomask with absorber, cap, and forty multilayer periods. Layer thicknesses are exaggerated."><div class="beam incoming"></div><div class="beam outgoing"></div><span class="beam-label">EUV · 13.5 nm</span><div class="stack-layers"><div class="surface"></div><div class="absorber"></div><div class="cap"></div><div class="periods"></div><div class="substrate"></div></div><div class="stack-label l1">Absorber <span>TaN</span></div><div class="stack-label l2">Reflector <span>40 × Mo / Si</span></div><div class="stack-label l3">Substrate <span>Si</span></div></div>`;
}
function phase(): string {
  const delta = data.synthetic.phaseDifferenceDeg;
  const wave = (offset: number) =>
    Array.from(
      { length: 121 },
      (_, i) =>
        `${i ? "L" : "M"}${i * 2.5},${38 + 22 * Math.sin((i / 120) * Math.PI * 5 + offset)}`,
    ).join(" ");
  return `<div class="phase-card"><div class="eyebrow">What intensity leaves out</div><div class="phase-number">${number(delta)}<span>°</span></div><p>Different relative reflection phase.<br>Almost the same measured intensity.</p><svg viewBox="0 0 300 78" role="img" aria-label="Illustration of two shifted waves; phase offset exaggerated for visibility"><path d="${wave(0)}" stroke="#54c7bd" fill="none" stroke-width="2"/><path d="${wave(0.45)}" stroke="#ecad72" fill="none" stroke-width="2" stroke-dasharray="5 4"/></svg><span class="fine">Wave offset exaggerated · reference at 13.5 nm, 6°</span></div>`;
}
function angleControls(angles: number[]): string {
  return `<div class="angle-control"><label for="angle">Incidence angle <strong id="angle-value">${angle}°</strong></label><input id="angle" aria-label="Incidence angle" type="range" min="0" max="${angles.length - 1}" value="${Math.max(0, angles.indexOf(angle))}" step="1"/><div class="range-label"><span>${angles[0]}°</span><span>from surface normal</span><span>${angles.at(-1)}°</span></div></div>`;
}
function syntheticView(): string {
  const s = data.synthetic,
    ai = Math.max(0, s.angles.indexOf(angle));
  const a = s.nominal.spectra[ai].map((p) => p[channel]),
    b = s.alternative.spectra[ai].map((p) => p[channel]);
  const diff = a.map((v, i) => (b[i] - v) / s.sigma[channel]);
  return `<div class="workbench"><section class="card spectra-card"><div class="card-top"><div><div class="eyebrow">01 / The observable</div><h2>Two stacks. One reflection?</h2></div><span class="badge amber">Synthetic challenge</span></div><p class="card-intro">These physically computed spectra nearly overlap. Their reflected waves still disagree on phase.</p><div class="channel-switch" role="group" aria-label="Reflectivity channel"><button data-channel="1" aria-pressed="${channel === 1}">Multilayer region</button><button data-channel="0" aria-pressed="${channel === 0}">Absorber region</button></div>${plot(
    s.wavelengths,
    [
      { ys: a, color: "#188c88", label: "Stack A" },
      { ys: b, color: "#cc7536", dashed: true, label: "Stack B" },
    ],
    `${channel ? "Multilayer" : "Absorber"} reflectivity for two synthetic stacks at ${angle} degrees`,
    "Reflectivity / fraction",
  )}<div class="difference-header"><h3>Look closer</h3><span>Difference ÷ assumed noise σ</span></div>${plot(s.wavelengths, [{ ys: diff, color: "#6b7796", label: "(B − A) / σ" }], "Difference between candidate intensities divided by illustrative noise", "Dimensionless")}<div class="plot-note">Saved exact-model calculations. The lines are predictions, not experimental observations.</div></section><aside class="side-column">${phase()}<section class="card measurement"><div class="eyebrow">02 / Ask a better question</div><h2>Change the angle.</h2><p>A different view can make the same two stacks easier to distinguish.</p>${angleControls(s.angles)}<div class="distance"><div><span>Pairwise spectral distance</span><strong>${number(s.distances[ai])}</strong></div><div class="distance-meter"><i style="width:${Math.min(100, (s.distances[ai] / Math.max(...s.distances)) * 100)}%"></i></div><small>Combined absorber + multilayer, at ${angle}°</small></div><button class="primary" id="reveal">${angle === 30 ? "Return to the 6° view" : "Compare at 30°"} <span aria-hidden="true">→</span></button><p class="fine">Same illustrative noise at every angle. This compares two candidates; it does not establish a unique solution.</p></section></aside></div><section class="context-strip"><div><span class="eyebrow">The hidden structure</span><h2>A mirror built from interfaces.</h2><p>Reflections from thin layers interfere. We measure the resulting intensity; the model also predicts the phase.</p></div>${stack()}<div class="context-fact"><strong>40</strong><span>multilayer periods</span><p>Schematic only.<br>Thicknesses exaggerated.</p></div></section>`;
}
function measuredView(): string {
  const m = data.measured;
  if (!m)
    return `<section class="card empty"><h2>Measured case unavailable</h2><p>This bundle does not contain a measured refit.</p></section>`;
  const ai = Math.max(0, m.angles.indexOf(angle));
  const fit = m.comparisons?.[comparison];
  const observed = m.observed[ai].map((p) => p[channel]),
    predicted = (fit?.predicted ?? m.predicted)[ai].map((p) => p[channel]);
  const rmse = fit?.rmse ?? m.rmse;
  return `<div class="workbench"><section class="card spectra-card"><div class="card-top"><div><div class="eyebrow">Experimental case / ${escape(m.label)}</div><h2>The model meets real light.</h2></div><span class="badge teal">Measured EUV data</span></div><p class="card-intro">${escape(m.metadata.description)}</p><div class="channel-switch" role="group" aria-label="Reflectivity channel"><button data-channel="1" aria-pressed="${channel === 1}">Multilayer region</button><button data-channel="0" aria-pressed="${channel === 0}">Absorber region</button></div>${plot(
    m.wavelengths,
    [
      { ys: observed, color: "#188c88", dots: true, label: "Observed" },
      {
        ys: predicted,
        color: "#cc7536",
        dashed: true,
        label: "Refitted model",
      },
    ],
    `Measured and fitted reflectivity at ${angle} degrees`,
    "Reflectivity / fraction",
  )}<div class="difference-header"><h3>Residuals</h3><span>Observed − predicted</span></div>${plot(m.wavelengths, [{ ys: observed.map((v, i) => v - predicted[i]), color: "#6b7796", label: "Residual" }], "Signed reflectivity residuals", "Reflectivity / fraction")}<div class="plot-note">Signed observations are preserved, including negative reduced intensities.</div></section><aside class="side-column"><section class="card measurement"><div class="eyebrow">Inspect the fit</div><h2>${m.heldoutAngles.includes(angle) ? "A held-out angle." : "A fitting angle."}</h2>${m.comparisons?.length ? `<label class="fine" for="comparison">Model assumptions<select id="comparison" class="comparison-select">${m.comparisons.map((f, i) => `<option value="${i}" ${i === comparison ? "selected" : ""}>${escape(f.label)}</option>`).join("")}</select></label><div class="model-phase">Predicted relative phase at 13.5 nm, 6°<strong>${number(fit!.phaseDeg)}°</strong></div><p class="fine">${escape(fit!.detail)}</p>` : ""}${angleControls(m.angles)}<p class="fine">Training angles: ${m.trainingAngles.join(", ")}°<br>Held-out angles: ${m.heldoutAngles.join(", ") || "None"}°</p><div class="metric-row"><span>Absorber RMSE</span><strong>${rmse[0].toExponential(2)}</strong></div><div class="metric-row"><span>Multilayer RMSE</span><strong>${rmse[1].toExponential(2)}</strong></div><p class="fine">RMSE in reflectivity fraction, across the reported evaluation grid.</p></section><section class="dark-note"><div class="eyebrow">What this establishes</div><h2>Prediction, not ground truth.</h2><p>A fit tests compatibility with measured intensity. These data do not independently validate absolute phase or layer thickness.</p></section></aside></div>`;
}
function scatter(): string {
  const evaluations = data.learning?.evaluations;
  if (!evaluations?.length) return "";
  const e = evaluations[evaluation];
  const lo = Math.min(...e.truth, ...e.prediction),
    hi = Math.max(...e.truth, ...e.prediction),
    pad = (hi - lo) * 0.05;
  const low = lo - pad,
    high = hi + pad;
  const x = (v: number) => 55 + ((v - low) / (high - low)) * 540;
  const y = (v: number) => 270 - ((v - low) / (high - low)) * 245;
  return `<div class="scatter-section"><div><h3>Where the estimator succeeds—and misses.</h3><p>Each point is an independently simulated test case. The diagonal marks a perfect phase prediction.</p><div class="channel-switch" role="group" aria-label="Evaluation regime">${evaluations.map((v, i) => `<button data-evaluation="${i}" aria-pressed="${i === evaluation}">${escape(v.name)}</button>`).join("")}</div><p class="fine">Reported interval half-width: ±${number(e.intervalHalfWidthDeg)}°. Marginal synthetic coverage is not a per-case guarantee.</p></div><svg class="scatter" viewBox="0 0 640 320" role="img" aria-label="Predicted versus true phase offsets for ${escape(e.name)}"><title>Predicted versus true phase offsets: ${escape(e.name)}</title>${[
    0, 1, 2, 3, 4,
  ]
    .map((i) => {
      const v = low + ((high - low) * i) / 4;
      return `<line x1="55" x2="595" y1="${y(v)}" y2="${y(v)}" stroke="#e2e9eb"/><text x="46" y="${y(v) + 4}" text-anchor="end" font-size="10" fill="#687984">${number(v, 0)}</text><text x="${x(v)}" y="289" text-anchor="middle" font-size="10" fill="#687984">${number(v, 0)}</text>`;
    })
    .join(
      "",
    )}<path d="M55 270L595 25" stroke="#9cacb4" stroke-dasharray="5 4"/>${e.truth.map((v, i) => `<circle cx="${x(v)}" cy="${y(e.prediction[i])}" r="2.4" fill="${evaluation ? "#c97942" : "#188c88"}" opacity=".45"/>`).join("")}<text x="325" y="312" text-anchor="middle" font-size="11" fill="#687984">True phase offset / °</text><text transform="translate(13 150) rotate(-90)" text-anchor="middle" font-size="11" fill="#687984">Predicted phase offset / °</text></svg></div>`;
}
function learningView(): string {
  const l = data.learning;
  if (!l)
    return `<section class="card empty"><h2>Benchmark unavailable</h2><p>This bundle does not contain a trained-estimator benchmark.</p></section>`;
  return `<section class="card benchmark"><div class="eyebrow">Trained estimator / saved evaluation</div><h2>${escape(l.title)}</h2><p class="card-intro">${escape(l.description)}</p><div class="metric-grid">${l.metrics.map((m) => `<div><span>${escape(m.label)}</span><strong>${escape(m.value)}</strong>${m.detail ? `<p>${escape(m.detail)}</p>` : ""}</div>`).join("")}</div>${scatter()}<div class="benchmark-bottom"><div><h3>Fast is useful only when checked.</h3><p>A trained model learns the specified simulation family. Its performance on that family does not establish accuracy on experimental data.</p></div><div><h3>Evaluation boundaries</h3><ul>${l.limitations.map((x) => `<li>${escape(x)}</li>`).join("")}</ul></div></div></section>`;
}
function render(): void {
  root.innerHTML = `<a class="skip-link" href="#workspace">Skip to explorer</a><header><a class="brand" href="./" aria-label="EUV Diagnose home"><span class="brand-mark" aria-hidden="true">≋</span>EUV<span>DIAGNOSE</span></a><div class="header-right"><span class="status-dot"></span> A reproducible optics experiment <a href="https://github.com/s-sherwin/EUV" target="_blank" rel="noopener noreferrer" aria-label="Original EUV data and code">Source data ${arrow}</a></div></header><main><section class="hero"><div><div class="eyebrow">Computational optics · 12.5–14.5 nm</div><h1>There’s more to light<br>than meets the <em>detector.</em></h1><p>Explore what an EUV reflection reveals about a layered surface—and what an excellent fit can leave uncertain.</p></div><div class="hero-aside"><div class="orbit" aria-hidden="true"><span></span><i></i></div><p>Intensity is measured.<br><strong>Phase is inferred.</strong></p></div></section><div class="workspace-toolbar" id="workspace"><nav class="tabs" aria-label="Experiment"><button data-tab="synthetic" aria-current="${tab === "synthetic" ? "page" : "false"}"><span>01</span> Phase ambiguity</button><button data-tab="measured" aria-current="${tab === "measured" ? "page" : "false"}"><span>02</span> Measured spectra</button><button data-tab="learning" aria-current="${tab === "learning" ? "page" : "false"}"><span>03</span> Learned inference</button></nav><span class="saved-label">◷ Saved, reproducible runs</span></div><div id="experiment">${tab === "synthetic" ? syntheticView() : tab === "measured" ? measuredView() : learningView()}</div><section class="provenance"><button id="provenance-toggle" aria-expanded="${expanded}" aria-controls="provenance-content"><span><span class="eyebrow">Open notebook</span><strong>Assumptions, provenance & exports</strong></span><span aria-hidden="true">${expanded ? "−" : "+"}</span></button><div id="provenance-content" ${expanded ? "" : "hidden"}><div class="provenance-grid"><div><h3>Scope of the result</h3><ul>${(tab === "synthetic" ? data.synthetic.metadata.assumptions : tab === "measured" ? (data.measured?.metadata.limitations ?? []) : (data.learning?.limitations ?? [])).map((s) => `<li>${escape(s)}</li>`).join("")}</ul></div><div><h3>Trace the calculation</h3><p>All plots come from the downloadable result bundle. Changing the angle selects a precomputed physical calculation; no inference runs in your browser.</p><a href="${safeLink(tab === "measured" ? (data.measured?.metadata.sourceUrl ?? "") : data.synthetic.metadata.sourceUrl)}" target="_blank" rel="noopener noreferrer">Original data and model ${arrow}</a><p class="fine">${data.revision ? `Code revision: ${escape(data.revision)}` : ""}${data.generatedAt ? `<br>Generated: ${escape(data.generatedAt)}` : ""}</p><div class="export-buttons"><button id="export-json">Result JSON ↓</button><button id="export-csv" ${tab === "learning" ? "disabled" : ""}>Spectra CSV ↓</button><button id="export-svg" ${tab === "learning" ? "disabled" : ""}>Plot SVG ↓</button></div></div></div></div></section></main><footer><span>EUV Diagnose <span class="footer-sep">/</span> Inspect the uncertainty.</span><span>Research demonstration · No instrument connection</span></footer><div id="announcement" class="sr-only" role="status" aria-live="polite"></div>`;
  bind();
}
function download(content: string, type: string, filename: string): void {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  document.querySelector("#announcement")!.textContent =
    `Downloaded ${filename}`;
}
function bind(): void {
  document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach(
    (b) =>
      (b.onclick = () => {
        tab = b.dataset.tab as typeof tab;
        angle =
          tab === "measured"
            ? data.measured?.angles.includes(6)
              ? 6
              : (data.measured?.angles[0] ?? 6)
            : 6;
        render();
        document
          .querySelector<HTMLButtonElement>(`[data-tab="${tab}"]`)
          ?.focus();
      }),
  );
  document.querySelectorAll<HTMLButtonElement>("[data-channel]").forEach(
    (b) =>
      (b.onclick = () => {
        channel = Number(b.dataset.channel);
        render();
        document
          .querySelector<HTMLButtonElement>(`[data-channel="${channel}"]`)
          ?.focus();
      }),
  );
  const select = document.querySelector<HTMLSelectElement>("#comparison");
  if (select)
    select.onchange = () => {
      comparison = Number(select.value);
      render();
      document.querySelector<HTMLSelectElement>("#comparison")?.focus();
    };
  document.querySelectorAll<HTMLButtonElement>("[data-evaluation]").forEach(
    (b) =>
      (b.onclick = () => {
        evaluation = Number(b.dataset.evaluation);
        render();
        document
          .querySelector<HTMLButtonElement>(`[data-evaluation="${evaluation}"]`)
          ?.focus();
      }),
  );
  const slider = document.querySelector<HTMLInputElement>("#angle");
  if (slider) {
    const commit = () => {
      const angles =
        tab === "measured" ? data.measured!.angles : data.synthetic.angles;
      angle = angles[Number(slider.value)];
      render();
      document.querySelector<HTMLInputElement>("#angle")?.focus();
    };
    slider.oninput = () => {
      const angles =
        tab === "measured" ? data.measured!.angles : data.synthetic.angles;
      document.querySelector("#angle-value")!.textContent =
        `${angles[Number(slider.value)]}°`;
    };
    slider.onchange = commit;
  }

  const reveal = document.querySelector<HTMLButtonElement>("#reveal");
  if (reveal)
    reveal.onclick = () => {
      angle =
        angle === 30
          ? 6
          : data.synthetic.angles.includes(30)
            ? 30
            : data.synthetic.angles.at(-1)!;
      render();
      document.querySelector<HTMLButtonElement>("#reveal")?.focus();
    };
  document.querySelector<HTMLButtonElement>("#provenance-toggle")!.onclick =
    () => {
      expanded = !expanded;
      render();
      document.querySelector<HTMLButtonElement>("#provenance-toggle")!.focus();
    };
  const json = document.querySelector<HTMLButtonElement>("#export-json");
  if (json)
    json.onclick = () =>
      download(
        JSON.stringify(data, null, 2),
        "application/json",
        "euv-diagnose-results.json",
      );
  const csv = document.querySelector<HTMLButtonElement>("#export-csv");
  if (csv)
    csv.onclick = () => {
      const m = data.measured;
      const s = data.synthetic;
      download(
        tab === "measured" && m
          ? csvRows(
              m.wavelengths,
              m.angles,
              m.observed,
              m.comparisons?.[comparison]?.predicted ?? m.predicted,
              ["observed", "predicted"],
            )
          : csvRows(
              s.wavelengths,
              s.angles,
              s.nominal.spectra,
              s.alternative.spectra,
              ["stack_a", "stack_b"],
            ),
        "text/csv",
        "euv-diagnose-spectra.csv",
      );
    };
  const svg = document.querySelector<HTMLButtonElement>("#export-svg");
  if (svg)
    svg.onclick = () => {
      const plot = document
        .querySelector<SVGElement>(".plot")
        ?.cloneNode(true) as SVGElement | undefined;
      if (plot) {
        plot.setAttribute("xmlns", "http://www.w3.org/2000/svg");
        download(
          new XMLSerializer().serializeToString(plot),
          "image/svg+xml",
          "euv-diagnose-spectrum.svg",
        );
      }
    };
}
async function start(): Promise<void> {
  root.innerHTML =
    '<main class="loading"><div class="brand-mark">≋</div><p>Opening the experiment…</p></main>';
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}data/demo.json`);
    if (!response.ok)
      throw new Error(
        `Result bundle could not be loaded (${response.status}).`,
      );
    data = (await response.json()) as DemoData;
    if (data.schemaVersion !== 1 || !data.synthetic?.angles.length)
      throw new Error("The result bundle has an unsupported format.");
    render();
  } catch (error) {
    root.innerHTML = `<main class="loading"><h1>Couldn’t open the experiment.</h1><p>${escape(error instanceof Error ? error.message : "Unknown error")}</p><p>Generate the demo bundle, then reload this page.</p><button onclick="location.reload()">Try again</button></main>`;
  }
}
void start();
