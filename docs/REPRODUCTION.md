# Reproduce the v0.1 studies

Run commands from the repository root. The root `pyproject.toml` and `uv.lock` define the CPU Python environment.

The saved results and weights need no retraining. Run the checks and browser demo:

```sh
make install
make check
make browser-install
make e2e
make demo
```

Recompute the measured fits and synthetic learning benchmark:

```sh
uv run python scripts/refit_measurements.py --starts 3 --max-nfev 1200
uv run python scripts/train_benchmark.py
uv run euv-diagnose export
npm --prefix app run build
```

The refit takes roughly two minutes on the recorded CPU. The learning run includes simulation, training, and a slower 32-case, three-start physical comparison; allow several minutes. Seeds, bounds, source hashes, termination flags, and measured runtimes are stored in the result JSON files. Numerical results and timings may vary across dependency versions and hardware. The [model card](MODEL_CARD.md) and [refit notes](REFIT.md) define the evaluation assumptions.

The export command rebuilds `app/public/data/demo.json` from saved experiment results and recomputed synthetic spectra. Rebuild the app after exporting. Serve `app/dist/` over HTTP, including under a subpath; the demo needs no backend. The Python CLI supplies a local HTTP server and expects the repository checkout. The wheel contains neither the research data nor the web app.

These commands require no MATLAB, Octave, GPU, private API key, or new research data download. Small original files and numerical reference fixtures are retained in the checkout. Initial dependency and browser installation requires network access.

Recompute the synthetic phase-ambiguity examples (roughly a minute or two on the current machine):

```sh
uv run python scripts/probe_phase_ambiguity.py --periodic-convention continuous --output-name phase-continuous-probe
uv run python scripts/probe_phase_ambiguity.py --periodic-convention continuous --material-fraction .1 --output-name phase-continuous-tight
uv run python scripts/verify_continuous_witness.py
uv run python scripts/plot_phase_probe.py
```

Each search writes a JSON report and NPZ arrays under `research/results`. It finds a pair of models with a one-degree phase difference under illustrative noise levels and parameter bounds. This pair does not bound the possible phase range. The local Jacobian calculation ignores parameter bounds and does not give a physical confidence interval.

The Python verifier checks the corrected periodic calculation by expanding all 40 periods in full. It does not use the legacy repeated-cell calculation. The figure uses the corrected results.

For a reproduction check of the earlier **legacy** candidates against the original code, install GNU Octave if it is not already on PATH, then run:

```sh
uv run python scripts/verify_phase_witness.py
```

The verifier exports numerical vectors, calls the original MIT-licensed MATLAB functions in Octave, and compares complex amplitudes. It does not deserialize MATLAB function handles into executable functions. The plotter reuses saved candidates/reference fixtures and evaluates additional angles on the same tabulated wavelength grid. It performs no new optimization and does not acquire measurements.

To regenerate the perturbed s/p reference fixtures used in tests:

```sh
octave --no-gui --quiet scripts/generate_optics_reference.m
uv run pytest -q
```

The reference generators use the committed numerical-only `sherwin-input-*.mat` exports.

Source attribution and licensing are recorded in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and the upstream manifests. The native implementation preserves the released phase, empirical roughness correction, etch convention, and tabulated optical factors. `ReleasedModel.load` defaults to the `legacy` periodic convention for saved-fit reproduction; use `dataclasses.replace(model, periodic_convention="continuous")` for new physical inference. The generic transfer-matrix kernel defaults to continuous boundaries. The corrected periodic calculation matches explicit layer expansion and analytic cases; the original shortcut does not. The classical repeat pilot uses the legacy calculation only to reproduce the original results. It does not test corrected-model uncertainty. Agreement with upstream code checks the port, not the physical assumptions. The Python interface is specific to this dataset.

## Historical repeat-scan pilot

```sh
uv run python scripts/audit_provenance.py
uv run python scripts/run_classical_pilot.py
```

The repeat pilot is conditional on the saved initial stack and includes a labeled post-hoc data sensitivity study. It retains the legacy convention for reproduction and is separate from the corrected-model release experiments.
