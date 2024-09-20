# Neural phase estimator

This is a trained, CPU-capable estimator for one synthetic EUV mask family. Given 126 reflectivity values, it predicts the relative reflection phase of the absorber and exposed multilayer at 13.5 nm and 6° from normal. It is a compact benchmark for learning a nonlinear optical quantity from intensity measurements.

**The estimator has not been validated on experimental phase measurements. Its uncertainty becomes unreliable under the tested change in material and measurement assumptions.** The measured-data refit and this synthetic learning benchmark are separate results.

## Model and input

Five independently initialized scikit-learn MLP regressors, each with 128 and 64 ReLU hidden units, share the training set. Their predictions are averaged; initialization seeds are 900–904. Training uses Adam, L2 regularization (`alpha=0.1`), batch size 128, and early stopping on an internal 15% training-data split. Input standardization and target scaling use only training data. The architecture and training configuration were fixed before the reported final run; an initial run checked the pipeline before extending the classical comparison.

The input is the full 63-point grid: wavelengths 12.5–14.5 nm in 0.1 nm steps at angles 2°, 5°, and 8°. Each grid row has absorber then multilayer intensity. Flatten in row-major order; the exact row order is stored as `grid` in the checkpoint. There is no sparse-input support. Missing measurements, arbitrary masks, other grids, and other mask families require new training and evaluation.

The target is `arg(r_absorber / r_multilayer)` at a shared reference plane, expressed as a phase offset from the nominal model. The reference angle is 6° even when the acquisition has an angle offset. Targets are wrapped to ±180° about the nominal phase. Scoring uses shortest circular differences. Reported interval endpoints are unwrapped about the prediction; they describe an arc, not an ordering on the full circle. The sampled nominal target range is recorded in the benchmark JSON. This is a scalar regressor, not a joint posterior or a map of plausible film structures.

## Simulation assumptions

The source is the MIT-licensed [Sherwin EUV repository](https://github.com/s-sherwin/EUV), model `Reflectivityapp_workspace_fit131.mat`. The saved model defines this synthetic family's center; it is not an independent experimental test. All simulations use the verified continuous-boundary solver, not the upstream legacy periodic shortcut.

Twenty-two physical parameters vary independently and uniformly:

- Ten thickness/etch parameters use the released fitting bounds.
- Twelve free elemental concentration parameters vary within ±3% of their saved value, intersected with the released bounds. These are concentration parameters in the source optical model, not normalized chemical fractions.

Each acquisition also has independent channel gains uniformly distributed from 0.98 to 1.02 and a shared angle offset from −0.1° to 0.1°. Independent Gaussian intensity errors have standard deviations 0.0001 for absorber and 0.001 for multilayer. Negative noisy intensities are preserved. These bounds and error scales are illustrative, not experimental prior estimates.

The challenge expands the material-concentration ranges to ±8% and adds a correlated wavelength sinusoid with random phase and amplitudes 0.0003/0.003 in the two channels. It changes two assumptions simultaneously; it does not isolate their separate effects or model every possible experimental error.

## Split and calibration contract

| Set | Cases | Seed | Purpose |
|---|---:|---:|---|
| Training | 8,000 | 4101 | Fit network and ridge baseline |
| Calibration | 500 | 4102 | Set interval radii only |
| Nominal test | 500 | 4103 | Evaluate the training regime |
| Shifted challenge | 500 | 4104 | Evaluate the specified mismatch |

The reported intervals use split-conformal absolute circular errors with the finite-sample quantile `ceil((n + 1) × coverage)`. Levels 50%, 80%, 90%, and 95% are saved. Radius is constant across cases; ensemble disagreement is not used as a confidence estimate. The coverage argument is marginal over exchangeable calibration/test cases. It gives no per-case guarantee, no assurance under distribution shift, and no experimental accuracy claim. Wilson 95% intervals quantify finite test-sample uncertainty in observed coverage, not uncertainty in the physical model.

There is no automatic out-of-distribution detector in this version. Supplying a correctly shaped array does not establish that the model is applicable. Physics-checked posterior proposals, case-specific uncertainty, and robust handling of unsupported inputs remain future work. Do not substitute this estimator for the exact physical fit when those capabilities are required.

## Baselines and results

The machine-readable results are in [`learning-benchmark.json`](../research/results/learning-benchmark.json); the comparison figure is [`learning-benchmark.svg`](../research/results/learning-benchmark.svg). They contain nominal and shifted errors, empirical coverage at all four levels, interval widths, and measured runtime.

A ridge regressor uses the same training spectra and targets, with five-fold cross-validation over 17 regularization strengths. It receives its own conformal calibration. A physical least-squares comparison uses 32 nominal-test cases sampled without replacement with seed 715. Each fit has three starts, matched physical/calibration bounds, matched noise weighting, and a 200-evaluation limit per start. The lowest-cost result is retained. Its timing includes all starts. Per-start termination flags are recorded; optimizer success does not prove a global optimum. This point-fit comparison does not compare posterior uncertainty or establish a general neural speedup at matched quality.

The neural model's nominal 90% interval covered 89.4% of 500 test cases, with a width of 6.17°. Coverage fell to 74.6% on the shifted challenge. Nominal phase RMSE was 2.40°, compared with 5.22° for ridge; shifted RMSE was 3.02° and 7.02°, respectively. The narrow nominal intervals must not be presented as trustworthy under mismatch.

On the 32-case physical-fit subset, three-start least squares reached 1.78° RMSE, compared with 2.56° for the neural model and 5.65° for ridge. The physical fit was more accurate on this subset. Its median runtime was 4.93 seconds per case; 31 of 32 retained solutions terminated successfully, and 7 of the 96 individual starts hit the evaluation limit. The neural estimator averaged 0.113 milliseconds per single-case call. This is a speed–accuracy tradeoff, not equal-quality replacement of physical inference.

Simulation took 18.2 seconds and neural training 29.8 seconds in the recorded run. Runtime includes simulation and training costs separately. Prediction latency is an average of 500 single-case calls after warmup with one BLAS thread. Machine and source hashes are recorded. Timing is local evidence, not a hardware-independent guarantee; training cost has not been converted into an equal-quality break-even claim.

## Reproduce and inspect

```bash
uv sync --locked
uv run python scripts/train_benchmark.py
uv run pytest tests/test_learning.py
```

The complete run uses CPU only and no network, credentials, or proprietary software. The checkpoint is [`artifacts/model-phase-ensemble.npz`](../artifacts/model-phase-ensemble.npz). It stores weights, biases, scaling arrays, exact input grid, nominal phase, and calibrated radii in NumPy format. Loading disables pickle.

```python
import numpy as np
from euv_diagnose.learning import PhaseEnsemble

model = PhaseEnsemble.load("artifacts/model-phase-ensemble.npz")
with np.load("research/results/learning-evaluation.npz", allow_pickle=False) as data:
    spectra = data["nominal_spectra"][:1]
print(model.predict(spectra))       # phase offset in degrees
print(model.interval(spectra, .9))  # nominal-regime arc endpoints
```

The included calibration and test arrays make the reported errors inspectable without retraining. Reusing these public tests to tune a new model turns them into development data; generate and disclose fresh final seeds for a new evaluation.

Code and generated weights use the repository MIT license. Upstream attribution remains in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
