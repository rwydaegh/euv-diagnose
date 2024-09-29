# Neural phase estimator

The estimator predicts phase from 126 reflectivity values for one simulated EUV mask family. It runs on a CPU. The target is the relative reflection phase between the absorber and exposed multilayer at 13.5 nm and 6° from normal.

**Experimental phase accuracy has not been tested. The intervals under-cover when the material and measurement assumptions change.** The measured-data refit is a separate experiment.

## Model and input

The estimator averages five scikit-learn MLP regressors trained on the same data. Each has 128 and 64 ReLU hidden units. Initialization seeds are 900–904. Training uses Adam, L2 regularization (`alpha=0.1`), batch size 128, and early stopping on an internal 15% training-data split. Input standardization and target scaling use only training data. The architecture and training configuration were fixed before the reported final run; an initial run checked the pipeline before extending the classical comparison.

The input is the full 63-point grid: wavelengths 12.5–14.5 nm in 0.1 nm steps at angles 2°, 5°, and 8°. Each grid row has absorber then multilayer intensity. Flatten in row-major order; the exact row order is stored as `grid` in the checkpoint. The model requires the complete grid. Missing measurements, arbitrary masks, other grids, and other mask families need new training and evaluation.

The target is `arg(r_absorber / r_multilayer)` at a shared reference plane, expressed as a phase offset from the nominal model. The reference angle is 6° even when the acquisition has an angle offset. Targets are wrapped to ±180° about the nominal phase. Scoring uses shortest circular differences. Reported interval endpoints are unwrapped about the prediction; they describe an arc, not an ordering on the full circle. The sampled nominal target range is recorded in the benchmark JSON. The output is a scalar phase offset. A joint posterior over plausible film structures remains open.

## Simulation assumptions

The source is the MIT-licensed [Sherwin EUV repository](https://github.com/s-sherwin/EUV), model `Reflectivityapp_workspace_fit131.mat`. The saved model defines this synthetic family's center; it is not an independent experimental test. All simulations use the verified continuous-boundary solver instead of the upstream legacy periodic shortcut.

Twenty-two physical parameters vary independently and uniformly:

- Ten thickness/etch parameters use the released fitting bounds.
- Twelve free elemental concentration parameters vary within ±3% of their saved value, intersected with the released bounds. These are concentration parameters in the source optical model, not normalized chemical fractions.

Each acquisition also has independent channel gains uniformly distributed from 0.98 to 1.02 and a shared angle offset from −0.1° to 0.1°. Independent Gaussian intensity errors have standard deviations 0.0001 for absorber and 0.001 for multilayer. Negative noisy intensities are preserved. The bounds and noise levels are illustrative; they were not estimated from experimental data.

The challenge expands the material-concentration ranges to ±8% and adds a correlated wavelength sinusoid with random phase and amplitudes 0.0003/0.003 in the two channels. It changes two assumptions simultaneously; it does not isolate their separate effects or model every possible experimental error.

## Data splits and interval calibration

| Set | Cases | Seed | Purpose |
|---|---:|---:|---|
| Training | 8,000 | 4101 | Fit network and ridge baseline |
| Calibration | 500 | 4102 | Set interval radii only |
| Nominal test | 500 | 4103 | Evaluate the training regime |
| Shifted challenge | 500 | 4104 | Evaluate the specified mismatch |

The reported intervals use split-conformal absolute circular errors with the finite-sample quantile `ceil((n + 1) × coverage)`. Levels 50%, 80%, 90%, and 95% are saved. Radius is constant across cases; ensemble disagreement is not used as a confidence estimate. The coverage guarantee is marginal over exchangeable calibration/test cases. It does not apply to each case, to distribution shift, or to experimental phase accuracy. Wilson 95% intervals quantify finite test-sample uncertainty in observed coverage, not uncertainty in the physical model.

The model checks array shape, but cannot detect inputs outside its training distribution. Physics-checked posterior proposals and case-specific uncertainty remain open, as does reliable handling of unsupported inputs. The estimator cannot replace a physical fit that requires these capabilities.

## Baselines and results

The machine-readable results are in [`learning-benchmark.json`](../research/results/learning-benchmark.json); the comparison figure is [`learning-benchmark.svg`](../research/results/learning-benchmark.svg). They contain nominal and shifted errors, empirical coverage at all four levels, interval widths, and measured runtime.

A ridge regressor uses the same training spectra and targets, with five-fold cross-validation over 17 regularization strengths. It receives its own conformal calibration. A physical least-squares comparison uses 32 nominal-test cases sampled without replacement with seed 715. Each fit has three starts, matched physical/calibration bounds, matched noise weighting, and a 200-evaluation limit per start. The lowest-cost result is retained. Its timing includes all starts. Per-start termination flags are recorded; optimizer success does not prove a global optimum. The comparison covers point estimates only. It does not test posterior uncertainty or establish a neural speedup at equal accuracy.

The neural model's nominal 90% interval covered 89.4% of 500 test cases, with a width of 6.17°. Coverage fell to 74.6% on the shifted challenge. Nominal phase RMSE was 2.40°, compared with 5.22° for ridge; shifted RMSE was 3.02° and 7.02°, respectively. The shifted result shows that nominal interval calibration does not cover this mismatch.

On the 32-case physical-fit subset, three-start least squares reached 1.78° RMSE, compared with 2.56° for the neural model and 5.65° for ridge.  Its median runtime was 4.93 seconds per case; 31 of 32 retained solutions terminated successfully, and 7 of the 96 individual starts hit the evaluation limit. The neural estimator averaged 0.113 milliseconds per single-case call. The physical fit was slower and more accurate on this subset.

Simulation took 18.2 seconds and neural training 29.8 seconds in the recorded run. Prediction latency is an average of 500 single-case calls after warmup with one BLAS thread. Machine and source hashes are recorded. These timings apply to the recorded machine. We have not established how many predictions would repay the training cost at equal quality.

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
print(model.predict(spectra))
print(model.interval(spectra, .9))
```

The prediction is a phase offset in degrees. The interval gives arc endpoints for the nominal simulation regime.

The included calibration and test arrays reproduce the reported errors without retraining. Reusing these public tests to tune a new model turns them into development data; generate and disclose fresh final seeds for a new evaluation.

Code and generated weights use the repository MIT license. Upstream attribution remains in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
