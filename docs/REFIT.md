# Refitting the measured EUV spectra

This case fits the initial absorber and exposed-multilayer measurements for mask 131 with the continuous-boundary solver. It tests whether the corrected physical model can reproduce measured spectra and predict angles omitted from its objective. It does **not** measure phase independently or identify a surface fault.

## Reproduce

```sh
uv sync --locked
uv run python scripts/refit_measurements.py --starts 3 --max-nfev 1200
```

The command writes [the result bundle](../research/results/refit-measured.json) and [a held-out spectrum plot](../research/results/refit-measured.png). It needs neither MATLAB nor a GPU. The original measurement files and saved model are already bundled with their source license. Each input checksum is recorded in the result.

## What enters the fit

The [Sherwin EUV release](https://github.com/s-sherwin/EUV) provides the measurements and original model. The provenance audit matches the released model to the original August 2019 absorber and June 2019 multilayer files. Both channels contain 101 wavelengths from 12.5 to 14.5 nm. We use their common released-model grid: angles 2–8° from the surface normal, s polarization. Training uses angles 2–5° (808 intensity values); evaluation uses 6–8° (606 values). The one negative reduced intensity on this grid is preserved.

All 22 coordinates marked free in the upstream parameter table vary within its bounds. These include thicknesses, a signed etch offset, and elemental concentration coefficients. Those coefficients are the released optical model's density/concentration multipliers; they are not normalized atomic fractions. Other coordinates, cached scattering factors, and the empirical substrate-roughness correction stay fixed. This remains a restricted material model.

The objective is the sum of squared intensity residuals after dividing each channel by its **training-only** RMS intensity. This prevents the much brighter multilayer channel from overwhelming the absorber. It is a declared fitting objective, not a likelihood based on independently measured noise. Negative measurements are neither clipped nor square-root transformed.

The first optimization starts at the saved upstream fit. Two additional starts perturb the normalized coordinates by a Gaussian with standard deviation 0.15, clipped inside the bounds. The saved fit previously saw every angle used here. Consequently, this is an **exploratory internal holdout**, not a blind generalization result. The implementation tests that changing held-out observations cannot alter the optimization or channel normalization.

## Calibration sensitivity

We compare a stack-only fit with one that also allows separate absorber/multilayer intensity gains from 0.95 to 1.05 and a shared angle offset from −0.1 to +0.1°. These ranges are illustrative sensitivity assumptions, not instrument calibration specifications. The measurements come from different acquisitions; a shared offset is only one possible calibration model.

Both fits optimize against the training angles alone. We report every start, including poor local minima and budget exhaustion, rather than silently discarding them. The exported phase is `arg(r_absorber / r_multilayer)` at the nominal 13.5 nm, 6° geometry and a shared reference plane. Calibration changes the inferred stack; the phase is evaluated at that same nominal geometry for both methods.

Several fitted coordinates reach their bounds. A least-squares convergence flag does not establish a unique physical stack or a global optimum. Differences between these fits demonstrate sensitivity to the stated assumptions; they are not confidence intervals. Multistart dispersion likewise has no calibrated uncertainty interpretation.

## Recorded result

| Model | Absorber train RMSE | Absorber holdout RMSE | Multilayer train RMSE | Multilayer holdout RMSE | Nominal relative phase |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stack only | 0.0000779 | 0.0001041 | 0.003021 | 0.003194 | 147.21° |
| Stack + calibration | 0.0000736 | 0.0001073 | 0.003058 | 0.003263 | 153.58° |

RMSE is in absolute reflectivity units. Both best fits converge. Two starts reach nearly the same useful solution for each model; one reaches a much worse local minimum despite its convergence flag. The full result includes those failures.

Allowing calibration freedom reduces the normalized training objective by about 4.4%, but slightly worsens held-out error in both channels. The predicted relative phase changes by 6.36°. The measured spectra therefore do not justify treating that phase prediction as independent of the calibration assumptions. These two fitted values do not bound the possible phase range.

## Scope of the result

Inspect the residual plot as well as the scalar errors. Smooth residual structure can indicate shortcomings in the optical model or measurement model that a point fit cannot resolve. The data provide no independent phase truth, so even a low intensity error cannot validate absolute phase accuracy.

A scientifically stronger extension would compare correlated residual models, material assumptions, and acquisition-specific calibration, then establish uncertainty on synthetic cases with known truth. The present artifact supplies the measured-data baseline and makes those limitations inspectable.
