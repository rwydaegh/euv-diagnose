# Refitting the measured EUV spectra

We fit the initial absorber and exposed-multilayer spectra for mask 131 with the continuous-boundary solver, then predict spectra at angles excluded from the objective. The data contain no independent phase measurement or labeled surface fault.

## Reproduce

```sh
uv sync --locked
uv run python scripts/refit_measurements.py --starts 3 --max-nfev 1200
```

The command writes [the result bundle](../research/results/refit-measured.json) and [a held-out spectrum plot](../research/results/refit-measured.png). It needs neither MATLAB nor a GPU. The original measurement files and saved model are already bundled with their source license. Each input checksum is recorded in the result.

## What enters the fit

The [Sherwin EUV release](https://github.com/s-sherwin/EUV) provides the measurements and original model. The provenance audit matches the released model to the original August 2019 absorber and June 2019 multilayer files. Both channels contain 101 wavelengths from 12.5 to 14.5 nm. We use their common released-model grid: angles 2–8° from the surface normal, s polarization. Training uses angles 2–5° (808 intensity values); evaluation uses 6–8° (606 values). The one negative reduced intensity on this grid is preserved.

All 22 coordinates marked free in the upstream parameter table vary within its bounds. These include thicknesses, a signed etch offset, and elemental concentration coefficients. Those coefficients are the released optical model's density/concentration multipliers; they are not normalized atomic fractions. Other coordinates, cached scattering factors, and the empirical substrate-roughness correction stay fixed.

We divide each channel by its **training-only** RMS intensity and minimize the sum of squared intensity residuals. This balances the bright multilayer and weaker absorber channels. The weights are not based on independently measured noise. Negative measurements are neither clipped nor square-root transformed.

The first optimization starts at the saved upstream fit. Two additional starts perturb the normalized coordinates by a Gaussian with standard deviation 0.15, clipped inside the bounds. The saved fit used all angles in this study, so this is an **exploratory internal holdout**. It is not a blind generalization test. Tests verify that changes to held-out observations leave the optimization and channel normalization unchanged.

## Calibration sensitivity

We compare a stack-only fit with one that also allows separate absorber/multilayer intensity gains from 0.95 to 1.05 and a shared angle offset from −0.1 to +0.1°. These illustrative ranges are not instrument calibration specifications. The measurements come from different acquisitions; a shared offset is only one possible calibration model.

Both fits optimize against the training angles alone. We report every start, including poor local minima and budget exhaustion. The exported phase is `arg(r_absorber / r_multilayer)` at the nominal 13.5 nm, 6° geometry and a shared reference plane. Calibration changes the inferred stack; the phase is evaluated at that same nominal geometry for both methods.

Several fitted coordinates reach their bounds. A least-squares convergence flag does not establish a unique physical stack or a global optimum. The differences between fits show sensitivity to the assumptions. Neither these differences nor the spread across starts are calibrated confidence intervals.

