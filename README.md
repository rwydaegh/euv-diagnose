# EUV-Diagnose

[![Checks](https://github.com/rwydaegh/euv-diagnose/actions/workflows/ci.yml/badge.svg)](https://github.com/rwydaegh/euv-diagnose/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2-4D6A80?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-fits-4D6A80?logo=scipy&logoColor=white)](https://scipy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-MLP-4D6A80?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![MIT](https://img.shields.io/badge/license-MIT-59636e)](LICENSE)

EUV-Diagnose fits measured reflectivity spectra and estimates the phase of reflected light. It includes a thin-film solver and a small neural estimator trained on simulated spectra.

[Demo](https://rwydaegh.github.io/euv-diagnose/) · [Model card](docs/MODEL_CARD.md) · [Reproduce the results](docs/REPRODUCTION.md)

The current code covers one TaN mask family. The target is the relative reflection phase of the absorber and exposed multilayer at 13.5 nm and 6° from normal. The measurements and original model come from [Stuart Sherwin's EUV repository](https://github.com/s-sherwin/EUV), accompanying [the EUV reflectometry study](https://doi.org/10.1117/1.JMM.20.3.031011).

![Reflectivity spectra and phase comparison](docs/assets/screenshots/explorer-desktop.png)

## Run

Install [uv](https://docs.astral.sh/uv/), Python 3.12 or later, and Node.js 22. From this checkout:

```sh
make install
make demo
```

Open http://127.0.0.1:8000. The results and trained weights are included, so there is no training step before opening the demo. After installing dependencies, it runs offline on CPU. The page exports data as CSV or JSON and figures as SVG. The Python commands require the checkout and its data files.

If port 8000 is in use, run `uv run euv-diagnose demo --port 4319` after building the app.

## Methods and results

The solver uses transfer-matrix optics. Tests compare its complex amplitudes with the original code and check analytic cases. One check found an inconsistent boundary in the original repeated-layer calculation. The corrected calculation agrees with an explicit expansion of all 40 periods. The original convention remains available for reproducing the saved upstream fits.

A numerical search found two synthetic stacks whose relative phases differ by 1.00° but whose intensity spectra are nearly identical across seven angles. The page compares their spectra at additional angles. The pair depends on the stated parameter bounds and noise assumptions. The pair does not define a phase uncertainty interval or establish which measurement is optimal.

The measured-data study fits 22 stack parameters to published absorber and multilayer spectra. A second fit also varies intensity gains and an angle offset. Its predicted phase is 153.58°, compared with 147.21° for the stack-only fit. Its holdout error is slightly higher. The holdout is exploratory: the saved initialization was fitted to all angles. No independent phase measurement is available to check either prediction. The [fit notes](docs/REFIT.md) describe the objective and calibration bounds.

The learning study trains five small multilayer perceptrons on 8,000 simulated spectra. Their mean prediction estimates phase. A separate calibration set determines the interval width. The tests compare the estimator with ridge regression and physical least squares, then change the material ranges and add correlated measurement error.

| Synthetic test | Result |
|---|---:|
| Neural phase RMSE, 500 nominal cases | 2.40° |
| Ridge phase RMSE, same cases | 5.22° |
| Coverage of nominal 90% intervals | 89.4% |
| Coverage with changed assumptions | 74.6% |
| Physical / neural RMSE, same 32-case subset | 1.78° / 2.56° |
| Physical median / neural mean time per case, local CPU | 4.93 s / 0.113 ms |

Physical fitting was more accurate on the matched subset. Neural predictions were faster. Simulation and training costs are reported separately. The interval coverage argument applies to test cases exchangeable with the synthetic calibration data. Coverage is not guaranteed for each case or for different measurement conditions. The [model card](docs/MODEL_CARD.md) lists the input grid, data splits, and full results.

## Status

The solver, measured fits, scalar neural estimator, and web page are implemented. A joint posterior over stack and calibration parameters remains planned. A method for choosing the next measurement also needs to be built and tested. Experimental phase accuracy and usefulness to an external user have not been established.

The current model requires a fixed measurement grid and covers one mask family. Industrial fault diagnosis and wafer imaging have not been tested.

## Reproduce and develop

The [reproduction guide](docs/REPRODUCTION.md) gives the commands for each study. The repository includes source checksums, numerical references, and model weights in NumPy format.

```sh
make check
make browser-install
make e2e
```

GitHub Actions runs these checks. The web page uses TypeScript and SVG. Simulation and fitting use Python.

Code and generated weights use the [MIT license](LICENSE). Upstream attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Citation metadata is in [CITATION.cff](CITATION.cff).
