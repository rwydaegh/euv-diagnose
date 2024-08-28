# Reproduction

Run `octave scripts/generate_optics_reference.m` to export numeric fixtures. Then run `uv run pytest tests/test_optics.py`. Complex-amplitude agreement checks the Python port against the released code. It does not independently check the source physics.

The generic transfer matrix uses continuous boundaries. `ReleasedModel.load` retains the legacy convention for saved-model replay. New calculations select `replace(model, periodic_convention="continuous")`. The repeated-cell result also agrees with explicit layer expansion.
