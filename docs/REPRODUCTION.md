# Reproduction

Run `octave scripts/generate_optics_reference.m` to export numeric fixtures. Then run `uv run pytest tests/test_optics.py`. Complex-amplitude agreement checks the Python port against the released code. It does not independently check the source physics.
