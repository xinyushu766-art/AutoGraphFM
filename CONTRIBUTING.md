# Contributing

Contributions are welcome through focused pull requests.

1. Create a branch from `main`.
2. Install the development extras: `python -m pip install -e ".[dev]"`.
3. Add or update tests for behavioral changes.
4. Run the two bounded test groups documented in `docs/REPRODUCIBILITY.md`.
5. Keep evidence scopes explicit; synthetic or demo results must not be described as real vehicle performance.

For new models, include a simple baseline, temporal split, and falsification control where applicable.
