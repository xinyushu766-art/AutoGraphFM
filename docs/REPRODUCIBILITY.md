# Reproducibility

## Validated environment

- Python 3.11+
- PyTorch 2.2–2.x
- NumPy 1.26+
- SQLAlchemy 2.0
- CPU execution is supported; a GPU reduces deep-model runtime.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Tests

The public release contains 108 passing automated tests. The release evidence was executed in bounded groups:

```bash
pytest -q tests/graph tests/test_*.py
pytest -q tests/database tests/forecasting tests/monitoring
pytest -q tests/realdata tests/reporting
pytest -q tests/pipeline/test_stage4_pipeline.py tests/pipeline/test_stage10a_pipeline.py
pytest -q tests/pipeline/test_stage10b_pipeline.py
pytest -q tests/pipeline/test_stage11_pipeline.py
pytest -q tests/pipeline/test_stage11b_pipeline.py
pytest -q tests/pipeline/test_stage12a_pipeline.py
python -m compileall -q src scripts tests
```

## Reproduce the real aggregate benchmark

```bash
python scripts/run_stage10b.py \
  --output-dir outputs/stage10b_real \
  --overwrite
```

Expected comparison evidence is retained in:

- `reports/real_market/STAGE10B_COMPARISON.csv`
- `reports/real_market/STAGE10B_REAL_DATA_REPORT.md`

## Reproduce graph stability validation

```bash
python scripts/validate_stage12a.py --overwrite --full
```

Expected controlled evidence:

- Precision@K: 0.8333
- Recall@K: 0.8333
- Cross-seed edge Jaccard: 1.0

## Database workflow

```bash
cp .env.example .env
python -m pip install -e ".[dev,postgres,snapshot]"
docker compose up -d postgres
python scripts/init_database.py
python scripts/check_database.py
```

Generated artifacts belong in `outputs/`, which is ignored by Git.
