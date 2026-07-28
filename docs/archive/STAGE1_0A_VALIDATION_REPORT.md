# Stage 1.0A Validation Report

## Completed checks

- `python -m compileall`: passed
- `ruff check .`: passed
- Stage 1.0A database tests: **13 passed**
- Package-version import test: **1 passed**
- Total Stage 1.0A validation set: **14 passed**

The tests cover the seven ORM tables, Pydantic contracts, lifecycle validation,
database constraints, entity normalization, alias resolution, chronological
queries, raw quality checks, database healthcheck, Alembic upgrade/downgrade,
and PostgreSQL DDL compilation.

## Migration and script smoke test

The following sequence completed successfully against a temporary SQLite
database:

```text
python scripts/init_database.py
python scripts/check_database.py
python scripts/seed_database.py
python scripts/seed_database.py
python scripts/inspect_database.py
```

The seed script remained idempotent after two runs. Final development rows:

```text
brands                       1
vehicle_series               1
vehicle_aliases              1
monthly_sales                3
```

## Local PostgreSQL acceptance still required

Docker is not installed in this execution environment, so a real PostgreSQL
container could not be started here. Run locally:

```cmd
copy .env.example .env
docker compose up -d postgres
python scripts\init_database.py
python scripts\check_database.py
python scripts\seed_database.py
python scripts\inspect_database.py
```
