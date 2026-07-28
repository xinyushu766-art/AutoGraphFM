"""Check whether the configured database accepts connections."""

from autogfm.database import create_database_engine, database_healthcheck


def main() -> None:
    engine = create_database_engine()
    if not database_healthcheck(engine):
        raise SystemExit("Database healthcheck failed.")
    print("Database healthcheck passed.")


if __name__ == "__main__":
    main()
