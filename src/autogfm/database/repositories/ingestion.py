"""Ingestion and quality-result repository."""

from sqlalchemy.orm import Session

from autogfm.database.models import DataQualityResult, IngestionRun
from autogfm.schemas import DataQualityResultCreate, IngestionRunCreate


class IngestionRepository:
    """Persist ingestion audit records and quality checks."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(
        self,
        payload: IngestionRunCreate,
    ) -> IngestionRun:
        run = IngestionRun(**payload.model_dump())
        self.session.add(run)
        self.session.flush()
        return run

    def add_quality_result(
        self,
        payload: DataQualityResultCreate,
    ) -> DataQualityResult:
        result = DataQualityResult(**payload.model_dump())
        self.session.add(result)
        self.session.flush()
        return result
