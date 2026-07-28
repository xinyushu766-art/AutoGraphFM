"""Create the initial AutoGraphFM automotive data schema.

Revision ID: 20260714_0001
Revises:
Create Date: 2026-07-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260714_0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _month_first_day_constraint() -> sa.CheckConstraint:
    return sa.CheckConstraint(
        sa.extract("day", sa.column("month")) == 1,
        name="month_first_day",
    )


def upgrade() -> None:
    op.create_table(
        "brands",
        sa.Column("brand_name", sa.String(length=120), nullable=False),
        sa.Column("normalized_name", sa.String(length=120), nullable=False),
        sa.Column("country", sa.String(length=80), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_name"),
    )
    op.create_index(
        "ix_brands_normalized_name",
        "brands",
        ["normalized_name"],
        unique=False,
    )

    op.create_table(
        "ingestion_runs",
        sa.Column("source", sa.String(length=80), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("records_read", sa.Integer(), nullable=False),
        sa.Column("records_written", sa.Integer(), nullable=False),
        sa.Column("records_rejected", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("code_version", sa.String(length=80), nullable=True),
        sa.Column("run_metadata", sa.JSON(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR finished_at >= started_at",
            name="finished_after_started",
        ),
        sa.CheckConstraint(
            (
                "records_read >= 0 AND records_written >= 0 "
                "AND records_rejected >= 0"
            ),
            name="record_counts_nonnegative",
        ),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'partial')",
            name="status_allowed",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_ingestion_runs_source",
        "ingestion_runs",
        ["source"],
        unique=False,
    )

    op.create_table(
        "vehicle_series",
        sa.Column("brand_id", sa.Uuid(), nullable=False),
        sa.Column("series_name", sa.String(length=160), nullable=False),
        sa.Column("normalized_name", sa.String(length=160), nullable=False),
        sa.Column("segment", sa.String(length=80), nullable=True),
        sa.Column("body_type", sa.String(length=80), nullable=True),
        sa.Column("energy_type", sa.String(length=40), nullable=False),
        sa.Column("launch_date", sa.Date(), nullable=True),
        sa.Column("discontinue_date", sa.Date(), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('active', 'discontinued', 'upcoming', 'unknown')",
            name="status_allowed",
        ),
        sa.CheckConstraint(
            (
                "discontinue_date IS NULL OR launch_date IS NULL "
                "OR discontinue_date >= launch_date"
            ),
            name="valid_lifecycle_dates",
        ),
        sa.ForeignKeyConstraint(
            ["brand_id"],
            ["brands.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "brand_id",
            "normalized_name",
            name="uq_vehicle_series_brand_normalized_name",
        ),
    )
    op.create_index(
        "ix_vehicle_series_brand_id",
        "vehicle_series",
        ["brand_id"],
        unique=False,
    )

    op.create_table(
        "vehicle_aliases",
        sa.Column("source", sa.String(length=80), nullable=False),
        sa.Column("source_name", sa.String(length=200), nullable=False),
        sa.Column("normalized_name", sa.String(length=200), nullable=False),
        sa.Column("series_id", sa.Uuid(), nullable=False),
        sa.Column("match_method", sa.String(length=24), nullable=False),
        sa.Column("match_score", sa.Float(), nullable=False),
        sa.Column("review_status", sa.String(length=24), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "match_method IN ('exact', 'rule', 'fuzzy', 'manual')",
            name="match_method_allowed",
        ),
        sa.CheckConstraint(
            "match_score >= 0 AND match_score <= 1",
            name="match_score_range",
        ),
        sa.CheckConstraint(
            "review_status IN ('pending', 'approved', 'rejected')",
            name="review_status_allowed",
        ),
        sa.ForeignKeyConstraint(
            ["series_id"],
            ["vehicle_series.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source",
            "normalized_name",
            name="uq_vehicle_aliases_source_normalized_name",
        ),
    )
    op.create_index(
        "ix_vehicle_aliases_normalized_name",
        "vehicle_aliases",
        ["normalized_name"],
        unique=False,
    )
    op.create_index(
        "ix_vehicle_aliases_series_id",
        "vehicle_aliases",
        ["series_id"],
        unique=False,
    )

    op.create_table(
        "monthly_sales",
        sa.Column("series_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("month", sa.Date(), nullable=False),
        sa.Column("sales", sa.Integer(), nullable=False),
        sa.Column("sales_rank", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(length=80), nullable=False),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("data_version", sa.String(length=80), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        _month_first_day_constraint(),
        sa.CheckConstraint("sales >= 0", name="sales_nonnegative"),
        sa.CheckConstraint(
            "sales_rank IS NULL OR sales_rank > 0",
            name="sales_rank_positive",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["series_id"],
            ["vehicle_series.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "series_id",
            "month",
            "source",
            "data_version",
            name="uq_monthly_sales_series_month_source_version",
        ),
    )
    op.create_index(
        "ix_monthly_sales_month",
        "monthly_sales",
        ["month"],
        unique=False,
    )
    op.create_index(
        "ix_monthly_sales_month_series",
        "monthly_sales",
        ["month", "series_id"],
        unique=False,
    )
    op.create_index(
        "ix_monthly_sales_run_id",
        "monthly_sales",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "ix_monthly_sales_series_id",
        "monthly_sales",
        ["series_id"],
        unique=False,
    )

    op.create_table(
        "monthly_series_features",
        sa.Column("series_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("month", sa.Date(), nullable=False),
        sa.Column("guide_price", sa.Numeric(12, 2), nullable=True),
        sa.Column("dealer_price", sa.Numeric(12, 2), nullable=True),
        sa.Column("rating", sa.Numeric(4, 2), nullable=True),
        sa.Column("review_count", sa.Integer(), nullable=True),
        sa.Column("sentiment_mean", sa.Numeric(6, 5), nullable=True),
        sa.Column("positive_ratio", sa.Numeric(6, 5), nullable=True),
        sa.Column("popularity_index", sa.Numeric(14, 4), nullable=True),
        sa.Column("months_since_launch", sa.Integer(), nullable=True),
        sa.Column("is_on_sale", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(length=80), nullable=False),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_version", sa.String(length=80), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        _month_first_day_constraint(),
        sa.CheckConstraint(
            "dealer_price IS NULL OR dealer_price >= 0",
            name="dealer_price_nonnegative",
        ),
        sa.CheckConstraint(
            "guide_price IS NULL OR guide_price >= 0",
            name="guide_price_nonnegative",
        ),
        sa.CheckConstraint(
            (
                "months_since_launch IS NULL "
                "OR months_since_launch >= 0"
            ),
            name="months_since_launch_nonnegative",
        ),
        sa.CheckConstraint(
            (
                "popularity_index IS NULL "
                "OR popularity_index >= 0"
            ),
            name="popularity_nonnegative",
        ),
        sa.CheckConstraint(
            (
                "positive_ratio IS NULL "
                "OR (positive_ratio >= 0 AND positive_ratio <= 1)"
            ),
            name="positive_ratio_range",
        ),
        sa.CheckConstraint(
            (
                "rating IS NULL "
                "OR (rating >= 0 AND rating <= 5)"
            ),
            name="rating_range",
        ),
        sa.CheckConstraint(
            "review_count IS NULL OR review_count >= 0",
            name="review_count_nonnegative",
        ),
        sa.CheckConstraint(
            (
                "sentiment_mean IS NULL "
                "OR (sentiment_mean >= -1 AND sentiment_mean <= 1)"
            ),
            name="sentiment_range",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["series_id"],
            ["vehicle_series.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "series_id",
            "month",
            "source",
            "feature_version",
            name="uq_monthly_features_series_month_source_version",
        ),
    )
    op.create_index(
        "ix_monthly_features_month_series",
        "monthly_series_features",
        ["month", "series_id"],
        unique=False,
    )
    op.create_index(
        "ix_monthly_series_features_month",
        "monthly_series_features",
        ["month"],
        unique=False,
    )
    op.create_index(
        "ix_monthly_series_features_run_id",
        "monthly_series_features",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "ix_monthly_series_features_series_id",
        "monthly_series_features",
        ["series_id"],
        unique=False,
    )

    op.create_table(
        "data_quality_results",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("check_name", sa.String(length=160), nullable=False),
        sa.Column("table_name", sa.String(length=120), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("failed_rows", sa.Integer(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "failed_rows >= 0",
            name="failed_rows_nonnegative",
        ),
        sa.CheckConstraint(
            "severity IN ('info', 'warning', 'error')",
            name="severity_allowed",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_data_quality_results_run_id",
        "data_quality_results",
        ["run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_data_quality_results_run_id",
        table_name="data_quality_results",
    )
    op.drop_table("data_quality_results")

    op.drop_index(
        "ix_monthly_series_features_series_id",
        table_name="monthly_series_features",
    )
    op.drop_index(
        "ix_monthly_series_features_run_id",
        table_name="monthly_series_features",
    )
    op.drop_index(
        "ix_monthly_series_features_month",
        table_name="monthly_series_features",
    )
    op.drop_index(
        "ix_monthly_features_month_series",
        table_name="monthly_series_features",
    )
    op.drop_table("monthly_series_features")

    op.drop_index(
        "ix_monthly_sales_series_id",
        table_name="monthly_sales",
    )
    op.drop_index(
        "ix_monthly_sales_run_id",
        table_name="monthly_sales",
    )
    op.drop_index(
        "ix_monthly_sales_month_series",
        table_name="monthly_sales",
    )
    op.drop_index(
        "ix_monthly_sales_month",
        table_name="monthly_sales",
    )
    op.drop_table("monthly_sales")

    op.drop_index(
        "ix_vehicle_aliases_series_id",
        table_name="vehicle_aliases",
    )
    op.drop_index(
        "ix_vehicle_aliases_normalized_name",
        table_name="vehicle_aliases",
    )
    op.drop_table("vehicle_aliases")

    op.drop_index(
        "ix_vehicle_series_brand_id",
        table_name="vehicle_series",
    )
    op.drop_table("vehicle_series")

    op.drop_index(
        "ix_ingestion_runs_source",
        table_name="ingestion_runs",
    )
    op.drop_table("ingestion_runs")

    op.drop_index(
        "ix_brands_normalized_name",
        table_name="brands",
    )
    op.drop_table("brands")
