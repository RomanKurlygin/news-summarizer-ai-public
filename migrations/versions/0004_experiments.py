"""Состояние экспериментов и ссылки на результаты."""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "experiment_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("specification", sa.JSON(), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("lease_token", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("error", sa.String(100)),
    )
    for column in ("status", "user_id"):
        op.create_index(f"ix_experiment_jobs_{column}", "experiment_jobs", [column])
    for table in ("evaluation_results", "model_comparisons"):
        with op.batch_alter_table(table) as batch:
            batch.add_column(sa.Column("experiment_id", sa.Uuid()))
            batch.create_foreign_key(f"fk_{table}_experiment_id_experiment_jobs",
                                     "experiment_jobs", ["experiment_id"], ["id"], ondelete="CASCADE")
            batch.create_index(f"ix_{table}_experiment_id", ["experiment_id"])


def downgrade():
    for table in ("evaluation_results", "model_comparisons"):
        with op.batch_alter_table(table) as batch:
            batch.drop_index(f"ix_{table}_experiment_id")
            batch.drop_constraint(f"fk_{table}_experiment_id_experiment_jobs", type_="foreignkey")
            batch.drop_column("experiment_id")
    op.drop_table("experiment_jobs")
