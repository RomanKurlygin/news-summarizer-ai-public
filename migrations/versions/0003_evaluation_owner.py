"""Связать результаты оценки с владельцами запросов."""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("evaluation_results") as batch:
        batch.add_column(sa.Column("user_id", sa.Uuid(), nullable=True))
        batch.create_foreign_key("fk_evaluation_results_user_id_users", "users", ["user_id"], ["id"],
                                 ondelete="SET NULL")
        batch.create_index("ix_evaluation_results_user_created", ["user_id", "created_at"])


def downgrade() -> None:
    with op.batch_alter_table("evaluation_results") as batch:
        batch.drop_index("ix_evaluation_results_user_created")
        batch.drop_constraint("fk_evaluation_results_user_id_users", type_="foreignkey")
        batch.drop_column("user_id")
