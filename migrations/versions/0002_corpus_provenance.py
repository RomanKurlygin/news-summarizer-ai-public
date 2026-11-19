"""Сохранить происхождение новостей и справочных записей."""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("knowledge_base", sa.Column("source_dataset", sa.String(160)))
    op.add_column("knowledge_base", sa.Column("source_document_id", sa.String(255)))
    op.add_column("knowledge_base", sa.Column("source_split", sa.String(16)))
    op.add_column("knowledge_base", sa.Column("content_hash", sa.String(64)))
    op.create_index("ix_knowledge_base_content_hash", "knowledge_base", ["content_hash"], unique=True)
    op.add_column("glossary", sa.Column("aliases", sa.JSON()))
    op.add_column("glossary", sa.Column("source", sa.JSON()))
    op.add_column("glossary", sa.Column("review_required", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    op.drop_column("glossary", "review_required")
    op.drop_column("glossary", "source")
    op.drop_column("glossary", "aliases")
    op.drop_index("ix_knowledge_base_content_hash", table_name="knowledge_base")
    op.drop_column("knowledge_base", "content_hash")
    op.drop_column("knowledge_base", "source_split")
    op.drop_column("knowledge_base", "source_document_id")
    op.drop_column("knowledge_base", "source_dataset")
