"""initial schema."""

from alembic import op
import sqlalchemy as sa


revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'cache',
        sa.Column('input_key', sa.String(length=255), nullable=False),
        sa.Column('output', sa.JSON(), nullable=False),
        sa.Column('hits', sa.Integer(), server_default='0', nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint('hits >= 0', name=op.f('ck_cache_nonnegative_hits')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_cache')),
        sa.UniqueConstraint('input_key', name=op.f('uq_cache_input_key')),
    )
    op.create_index(op.f('ix_cache_expires_at'), 'cache', ['expires_at'], unique=False)
    op.create_table(
        'evaluation_results',
        sa.Column('model_name', sa.String(length=255), nullable=False),
        sa.Column('mode', sa.String(length=32), nullable=False),
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('metrics_json', sa.JSON(), nullable=False),
        sa.Column('evaluation_time', sa.Float(), nullable=False),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint('evaluation_time >= 0', name=op.f('ck_evaluation_results_nonnegative_evaluation_time')),
        sa.CheckConstraint('sample_size > 0', name=op.f('ck_evaluation_results_positive_sample_size')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_evaluation_results')),
    )
    op.create_index('ix_evaluation_results_model_mode_size', 'evaluation_results', ['model_name', 'mode', 'sample_size'], unique=False)
    op.create_table(
        'glossary',
        sa.Column('term', sa.String(length=255), nullable=False),
        sa.Column('definition', sa.Text(), nullable=False),
        sa.Column('category', sa.String(length=120), server_default='general', nullable=False),
        sa.Column('language', sa.String(length=16), server_default='und', nullable=False),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_glossary')),
        sa.UniqueConstraint('term', 'category', 'language', name='uq_glossary_term_category_language'),
    )
    op.create_index('ix_glossary_category_language', 'glossary', ['category', 'language'], unique=False)
    op.create_table(
        'knowledge_base',
        sa.Column('source_text', sa.Text(), nullable=False),
        sa.Column('target_text', sa.Text(), nullable=False),
        sa.Column('category', sa.String(length=120), server_default='general', nullable=False),
        sa.Column('language', sa.String(length=16), server_default='und', nullable=False),
        sa.Column('source_url', sa.Text(), nullable=True),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_knowledge_base')),
    )
    op.create_index('ix_knowledge_base_category_language', 'knowledge_base', ['category', 'language'], unique=False)
    op.create_table(
        'model_comparisons',
        sa.Column('models_compared', sa.JSON(), nullable=False),
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('results_json', sa.JSON(), nullable=False),
        sa.Column('winner', sa.String(length=255), nullable=True),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint('sample_size > 0', name=op.f('ck_model_comparisons_positive_sample_size')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_model_comparisons')),
    )
    op.create_index(op.f('ix_model_comparisons_sample_size'), 'model_comparisons', ['sample_size'], unique=False)
    op.create_table(
        'users',
        sa.Column('email', sa.String(length=320), nullable=False),
        sa.Column('username', sa.String(length=64), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=16), server_default='user', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("role IN ('user', 'editor', 'admin')", name=op.f('ck_users_valid_role')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
        sa.UniqueConstraint('email', name=op.f('uq_users_email')),
        sa.UniqueConstraint('username', name=op.f('uq_users_username')),
    )
    op.create_table(
        'summarization_requests',
        sa.Column('user_id', sa.Uuid(), nullable=True),
        sa.Column('input_text', sa.Text(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('mode', sa.String(length=32), server_default='full', nullable=False),
        sa.Column('status', sa.String(length=32), server_default='pending', nullable=False),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_summarization_requests_user_id_users'), ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_summarization_requests')),
    )
    op.create_index('ix_summarization_requests_user_created', 'summarization_requests', ['user_id', 'created_at'], unique=False)
    op.create_table(
        'llm_calls',
        sa.Column('request_id', sa.Uuid(), nullable=False),
        sa.Column('agent_name', sa.String(length=100), nullable=False),
        sa.Column('model_name', sa.String(length=255), nullable=False),
        sa.Column('provider', sa.String(length=64), nullable=False),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('prompt_hash', sa.String(length=64), nullable=False),
        sa.Column('response_hash', sa.String(length=64), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint('latency_ms >= 0', name=op.f('ck_llm_calls_nonnegative_latency')),
        sa.CheckConstraint('tokens_used >= 0', name=op.f('ck_llm_calls_nonnegative_tokens')),
        sa.ForeignKeyConstraint(['request_id'], ['summarization_requests.id'], name=op.f('fk_llm_calls_request_id_summarization_requests'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_llm_calls')),
    )
    op.create_index('ix_llm_calls_request_created', 'llm_calls', ['request_id', 'created_at'], unique=False)
    op.create_table(
        'reasoning_logs',
        sa.Column('request_id', sa.Uuid(), nullable=False),
        sa.Column('agent_name', sa.String(length=100), nullable=False),
        sa.Column('input_summary', sa.Text(), nullable=True),
        sa.Column('output_summary', sa.Text(), nullable=True),
        sa.Column('model_name', sa.String(length=255), nullable=True),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('errors', sa.JSON(), nullable=True),
        sa.Column('warnings', sa.JSON(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['request_id'], ['summarization_requests.id'], name=op.f('fk_reasoning_logs_request_id_summarization_requests'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_reasoning_logs')),
    )
    op.create_index('ix_reasoning_logs_request_started', 'reasoning_logs', ['request_id', 'started_at'], unique=False)
    op.create_table(
        'uncertain_examples',
        sa.Column('request_id', sa.Uuid(), nullable=False),
        sa.Column('input_text', sa.Text(), nullable=False),
        sa.Column('model_output', sa.Text(), nullable=False),
        sa.Column('critic_score', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=32), server_default='pending', nullable=False),
        sa.Column('reasons', sa.JSON(), nullable=True),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint('critic_score BETWEEN 0 AND 10', name=op.f('ck_uncertain_examples_critic_score_range')),
        sa.ForeignKeyConstraint(['request_id'], ['summarization_requests.id'], name=op.f('fk_uncertain_examples_request_id_summarization_requests'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_uncertain_examples')),
        sa.UniqueConstraint('request_id', name=op.f('uq_uncertain_examples_request_id')),
    )
    op.create_index('ix_uncertain_examples_status_created', 'uncertain_examples', ['status', 'created_at'], unique=False)
    op.create_table(
        'feedback',
        sa.Column('uncertain_example_id', sa.Uuid(), nullable=False),
        sa.Column('editor_id', sa.Uuid(), nullable=True),
        sa.Column('decision', sa.String(length=16), nullable=False),
        sa.Column('corrected_summary', sa.Text(), nullable=True),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("decision IN ('approved', 'rejected')", name=op.f('ck_feedback_valid_decision')),
        sa.ForeignKeyConstraint(['editor_id'], ['users.id'], name=op.f('fk_feedback_editor_id_users'), ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['uncertain_example_id'], ['uncertain_examples.id'], name=op.f('fk_feedback_uncertain_example_id_uncertain_examples'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_feedback')),
    )
    op.create_index(op.f('ix_feedback_editor_id'), 'feedback', ['editor_id'], unique=False)
    op.create_index('ix_feedback_example_created', 'feedback', ['uncertain_example_id', 'created_at'], unique=False)


def downgrade() -> None:
    raise NotImplementedError("Downgrading the initial schema would delete application data")
