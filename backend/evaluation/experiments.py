"""Выполнение, контрольные точки и сравнение экспериментов."""

import asyncio
from collections import Counter
from datetime import UTC, datetime, timedelta
from statistics import fmean
from time import perf_counter
from uuid import UUID, uuid4, uuid5

from sqlalchemy import and_, func, or_, select, update

from backend.api.schemas.summarization import MAX_NEWS_CHARS, ModelChoice
from backend.db.models import EvaluationResult, ExperimentJob, LLMCall, ModelComparison
from backend.evaluation.metrics_calculator import SummaryMetrics, average_metrics
from backend.evaluation.protocol import (
    ENSEMBLE_TEMPERATURES, ExperimentError, ExperimentPlan, Mode, Protocol,
    digest, model_key, model_versions, prepare_protocol,
)

_FACTUAL_CODES = {
    "unsupported_vehicle_impact", "unsupported_causal_link", "unattributed_allegation",
    "unsupported_name", "unsupported_number", "lost_qualification",
}


class ExperimentBusy(ExperimentError):
    def __init__(self):
        super().__init__("experiment_running")


def lease() -> datetime:
    # Последовательность из двух кандидатов может использовать два таймаута по 600 секунд.
    return datetime.now(UTC) + timedelta(minutes=30)


async def create_job(sessions, plan: ExperimentPlan, user_id: UUID | None = None, *, diagnostic: int | None = None):
    total = len(plan.models) * len(plan.modes) * sum(diagnostic or size for size in plan.sample_sizes)
    async with sessions.begin() as session:
        row = ExperimentJob(user_id=user_id, status="queued", total=total,
                            specification={"plan": plan.model_dump(mode="json"), "diagnostic": diagnostic})
        session.add(row)
        await session.flush()
    return row


def _report_fields(report: dict, request_id: UUID) -> dict:
    findings = report.get("factual_findings", [])
    errors = report["validation_error_codes"]
    status = report["status"]
    error = None
    if status in {"failed", "incomplete_configuration", "cancelled"}:
        error = report.get("failure_code") or next(iter(report["error_codes"]), "generation_failed")
    return {
        "request_id": str(request_id), "status": status, "error": error,
        "models": report["models"], "rag_ids": report["rag_ids"],
        "validator_checked": report["validation_passed"] is not None,
        "validator_errors": errors, "validator_warnings": report["validation_warning_codes"],
        "factual_flags": sorted({finding["code"] for finding in findings} | (set(errors) & _FACTUAL_CODES)),
        "critic_factual_error": report.get("critical_factual_error"),
        "critic_score": None if report["critic_fallback_used"] else report["critic_score"],
        "edits": report["iterations"], "llm_attempts": report["llm_attempts"],
    }


async def generate_example(runtime, row, model: ModelChoice, mode: Mode, plan: ExperimentPlan,
                           protocol: Protocol, user_id: UUID | None):
    started = perf_counter()
    output = None
    candidates = []
    headline = {"test_id": str(row.id), "source_document_id": row.source_document_id}
    if len(row.source_text) > MAX_NEWS_CHARS:
        return headline | {"status": "failed", "error": "input_too_long", "duration_s": 0}, None
    temperatures = ENSEMBLE_TEMPERATURES if mode == "+ensemble" else (0.0,)
    for temperature in temperatures:
        request_id = uuid4()
        effective_mode = "+agents" if mode == "+ensemble" else mode
        judge = (plan.judge.provider, plan.judge.model) if effective_mode == "+agents" else None
        try:
            report = await runtime.run_pipeline(
                row.source_text, mode=effective_mode, provider=model.provider, model=model.model,
                user_id=user_id, request_id=request_id, language=row.language, temperature=temperature,
                seed=plan.seed, timeout=plan.timeout, judge=judge,
                terms=protocol.terms, manifest=protocol.manifest, holdout=protocol.holdout,
            )
            expected = {(model.provider, model.model)}
            if judge is not None:
                expected.add(judge)
            expected |= {(provider, value + ":latest") for provider, value in expected
                         if provider == "ollama" and ":" not in value}
            if any((used["provider"], used["model"]) not in expected for used in report["models"]):
                raise ExperimentError("unexpected_model")
            entry = _report_fields(report, request_id)
            candidates.append((entry | {"temperature": temperature}, report["final_summary"]))
        except TimeoutError:
            async with runtime.database.sessions() as session:
                calls = await session.scalar(select(func.count()).select_from(LLMCall)
                                             .where(LLMCall.request_id == request_id))
            candidates.append(({"request_id": str(request_id), "status": "failed", "error": "timeout",
                                "llm_attempts": calls, "temperature": temperature}, None))
    if mode == "+ensemble":
        eligible = [index for index, (entry, text) in enumerate(candidates)
                    if text and entry["status"] == "verified" and entry["critic_score"] is not None]
        winner = max(eligible, key=lambda index: candidates[index][0]["critic_score"], default=None)
        if winner is None:
            result = {"status": "needs_review" if any(e["status"] == "needs_review" for e, _ in candidates)
                      else "failed", "error": "no_verified_candidate"}
        else:
            result, output = candidates[winner]
        result = result | {
            "candidates": [entry for entry, _ in candidates], "winner_index": winner,
            "edits": sum(entry.get("edits", 0) for entry, _ in candidates),
            "llm_attempts": sum(entry.get("llm_attempts", 0) for entry, _ in candidates),
        }
    else:
        result, output = candidates[0]
    return headline | result | {"duration_s": round(perf_counter() - started, 4)}, (
        (output, row.target_text) if output and not result["error"] else None
    )


def aggregate(items: list[dict]) -> dict:
    scores = [SummaryMetrics.model_validate(item["metrics"]) for item in items if item.get("metrics") is not None]
    attempts = [candidate for item in items for candidate in item.get("candidates", [item])]
    checked = [item for item in attempts if item.get("validator_checked")]
    judge_scores = [item["critic_score"] for item in attempts if item.get("critic_score") is not None]
    statuses = Counter(item["status"] for item in items)
    errors = Counter(item["error"] for item in items if item.get("error"))
    factual = sum(bool(item.get("factual_flags") or item.get("critic_factual_error")
                       or item["status"] == "needs_review") for item in checked)
    calls = sum(item.get("llm_attempts", 0) for item in items)
    return {
        "processed_count": len(items),
        "success_count": len(scores), "error_count": sum(errors.values()), "errors": dict(errors),
        "success_rate": len(scores) / len(items) if items else None,
        "verified_count": statuses["verified"], "needs_review_count": statuses["needs_review"],
        "edit_count": sum(item.get("edits", 0) for item in items),
        "llm_attempts": calls, "mean_llm_attempts": calls / len(items) if items else None,
        "validator_checked_count": len(checked),
        "validator_notice_rate": (sum(bool(item.get("validator_errors") or item.get("validator_warnings"))
                                      for item in checked) / len(checked) if checked else None),
        "factual_notice_count": factual if checked else None,
        "factual_notice_rate": factual / len(checked) if checked else None,
        "critic_scored_count": len(judge_scores),
        "mean_critic": fmean(judge_scores) if judge_scores else None,
        "mean_seconds": fmean(item["duration_s"] for item in items) if items else None,
        "metrics": average_metrics(scores).model_dump() if scores else None,
    }


def comparison(rows: list[EvaluationResult]) -> dict:
    entries = [{
        "evaluation_id": str(row.id), "model": row.model_name, "mode": row.mode,
        "provider": row.metrics_json["provider"], "sample_size": row.sample_size,
        "actual_models": row.metrics_json["actual_models"], "evaluation_time": row.evaluation_time,
        **row.metrics_json["aggregate"],
    } for row in rows]
    eligible = [entry for entry in entries if entry["success_count"] == entry["sample_size"]]
    winner = max(eligible, key=lambda entry: (entry["metrics"]["rouge_l"], entry["metrics"]["bertscore_f1"]),
                 default=None) if len(eligible) >= 2 else None
    return {
        "criterion": ["rouge_l", "bertscore_f1"], "runs": entries,
        "winner_evaluation_id": winner["evaluation_id"] if winner else None,
    }


class ExperimentRunner:
    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self.sessions = runtime.database.sessions

    async def _claim(self, job_id: UUID, token: UUID):
        async with self.sessions.begin() as session:
            request = update(ExperimentJob).where(ExperimentJob.id == job_id, or_(
                ExperimentJob.status.in_(("queued", "failed", "cancelled", "broker_unavailable")),
                and_(ExperimentJob.status == "running", ExperimentJob.lease_until < datetime.now(UTC)),
            )).values(status="running", lease_token=token, lease_until=lease(), error=None)
            claimed = await session.scalar(request.returning(ExperimentJob))
            if claimed is not None:
                return claimed
            existing = await session.get(ExperimentJob, job_id)
            if existing is None:
                raise ExperimentError("experiment_not_found")
            if existing.status != "completed":
                raise ExperimentBusy()
            return existing

    @staticmethod
    async def _touch(session, job_id, token, **values):
        result = await session.scalar(update(ExperimentJob).where(
            ExperimentJob.id == job_id, ExperimentJob.lease_token == token,
        ).values(lease_until=lease(), **values).returning(ExperimentJob.id))
        if result is None:
            raise ExperimentError("lease_lost")

    async def _finish(self, job_id, token, status, error=None):
        async with self.sessions.begin() as session:
            await self._touch(session, job_id, token, status=status, error=error, lease_token=None)

    async def run(self, job_id: UUID) -> None:
        token = uuid4()
        job = await self._claim(job_id, token)
        if job.status == "completed":
            return
        try:
            plan = ExperimentPlan.model_validate(job.specification["plan"])
            protocol = await prepare_protocol(self.runtime, plan, job.specification["diagnostic"])
            old = job.specification.get("protocol")
            if old is not None and digest(old) != digest(protocol.metadata):
                raise ExperimentError("protocol_changed")
            async with self.sessions.begin() as session:
                await self._touch(session, job_id, token,
                                  specification=job.specification | {"protocol": protocol.metadata})
            protocol_hash = digest(protocol.metadata)
            # Только результаты этого задания с тем же протоколом и конфигурацией.
            reusable: dict[tuple[str, Mode], dict[str, dict]] = {}
            for size, sample in sorted(protocol.samples.items()):
                rows = []
                for model in plan.models:
                    for mode in plan.modes:
                        prior = reusable.setdefault((model_key(model), mode), {})
                        row = await self._evaluate(job, token, protocol, plan, model, mode,
                                                   size, sample, prior, protocol_hash)
                        rows.append(row)
                        for item in row.metrics_json["items"]:
                            prior.setdefault(item["test_id"], item | {"reused_from_evaluation_id": str(row.id)})
                if len(rows) > 1:
                    await self._compare(job_id, token, size, rows)
            await self._finish(job_id, token, "completed")
        except asyncio.CancelledError:
            await self._finish(job_id, token, "cancelled", "cancelled")
            raise
        except Exception as exc:
            code = exc.code if isinstance(exc, ExperimentError) else type(exc).__name__
            if code != "lease_lost":
                await self._finish(job_id, token, "failed", code)
            raise

    async def _evaluate(self, job, token, protocol, plan, model, mode, size, sample, reusable, protocol_hash):
        row_id = uuid5(job.id, f"{size}:{mode}:{model_key(model)}")
        async with self.sessions() as session:
            row = await session.get(EvaluationResult, row_id)
        if row is not None and row.metrics_json["protocol_sha256"] != protocol_hash:
            raise ExperimentError("protocol_changed")
        existing = row.metrics_json["items"] if row is not None else []
        completed = {item["test_id"] for item in existing}
        pending = [item for item in sample if str(item.id) not in completed]
        for offset in range(0, len(pending), plan.parallelism):
            chunk = pending[offset:offset + plan.parallelism]
            fresh = [item for item in chunk if str(item.id) not in reusable]
            async with asyncio.TaskGroup() as tasks:
                work = {str(item.id): tasks.create_task(
                    generate_example(self.runtime, item, model, mode, plan, protocol, job.user_id),
                ) for item in fresh}
            generated = [task.result() for task in work.values()]
            pairs = [pair for _, pair in generated if pair is not None]
            grades = iter(await asyncio.to_thread(self.runtime.metrics.calculate, pairs)) if pairs else iter(())
            for item, pair in generated:
                item["metrics"] = next(grades).model_dump() if pair is not None else None
            if fresh and await model_versions(self.runtime, protocol.references) != protocol.metadata["model_versions"]:
                raise ExperimentError("model_changed")
            available = reusable | {item["test_id"]: item for item, _ in generated}
            existing = [*existing, *(available[str(item.id)] for item in chunk)]
            row = await self._checkpoint(job, token, row_id, protocol_hash, model, mode, sample, existing, len(chunk))
        return row

    async def _checkpoint(self, job, token, row_id, protocol_hash, model, mode, sample, items, count):
        reused = sum("reused_from_evaluation_id" in item for item in items)
        data = {
            "provider": model.provider, "requested_model": model.model, "items": items,
            "sample_ids": [str(row.id) for row in sample], "aggregate": aggregate(items),
            "actual_models": sorted({f"{value['provider']}:{value['model']}"
                                     for item in items for candidate in item.get("candidates", [item])
                                     for value in candidate.get("models", [])}),
            "protocol_sha256": protocol_hash,
            "reused_count": reused, "generated_count": len(items) - reused,
        }
        seconds = sum(item["duration_s"] for item in items)
        async with self.sessions.begin() as session:
            await self._touch(session, job.id, token, progress=ExperimentJob.progress + count)
            row = await session.get(EvaluationResult, row_id)
            if row is None:
                row = EvaluationResult(id=row_id, experiment_id=job.id, user_id=job.user_id,
                                       model_name=model.model, mode=mode, sample_size=len(sample),
                                       metrics_json=data, evaluation_time=seconds)
                session.add(row)
            else:
                row.metrics_json = data
                row.evaluation_time = seconds
        return row

    async def _compare(self, job_id, token, size, rows):
        result = comparison(rows)
        async with self.sessions.begin() as session:
            await self._touch(session, job_id, token)
            await session.merge(ModelComparison(
                id=uuid5(job_id, f"comparison:{size}"), experiment_id=job_id, sample_size=rows[0].sample_size,
                models_compared=sorted({row.model_name for row in rows}), results_json=result,
                winner=result["winner_evaluation_id"],
            ))


async def job_report(session, job: ExperimentJob) -> dict:
    evaluations = await session.scalars(select(EvaluationResult).where(EvaluationResult.experiment_id == job.id)
                                        .order_by(EvaluationResult.sample_size, EvaluationResult.model_name,
                                                  EvaluationResult.mode, EvaluationResult.id))
    comparisons = await session.scalars(select(ModelComparison).where(ModelComparison.experiment_id == job.id)
                                        .order_by(ModelComparison.sample_size))
    return {
        "experiment_id": str(job.id), "user_id": str(job.user_id) if job.user_id else None,
        "status": job.status, "progress": job.progress, "total": job.total, "error": job.error,
        "created_at": job.created_at.isoformat(), "updated_at": job.updated_at.isoformat(),
        "protocol": job.specification.get("protocol"),
        "evaluations": [{
            "id": str(row.id), "model_name": row.model_name, "mode": row.mode, "sample_size": row.sample_size,
            "evaluation_time": row.evaluation_time, **row.metrics_json,
        } for row in evaluations],
        "comparisons": [{"id": str(row.id), "sample_size": row.sample_size, **row.results_json} for row in comparisons],
    }
