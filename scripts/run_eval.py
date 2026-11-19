"""Запустить или продолжить воспроизводимый эксперимент на XSum."""

import argparse
import asyncio
import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import get_args
from uuid import UUID

from pydantic import ValidationError

from backend.core.config import Settings
from backend.db.models import ExperimentJob
from backend.db.session import Database
from backend.evaluation.experiments import create_job, job_report
from backend.evaluation.metrics_calculator import SummaryMetrics
from backend.evaluation.protocol import ExperimentError, ExperimentPlan, Mode
from backend.worker.tasks import execute


def _atomic(path, encode, *, encoding):
    with NamedTemporaryFile(mode="w", encoding=encoding, newline="", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            encode(stream)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export(report: dict, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / report["experiment_id"]
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    rows = [{
        "model": row["model_name"], "provider": row["provider"], "mode": row["mode"], "sample_size": row["sample_size"],
        "evaluation_time": row["evaluation_time"],
        **{key: value for key, value in row["aggregate"].items() if key not in {"metrics", "errors"}},
        **(row["aggregate"]["metrics"] or dict.fromkeys(SummaryMetrics.model_fields)),
    } for row in report["evaluations"]]

    def write_csv(stream):
        fields = list(rows[0]) if rows else ["model", "provider", "mode", "sample_size"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    _atomic(json_path, lambda file: json.dump(report, file, indent=2, ensure_ascii=False), encoding="utf-8")
    _atomic(csv_path, write_csv, encoding="utf-8-sig")
    return json_path, csv_path


def _plan(args) -> ExperimentPlan:
    return ExperimentPlan(
        models=[{"provider": provider, "model": model} for provider, model in (
            tuple(value.split("=", 1)) if "=" in value else (args.provider or "ollama", value)
            for value in args.models
        )],
        modes=args.modes or ["baseline"], sample_sizes=args.sample_sizes or [20], seed=args.seed or 42,
        **({} if args.parallelism is None else {"parallelism": args.parallelism}),
        **({} if args.timeout is None else {"timeout": args.timeout}),
    )


async def _run(settings, args):
    if settings.database_url is None:
        raise ExperimentError("database_url_required")
    db = Database(settings.database_url.get_secret_value(), settings)
    try:
        if args.resume:
            async with db.sessions() as session:
                job = await session.get(ExperimentJob, args.resume)
                if job is None:
                    raise ExperimentError("experiment_not_found")
        else:
            job = await create_job(db.sessions, _plan(args), diagnostic=args.diagnostic)
        print(f"experiment_id={job.id}", flush=True)
        started = perf_counter()
        try:
            await execute(settings, job.id)
        finally:
            async with db.sessions() as session:
                record = await session.get(ExperimentJob, job.id)
                report = await job_report(session, record)
            report["invocation_seconds"] = round(perf_counter() - started, 3)
            paths = export(report, args.output_dir)
            print(f"{report['status']} {report['progress']}/{report['total']}, "
                  f"{report['invocation_seconds']} s", flush=True)
            for row in report["evaluations"]:
                print(row["provider"], row["model_name"], row["mode"], row["sample_size"],
                      "success", row["aggregate"]["success_count"], "errors", row["aggregate"]["error_count"])
            print(*paths, sep="\n", flush=True)
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="Модель или provider=модель для смешанного набора")
    parser.add_argument("--provider", choices=("ollama", "transformers"))
    parser.add_argument("--modes", nargs="+", choices=get_args(Mode))
    parser.add_argument("--sample-sizes", nargs="+", type=int, choices=(20, 50, 100))
    parser.add_argument("--seed", type=int, choices=(42,))
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("data/indexes/experiments"))
    parser.add_argument("--diagnostic", type=int, choices=(2, 3, 4), help="Проверить первые записи из выборки 20")
    parser.add_argument("--resume", type=UUID, help="Продолжить эксперимент с исходными параметрами")
    args = parser.parse_args()
    if not args.models and not args.resume:
        parser.error("Укажите --models или --resume.")
    if args.resume and any(value is not None for value in (
        args.models, args.provider, args.modes, args.sample_sizes, args.seed,
        args.parallelism, args.timeout, args.diagnostic,
    )):
        parser.error("--resume использует сохранённую конфигурацию. Допускается указать --output-dir.")
    if args.diagnostic and args.sample_sizes not in (None, [20]):
        parser.error("Диагностика допускается только для нового эксперимента с --sample-sizes 20.")
    try:
        asyncio.run(_run(Settings(), args))
    except (ValidationError, ExperimentError, KeyboardInterrupt) as exc:
        parser.exit(1, f"Эксперимент остановлен ({getattr(exc, 'code', type(exc).__name__)}).\n")


if __name__ == "__main__":
    main()
