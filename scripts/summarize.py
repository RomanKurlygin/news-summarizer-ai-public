"""Запустить суммаризацию с локальными индексами и реальными адаптерами."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
from sqlalchemy.exc import SQLAlchemyError

from backend.core.config import Settings
from backend.core.constants import PROJECT_ROOT
from backend.data.llm_manager import LLMManager
from backend.data.llm_types import LLMError
from backend.data.ollama import OllamaProvider
from backend.db.session import Database
from backend.retrieval.glossary import glossary_context, glossary_snapshot
from backend.services.summarization import SummarizationService
from backend.utils.text_processing import detect_language
from scripts.corpus import News, load_prepared

TEST_FILE = PROJECT_ROOT / "data" / "test" / "news_test.json"


def _news(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.test_id:
        for item in load_prepared(args.test_file, News):
            if item.split == "test" and args.test_id in {str(item.id), item.source_document_id}:
                return item.source_text, args.language or item.language
        raise ValueError("Новость с указанным ID не найдена в test.")
    return args.news.read_text(encoding="utf-8-sig"), args.language


async def _run(args: argparse.Namespace) -> dict:
    settings = Settings()
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds, trust_env=False) as http:
        if args.check_ollama:
            model = args.model or settings.ollama_model
            await OllamaProvider(http, str(settings.ollama_base_url)).check_model(model)
            return {"status": "available", "provider": "ollama", "model": model}
        if settings.database_url is None:
            raise ValueError("Задайте DATABASE_URL для журналов агента и LLM.")
        news, language = _news(args)
        language = language or detect_language(news)

        async def enrich(state):
            from backend.data.embeddings import Embeddings
            from backend.retrieval.index_bundle import open_indexes
            from scripts.build_indexes import load_holdout

            guard = load_holdout(args.test_file)
            async with open_indexes(settings, guard, Embeddings(settings)) as (retriever, _):
                return await retriever.enrich_state(state, language=language, cross_language=True)

        database = Database(settings.database_url.get_secret_value(), settings)
        async def glossary(state):
            return glossary_context(state, await glossary_snapshot(database.sessions), language=language)

        try:
            service = SummarizationService(
                settings, database.sessions, LLMManager.providers_for(settings, http), rag=enrich, glossary=glossary,
            )
            report = await service.run(
                news, mode=args.mode, provider=args.provider, model=args.model, allow_external=args.allow_paid,
                summary_language=args.summary_language, rag_language=language,
            )
            if args.test_id:
                report["test_id"] = args.test_id
            if args.show_calls:
                from uuid import UUID

                report["calls"] = [
                    {"agent": row.agent_name, "provider": row.provider, "model": row.model_name,
                     "status": row.status, "latency_ms": row.latency_ms, "tokens_used": row.tokens_used,
                     "created_at": row.created_at.isoformat(), "prompt_hash": row.prompt_hash,
                     "response_hash": row.response_hash, "parameters": row.parameters}
                    for row in await service.calls.list_for_request(UUID(report["request_id"]))
                ]
            return report
        finally:
            await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--news", type=Path, help="Файл с новостью UTF-8")
    source.add_argument("--test-id", help="UUID или исходный ID тестовой новости XSum")
    source.add_argument("--check-ollama", action="store_true")
    parser.add_argument("--test-file", type=Path, default=TEST_FILE)
    parser.add_argument("--rag-language", "--language", dest="language", help="Языковой фильтр RAG и справочника")
    parser.add_argument("--summary-language", choices=("auto", "ru", "en"), default="auto", help="Язык резюме")
    parser.add_argument("--mode", choices=("baseline", "full"), default="full")
    parser.add_argument("--provider", choices=("ollama", "transformers", "anthropic", "gemini"))
    parser.add_argument("--model")
    parser.add_argument("--allow-paid", action="store_true", help="Явно разрешить платный вызов выбранного внешнего API")
    parser.add_argument("--show-calls", action="store_true")
    args = parser.parse_args()
    try:
        report = asyncio.run(_run(args))
    except (OSError, SQLAlchemyError, ValueError, LLMError) as exc:
        detail = str(exc) if isinstance(exc, LLMError) else "Проверьте параметры, входной файл и миграции БД."
        parser.exit(1, f"Запуск не выполнен ({type(exc).__name__}). {detail}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] in {"failed", "cancelled", "incomplete_configuration"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
