"""Сравнить BM25, cosine Qdrant и гибридную выдачу на готовом снимке."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from backend.core.config import Settings
from backend.data.embeddings import EmbeddingError, Embeddings
from backend.retrieval.documents import Document, IndexIntegrityError
from backend.retrieval.index_bundle import open_indexes
from backend.retrieval.vector_index import VectorIndexError
from scripts.build_indexes import TEST_FILE, load_holdout


def _row(document: Document, score: float, rank: int) -> dict:
    return {
        "rank": rank, "id": str(document.id), "source_document_id": document.source_document_id,
        "score": round(score, 8), "excerpt": document.source_text[:110].strip(),
        "category": document.category, "language": document.language,
        "source_dataset": document.source_dataset,
    }


async def _run(args: argparse.Namespace) -> dict:
    settings = Settings()
    embeddings = Embeddings(settings)
    holdout = load_holdout(args.test_file)
    filters = {"language": args.language, "category": args.category}
    modes = ("qdrant", "bm25", "rrf") if args.mode == "all" else (args.mode,)
    async with open_indexes(settings, holdout, embeddings) as (retriever, manifest):
        if any(mode != "bm25" for mode in modes):
            await embeddings.embed(args.query)
        report = {"query": args.query, "version": manifest.version, "mode": manifest.mode, "search": {}}
        for mode in modes:
            started = perf_counter()
            if mode == "rrf":
                hits = await retriever.search(args.query, args.top_k, **filters)
                elapsed = perf_counter() - started
                rows = [{**_row(hit.document, hit.score, i), "bm25_rank": hit.bm25_rank, "qdrant_rank": hit.qdrant_rank}
                        for i, hit in enumerate(hits, 1)]
            else:
                lookup = retriever.semantic if mode == "qdrant" else retriever.lexical
                matches = await lookup(args.query, args.top_k, **filters)
                elapsed = perf_counter() - started
                rows = []
                for i, match in enumerate(matches, 1):
                    document = retriever.bm25.get(match.id)
                    if document is None:
                        raise IndexIntegrityError(f"Документ {match.id} отсутствует в снимке BM25.")
                    rows.append(_row(document, match.score, i))
            report["search"][mode] = {"milliseconds": round(elapsed * 1000, 3), "hits": rows}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--mode", choices=("all", "qdrant", "bm25", "rrf"), default="all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--language")
    parser.add_argument("--category")
    parser.add_argument("--test-file", type=Path, default=TEST_FILE)
    args = parser.parse_args()
    if not args.query.strip() or args.top_k < 1:
        parser.error("Нужны непустой запрос и положительный top-k.")
    try:
        result = asyncio.run(_run(args))
    except (EmbeddingError, IndexIntegrityError, VectorIndexError, OSError, ValueError) as exc:
        parser.exit(1, f"Ошибка поиска: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
