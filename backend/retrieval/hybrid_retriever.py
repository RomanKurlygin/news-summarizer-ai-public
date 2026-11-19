"""Объединение лексической и векторной выдач и подготовка примеров."""

import asyncio
from dataclasses import dataclass
from uuid import UUID

from backend.agents.state import AgentState, RagSearchMode
from backend.data.embeddings import Embeddings
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.documents import Document, Hit, Holdout, IndexIntegrityError
from backend.retrieval.vector_index import VectorIndex
from backend.utils.corpus_text import clean_text, content_hash
from backend.utils.rag_context import MAX_CONTEXT_CHARS, MAX_EXAMPLES, limit_style_examples, reference_style

CROSS_LANGUAGE_MIN_SCORE = 0.30


@dataclass(frozen=True)
class RankedHit:
    id: UUID
    score: float
    bm25_rank: int | None
    qdrant_rank: int | None


@dataclass(frozen=True)
class HybridHit:
    document: Document
    score: float
    bm25_rank: int | None
    qdrant_rank: int | None
    cosine: float | None = None


def reciprocal_rank_fusion(bm25: list[Hit], qdrant: list[Hit], k: int = 60) -> list[RankedHit]:
    if k < 1:
        raise ValueError("Константа RRF k должна быть положительной.")
    ranked = [
        {doc_id: rank for rank, doc_id in enumerate(dict.fromkeys(hit.id for hit in channel), 1)}
        for channel in (bm25, qdrant)
    ]
    result = [
        RankedHit(doc_id, sum(1 / (k + ranks[doc_id]) for ranks in ranked if doc_id in ranks),
                  ranked[0].get(doc_id), ranked[1].get(doc_id))
        for doc_id in ranked[0].keys() | ranked[1].keys()
    ]
    return sorted(result, key=lambda hit: (-hit.score, hit.id))


class HybridRetriever:
    def __init__(self, bm25: BM25Index, vector: VectorIndex, embeddings: Embeddings, holdout: Holdout) -> None:
        self.bm25 = bm25
        self.vector = vector
        self.embeddings = embeddings
        self.holdout = holdout

    async def lexical(self, query: str, top_k: int = 5, *, language: str | None = None,
                      category: str | None = None) -> list[Hit]:
        return await asyncio.to_thread(
            self.bm25.search, query, top_k, language=language, category=category, exclude_self=True,
        )

    async def semantic(self, query: str, top_k: int = 5, *, language: str | None = None,
                       category: str | None = None) -> list[Hit]:
        if top_k < 1:
            raise ValueError("top_k должен быть положительным.")
        if not clean_text(query):
            return []
        vector = await self.embeddings.embed(query)
        return await self.vector.search(
            vector, top_k, language=language, category=category, excluded_hash=content_hash(query),
        )

    async def search(self, query: str, top_k: int = 5, *, rrf_k: int = 60, candidate_k: int = 20,
                     language: str | None = None, category: str | None = None) -> list[HybridHit]:
        results, _ = await self.search_with_mode(query, top_k, rrf_k=rrf_k, candidate_k=candidate_k,
                                                 language=language, category=category)
        return results

    async def search_with_mode(self, query: str, top_k: int = 5, *, rrf_k: int = 60, candidate_k: int = 20,
                               language: str | None = None, category: str | None = None, cross_language: bool = False
                               ) -> tuple[list[HybridHit], RagSearchMode]:
        if top_k < 1 or candidate_k < 1 or rrf_k < 1:
            raise ValueError("Параметры глубины поиска и RRF должны быть положительными.")
        if not clean_text(query):
            return [], "no_results"
        depth = max(candidate_k, top_k)
        lexical, vector = await asyncio.gather(
            self.lexical(query, depth, language=language, category=category), self.embeddings.embed(query),
        )
        query_hash = content_hash(query)
        semantic = await self.vector.search(vector, depth, language=language, category=category,
                                            excluded_hash=query_hash)
        ranked = reciprocal_rank_fusion(lexical, semantic, rrf_k)[:top_k]
        mode: RagSearchMode = ("same_language" if language else "unfiltered") if ranked else "no_results"
        if cross_language and language and len(ranked) < top_k:
            foreign = await self.vector.search(vector, top_k - len(ranked), category=category,
                                               excluded_hash=query_hash, excluded_language=language)
            foreign = [hit for hit in foreign if hit.score >= CROSS_LANGUAGE_MIN_SCORE]
            if foreign:
                mode = "mixed" if ranked else "cross_language"
                ranked.extend(reciprocal_rank_fusion([], foreign, rrf_k))
                semantic.extend(foreign)
        cosines = {hit.id: hit.score for hit in semantic}
        results = []
        for hit in ranked:
            doc = self.bm25.get(hit.id)
            if doc is None:
                raise IndexIntegrityError(f"Документ {hit.id} есть в Qdrant, но отсутствует в BM25.")
            self.holdout.validate([doc])
            if query_hash in doc.hashes():
                raise IndexIntegrityError(f"В выдачу попал исходный документ запроса {hit.id}.")
            results.append(HybridHit(doc, hit.score, hit.bm25_rank, hit.qdrant_rank, cosines.get(hit.id)))
        return results, mode

    async def enrich_state(self, state: AgentState, *, top_k: int = 5, language: str | None = None,
                           category: str | None = None, max_examples: int = MAX_EXAMPLES,
                           max_chars: int = MAX_CONTEXT_CHARS, cross_language: bool = False) -> AgentState:
        hits, mode = await self.search_with_mode(state.original_text, top_k, language=language, category=category,
                                                 cross_language=cross_language)
        return state.model_copy(deep=True, update={
            "rag_context": limit_style_examples(
                (reference_style(hit.document.target_text) for hit in hits),
                max_examples=max_examples, max_chars=max_chars,
            ),
            "rag_document_ids": [hit.document.id for hit in hits],
            "rag_search_mode": mode,
        })
