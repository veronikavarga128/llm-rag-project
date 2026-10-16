"""Keresési változatok a kísérletekhez: dense, BM25, hibrid (RRF) és opcionális reranker.

Minden retriever `search(query, k) -> list[Hit]` metódust ad. A BM25 és a reranker a chunk
címét és szakaszcímét is látja (a dense embedding alapértelmezésben szintén), így a
változatok összehasonlíthatók.

Retriever-nevek (lásd build_retriever):
    dense          Qdrant vektorkeresés (a gyűjtemény embeddingjével)
    bm25           BM25, szóalakok pontos egyezésével
    bm25_stem5     BM25, a szavak első 5 karakterére vágva (egyszerű "szótövezés": a magyar
                   toldalékolás miatt a "kastély", "kastélyt", "kastélyban" egy tőre kerül)
    hybrid         dense + bm25, Reciprocal Rank Fusion-nel
    hybrid_stem5   dense + bm25_stem5, Reciprocal Rank Fusion-nel
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

RETRIEVER_NAMES = ("dense", "bm25", "bm25_stem5", "hybrid", "hybrid_stem5")
STEM_LENGTH = 5
RRF_K = 60
HYBRID_POOL = 50
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

_TOKEN_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class IndexedChunk:
    doc_id: str
    title: str
    section: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class Hit:
    score: float
    doc_id: str
    title: str
    section: str
    chunk_index: int
    text: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.doc_id, self.chunk_index)


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


def tokenize(text: str, stem: int = 0) -> list[str]:
    """Kisbetűsít és szavakra bont; `stem` > 0 esetén a szavakat az első `stem` karakterre vágja."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t[:stem] for t in tokens] if stem else tokens


def searchable_text(item) -> str:
    """A keresésben (BM25, reranker) használt szöveg: cím + szakasz + chunk-szöveg."""
    return f"{item.title} {item.section} {item.text}".replace("  ", " ")


def _hit_from(chunk: IndexedChunk, score: float) -> Hit:
    return Hit(score, chunk.doc_id, chunk.title, chunk.section, chunk.chunk_index, chunk.text)


def fetch_chunks(client, collection: str) -> list[IndexedChunk]:
    """A gyűjtemény összes chunkja (determinisztikus sorrendben)."""
    chunks: list[IndexedChunk] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=500, offset=offset, with_payload=True, with_vectors=False
        )
        for p in points:
            payload = p.payload
            chunks.append(
                IndexedChunk(
                    doc_id=payload["doc_id"],
                    title=payload["title"],
                    section=payload.get("section", ""),
                    chunk_index=payload["chunk_index"],
                    text=payload["text"],
                )
            )
        if offset is None:
            break
    chunks.sort(key=lambda c: (c.doc_id, c.chunk_index))
    return chunks


class DenseRetriever:
    def __init__(self, client, embedder, collection: str) -> None:
        self.client, self.embedder, self.collection = client, embedder, collection

    def search(self, query: str, k: int) -> list[Hit]:
        from rag.search import dense_search

        hits = dense_search(self.client, self.embedder, self.collection, query, k)
        return [Hit(h.score, h.doc_id, h.title, h.section, h.chunk_index, h.text) for h in hits]


class BM25Retriever:
    def __init__(self, chunks: list[IndexedChunk], stem: int = 0) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks, self.stem = chunks, stem
        self.bm25 = BM25Okapi([tokenize(searchable_text(c), stem) for c in chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query, self.stem))
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]
        return [_hit_from(self.chunks[i], float(scores[i])) for i in order]


def rrf_fuse(rankings: list[list[Hit]], k: int, rrf_k: int = RRF_K) -> list[Hit]:
    """Reciprocal Rank Fusion: egy találat pontszáma a rangjai 1/(rrf_k + rang) értékeinek összege."""
    scores: dict[tuple[str, int], float] = {}
    first: dict[tuple[str, int], Hit] = {}
    for hits in rankings:
        for rank, hit in enumerate(hits, 1):
            scores[hit.key] = scores.get(hit.key, 0.0) + 1.0 / (rrf_k + rank)
            first.setdefault(hit.key, hit)
    order = sorted(scores, key=lambda key: (-scores[key], key))[:k]
    return [replace(first[key], score=scores[key]) for key in order]


class HybridRetriever:
    def __init__(self, dense: Retriever, sparse: Retriever, pool: int = HYBRID_POOL) -> None:
        self.dense, self.sparse, self.pool = dense, sparse, pool

    def search(self, query: str, k: int) -> list[Hit]:
        pool = max(self.pool, k)
        return rrf_fuse([self.dense.search(query, pool), self.sparse.search(query, pool)], k)


class Reranker:
    """Cross-encoder újrarendező (sentence-transformers CrossEncoder)."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, max_length: int = 512, batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        self.model_name, self.batch_size = model_name, batch_size
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        pairs = [(query, searchable_text(h)) for h in hits]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        order = sorted(range(len(hits)), key=lambda i: (-float(scores[i]), i))[:k]
        return [replace(hits[i], score=float(scores[i])) for i in order]


class RerankedRetriever:
    """Az alap retriever top-N találatát újrarendezi, és a legjobb k-t adja vissza."""

    def __init__(self, base: Retriever, reranker, top_n: int = 20) -> None:
        self.base, self.reranker, self.top_n = base, reranker, top_n

    def search(self, query: str, k: int) -> list[Hit]:
        hits = self.base.search(query, max(self.top_n, k))
        return self.reranker.rerank(query, hits, k)


def build_retriever(
    name: str, chunks: list[IndexedChunk], dense_factory: Callable[[], Retriever]
) -> Retriever:
    """A név alapján összeállítja a retrievert. A dense részt csak akkor hozza létre, ha kell."""
    if name not in RETRIEVER_NAMES:
        raise ValueError(f"Ismeretlen retriever: {name!r}. Lehetséges: {', '.join(RETRIEVER_NAMES)}")
    stem = STEM_LENGTH if name.endswith("stem5") else 0
    if name == "dense":
        return dense_factory()
    if name.startswith("bm25"):
        return BM25Retriever(chunks, stem)
    return HybridRetriever(dense_factory(), BM25Retriever(chunks, stem))