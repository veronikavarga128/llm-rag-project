"""Egyszerű (dense) keresés a betöltött gyűjteményben – ellenőrzéshez és későbbi újrahasznosításra.

Használat:

    python -m rag.search "Melyik kastélyt építtette a Festetics család?"
    python -m rag.search "diósgyőri vár története" --k 8
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from qdrant_client import QdrantClient

from rag.config import get_settings
from rag.embedding import Embedder


@dataclass(frozen=True)
class Hit:
    score: float
    doc_id: str
    title: str
    section: str
    chunk_index: int
    text: str


def dense_search(
    client: QdrantClient, embedder: Embedder, collection: str, query: str, k: int
) -> list[Hit]:
    response = client.query_points(
        collection_name=collection,
        query=embedder.embed_query(query),
        limit=k,
        with_payload=True,
    )
    return [
        Hit(
            score=point.score,
            doc_id=point.payload["doc_id"],
            title=point.payload["title"],
            section=point.payload.get("section", ""),
            chunk_index=point.payload["chunk_index"],
            text=point.payload["text"],
        )
        for point in response.points
    ]


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Keresés a Qdrant gyűjteményben.")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--collection", default=settings.collection)
    args = parser.parse_args()

    client = QdrantClient(url=settings.qdrant_url)
    if not client.collection_exists(args.collection):
        sys.exit(f"A '{args.collection}' gyűjtemény nem létezik. Előbb futtasd: python -m rag.ingest --recreate")

    embedder = Embedder(settings.embedding_model)
    for rank, hit in enumerate(dense_search(client, embedder, args.collection, args.query, args.k), 1):
        where = hit.title + (f" › {hit.section}" if hit.section else "")
        snippet = hit.text.replace("\n", " ")
        print(f"{rank}. [{hit.score:.3f}] {where}")
        print(f"   {snippet[:240]}{'...' if len(snippet) > 240 else ''}\n")


if __name__ == "__main__":
    main()