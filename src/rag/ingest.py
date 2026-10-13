"""A korpusz chunkolása, embeddingje és betöltése Qdrantba.

Használat (a repó gyökeréből, aktivált virtuális környezetben, futó Qdrant mellett):

    python -m rag.ingest --recreate
    python -m rag.ingest --recreate --chunk-size 800 --overlap 150 --collection corpus_c800_o150

Kísérletekhez használj külön gyűjteményt (--collection) minden konfigurációhoz, így az
eredmények egymás mellett összehasonlíthatók maradnak.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from rag.chunking import chunk_document
from rag.config import DEFAULT_CORPUS, get_settings
from rag.embedding import Embedder


def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Nem találom a korpuszt: {path}\nElőbb futtasd a scripts/download_corpus.py-t.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_chunks(docs: list[dict], size: int, overlap: int, title_prefix: bool) -> list[dict]:
    """Chunk-rekordok: `embed_text` kerül az embeddingbe, a `payload` a Qdrantba."""
    records: list[dict] = []
    for doc in docs:
        for chunk in chunk_document(doc["text"], size=size, overlap=overlap):
            if title_prefix:
                header = doc["title"] + (f" – {chunk.section}" if chunk.section else "")
                embed_text = f"{header}\n\n{chunk.text}"
            else:
                embed_text = chunk.text
            records.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc['id']}#{chunk.index}")),
                    "embed_text": embed_text,
                    "payload": {
                        "doc_id": doc["id"],
                        "title": doc["title"],
                        "url": doc.get("url", ""),
                        "section": chunk.section,
                        "chunk_index": chunk.index,
                        "text": chunk.text,
                    },
                }
            )
    return records


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Korpusz betöltése Qdrantba.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--title-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="a cikk címét (és szakaszát) az embedding szövegének elejére teszi (alapból be)",
    )
    parser.add_argument("--recreate", action="store_true", help="a meglévő gyűjtemény törlése és újraépítése")
    args = parser.parse_args()

    docs = load_corpus(args.corpus)
    records = build_chunks(docs, args.chunk_size, args.overlap, args.title_prefix)
    lengths = [len(r["payload"]["text"]) for r in records]
    print(
        f"{len(docs)} cikk -> {len(records)} chunk "
        f"(átlag {sum(lengths) / len(lengths):.0f}, max {max(lengths)} karakter)"
    )

    client = QdrantClient(url=settings.qdrant_url)
    exists = client.collection_exists(args.collection)
    if exists and not args.recreate:
        sys.exit(
            f"A '{args.collection}' gyűjtemény már létezik. Add meg a --recreate kapcsolót az "
            "újraépítéshez, vagy válassz másik nevet a --collection-nel."
        )

    print(f"Embedding-modell betöltése: {settings.embedding_model}")
    embedder = Embedder(settings.embedding_model)

    if exists:
        client.delete_collection(args.collection)
    client.create_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
    )

    for start in tqdm(range(0, len(records), args.batch_size), desc="Betöltés", unit="batch"):
        batch = records[start : start + args.batch_size]
        vectors = embedder.embed_passages([r["embed_text"] for r in batch])
        client.upsert(
            collection_name=args.collection,
            points=[
                PointStruct(id=r["id"], vector=v, payload=r["payload"])
                for r, v in zip(batch, vectors, strict=True)
            ],
        )

    count = client.count(args.collection, exact=True).count
    print(f"Kész: {count} pont a '{args.collection}' gyűjteményben.")


if __name__ == "__main__":
    main()