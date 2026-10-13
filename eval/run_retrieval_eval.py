#!/usr/bin/env python3
"""Retrieval-kiértékelés: a kérdéskészlet végigfuttatása egy Qdrant gyűjteményen.

Használat a repó gyökeréből (futó Qdrant és betöltött gyűjtemény mellett):

    python eval/run_retrieval_eval.py
    python eval/run_retrieval_eval.py --collection corpus_c800_o150 --k 10

Mit mér? A `nincs válasz` típusú kérdéseket kihagyja (azoknál nincs helyes találat), a többinél:

  Cikkszint   a találatok között szerepel-e a helyes cikk (`gold_titles`)
              Recall@k = a helyes cikkek hányada az első k chunk között
              MRR      = az első, helyes cikkből származó chunk rangjának reciproka

  Chunkszint  a találatok között van-e olyan chunk, amely a forrásidézetet (`evidence`)
              tartalmazza. Ez szigorúbb: nem elég a jó cikk, a jó szövegrészt kell hozni.
              Recall@k = a bizonyítékok hányada, amelyeket legalább egy top-k chunk tartalmaz
              MRR      = az első, bizonyítékot tartalmazó chunk rangjának reciproka

A chunkszintű relevancia a gyűjtemény tényleges chunkjaiból számolódik (whitespace-
normalizált részszöveg-egyezéssel), ezért bármilyen chunkméretű gyűjteményre működik,
és a különböző konfigurációk eredményei összevethetők. Ha egy idézet nem fér el egyetlen
chunkban sem (pl. chunkhatáron átnyúlik), az a bizonyíték kimarad a chunkszintű mérésből,
és a szkript jelzi.

A `nincs válasz` kérdéseknél a legjobb találat hasonlósági pontszámát írja ki, hogy később
eldönthető legyen, használható-e pontszámküszöb a "nem tudom" válaszhoz.

Az eredmény részletesen a --out fájlba kerül (alapból experiments/results/<gyűjtemény>.json).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "experiments" / "results"
DEFAULT_KS = (1, 3, 5, 10)
NO_ANSWER = "nincs válasz"


@dataclass(frozen=True)
class StoredChunk:
    doc_id: str
    title: str
    chunk_index: int
    text_norm: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.doc_id, self.chunk_index)


@dataclass(frozen=True)
class Result:
    doc_id: str
    title: str
    section: str
    chunk_index: int
    score: float

    @property
    def key(self) -> tuple[str, int]:
        return (self.doc_id, self.chunk_index)


Retriever = Callable[[str, int], list[Result]]


# ----------------------------------------------------------------------- relevancia


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def evidence_chunk_keys(
    question: dict, chunks_by_title: dict[str, list[StoredChunk]]
) -> list[set[tuple[str, int]]]:
    """Bizonyítékonként azon chunkok kulcsai, amelyek tartalmazzák az idézetet."""
    result = []
    for ev in question["evidence"]:
        quote = norm(ev["quote"])
        result.append(
            {c.key for c in chunks_by_title.get(ev["title"], []) if quote in c.text_norm}
        )
    return result


def article_metrics(
    ranked_titles: list[str], gold: set[str], ks: tuple[int, ...]
) -> tuple[dict[str, float], float]:
    first = next((i for i, t in enumerate(ranked_titles, 1) if t in gold), None)
    recall = {str(k): len(gold & set(ranked_titles[:k])) / len(gold) for k in ks}
    return recall, (1.0 / first if first else 0.0)


def chunk_metrics(
    ranked_keys: list[tuple[str, int]], evidence_sets: list[set], ks: tuple[int, ...]
) -> tuple[dict[str, float], float] | None:
    usable = [s for s in evidence_sets if s]
    if not usable:
        return None
    relevant = set().union(*usable)
    first = next((i for i, key in enumerate(ranked_keys, 1) if key in relevant), None)
    recall = {
        str(k): sum(1 for s in usable if s & set(ranked_keys[:k])) / len(usable) for k in ks
    }
    return recall, (1.0 / first if first else 0.0)


# ----------------------------------------------------------------------- kiértékelés


def evaluate(
    questions: list[dict],
    chunks_by_title: dict[str, list[StoredChunk]],
    retrieve: Retriever,
    k: int,
    ks: tuple[int, ...],
) -> list[dict]:
    records = []
    for q in questions:
        results = retrieve(q["question"], k)
        record: dict = {
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "popularity": q["popularity"],
            "gold_titles": q.get("gold_titles", []),
            "related_titles": q.get("related_titles", []),
            "top1_score": results[0].score if results else None,
            "top": [
                {
                    "title": r.title,
                    "section": r.section,
                    "chunk_index": r.chunk_index,
                    "score": round(r.score, 4),
                }
                for r in results
            ],
        }
        if q["type"] != NO_ANSWER:
            gold = set(q["gold_titles"])
            recall, rr = article_metrics([r.title for r in results], gold, ks)
            record["article_recall"], record["article_rr"] = recall, rr

            evidence_sets = evidence_chunk_keys(q, chunks_by_title)
            record["unfindable_evidence"] = [
                ev["quote"] for ev, s in zip(q["evidence"], evidence_sets, strict=True) if not s
            ]
            chunk_result = chunk_metrics([r.key for r in results], evidence_sets, ks)
            if chunk_result is None:
                record["chunk_recall"], record["chunk_rr"] = None, None
            else:
                record["chunk_recall"], record["chunk_rr"] = chunk_result
        records.append(record)
    return records


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarize(records: list[dict], ks: tuple[int, ...]) -> dict:
    summary: dict = {"n": len(records)}
    for level, recall_key, rr_key in (
        ("article", "article_recall", "article_rr"),
        ("chunk", "chunk_recall", "chunk_rr"),
    ):
        usable = [r for r in records if r.get(recall_key) is not None]
        summary[level] = {
            "n": len(usable),
            **{f"R@{k}": mean([r[recall_key][str(k)] for r in usable]) for k in ks},
            "MRR": mean([r[rr_key] for r in usable]),
        }
    return summary


def group_records(records: list[dict]) -> dict[str, list[dict]]:
    answerable = [r for r in records if r["type"] != NO_ANSWER]
    groups: dict[str, list[dict]] = {"összes (válaszolható)": answerable}
    for name in sorted({r["type"] for r in answerable}):
        groups[f"típus: {name}"] = [r for r in answerable if r["type"] == name]
    for name in sorted({r["popularity"] for r in answerable}):
        groups[f"hely: {name}"] = [r for r in answerable if r["popularity"] == name]
    return groups


# ------------------------------------------------------------------------- kiírás


def print_table(title: str, level: str, summaries: dict[str, dict], ks: tuple[int, ...]) -> None:
    print(f"\n{title}")
    header = f"{'csoport':<28}{'N':>4}" + "".join(f"{'R@' + str(k):>8}" for k in ks) + f"{'MRR':>8}"
    print(header)
    print("-" * len(header))
    for name, summary in summaries.items():
        s = summary[level]
        row = f"{name:<28}{s['n']:>4}" + "".join(f"{s[f'R@{k}']:>8.3f}" for k in ks)
        print(row + f"{s['MRR']:>8.3f}")


def pick_miss_k(ks: tuple[int, ...]) -> int:
    small = [k for k in ks if k <= 5]
    return max(small) if small else ks[0]


def print_misses(records: list[dict], miss_k: int) -> None:
    misses = []
    for r in records:
        if r["type"] == NO_ANSWER:
            continue
        recall = r["chunk_recall"] if r.get("chunk_recall") is not None else r["article_recall"]
        if recall[str(miss_k)] < 1.0:
            misses.append(r)
    print(f"\nKérdések, ahol az első {miss_k} találat nem tartalmazza az összes bizonyítékot: "
          f"{len(misses)}")
    for r in misses:
        print(f"  {r['id']} [{r['type']}, {r['popularity']}] {r['question']}")
        print(f"      várt cikk: {', '.join(r['gold_titles'])}")
        for rank, hit in enumerate(r["top"][:3], 1):
            where = hit["title"] + (f" › {hit['section']}" if hit["section"] else "")
            print(f"      {rank}. [{hit['score']:.3f}] {where}")


def print_unanswerable(records: list[dict], miss_k: int) -> None:
    unanswerable = [r for r in records if r["type"] == NO_ANSWER]
    answerable = [r for r in records if r["type"] != NO_ANSWER and r["top1_score"] is not None]
    if not unanswerable:
        return
    print("\n'Nincs válasz' kérdések (retrieval-metrika nem értelmezett, a legjobb találat pontszáma érdekes):")
    for r in unanswerable:
        top = r["top"][0] if r["top"] else None
        related = set(r["related_titles"])
        in_top = any(h["title"] in related for h in r["top"][:miss_k])
        label = f"{top['title']}" if top else "-"
        print(f"  {r['id']} top1={r['top1_score']:.3f} ({label}); a kapcsolódó cikk az első "
              f"{miss_k} között: {'igen' if in_top else 'nem'}")
    print(
        f"  Átlagos top-1 pontszám: válaszolható {mean([r['top1_score'] for r in answerable]):.3f}"
        f" | nincs válasz {mean([r['top1_score'] for r in unanswerable]):.3f}"
    )


# -------------------------------------------------------------------- Qdrant / main


def load_all_chunks(client, collection: str) -> list[StoredChunk]:
    chunks: list[StoredChunk] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=500, offset=offset, with_payload=True, with_vectors=False
        )
        for p in points:
            payload = p.payload
            chunks.append(
                StoredChunk(
                    doc_id=payload["doc_id"],
                    title=payload["title"],
                    chunk_index=payload["chunk_index"],
                    text_norm=norm(payload["text"]),
                )
            )
        if offset is None:
            return chunks


def index_by_title(chunks: list[StoredChunk]) -> dict[str, list[StoredChunk]]:
    by_title: dict[str, list[StoredChunk]] = defaultdict(list)
    for chunk in chunks:
        by_title[chunk.title].append(chunk)
    return by_title


def parse_ks(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(x) for x in value.split(",") if x.strip()}))


def main() -> None:
    from qdrant_client import QdrantClient

    from rag.config import get_settings
    from rag.embedding import Embedder
    from rag.search import dense_search

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Retrieval-kiértékelés Qdrant gyűjteményen.")
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--k", type=int, default=10, help="ennyi találatot kér le kérdésenként")
    parser.add_argument("--ks", type=parse_ks, default=DEFAULT_KS, help="a Recall@k értékei, pl. 1,3,5,10")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ks: tuple[int, ...] = args.ks
    if not ks or max(ks) > args.k:
        sys.exit(f"A --ks értékei nem lehetnek nagyobbak a --k értékénél ({args.k}).")

    client = QdrantClient(url=settings.qdrant_url)
    if not client.collection_exists(args.collection):
        sys.exit(f"A '{args.collection}' gyűjtemény nem létezik. Előbb: python -m rag.ingest --recreate")

    with args.questions.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    chunks = load_all_chunks(client, args.collection)
    chunks_by_title = index_by_title(chunks)
    avg_len = mean([len(c.text_norm) for c in chunks])
    print(f"Gyűjtemény: {args.collection} ({len(chunks)} chunk, átlag {avg_len:.0f} karakter)")
    print(f"Embedding-modell: {settings.embedding_model}")

    embedder = Embedder(settings.embedding_model)

    def retrieve(question: str, k: int) -> list[Result]:
        hits = dense_search(client, embedder, args.collection, question, k)
        return [Result(h.doc_id, h.title, h.section, h.chunk_index, h.score) for h in hits]

    records = evaluate(questions, chunks_by_title, retrieve, args.k, ks)

    groups = group_records(records)
    summaries = {name: summarize(recs, ks) for name, recs in groups.items()}
    print_table("CIKKSZINT (a helyes cikk benne van-e a találatok között)", "article", summaries, ks)
    print_table("CHUNKSZINT (a bizonyítékot tartalmazó chunk benne van-e)", "chunk", summaries, ks)

    unfindable = [(r["id"], q) for r in records for q in r.get("unfindable_evidence", [])]
    if unfindable:
        print(f"\nFigyelem: {len(unfindable)} bizonyíték nem fér el egyetlen chunkban sem, "
              "ezek kimaradnak a chunkszintű mérésből:")
        for qid, quote in unfindable:
            print(f"  {qid}: {quote[:90]}")

    miss_k = pick_miss_k(ks)
    print_misses(records, miss_k)
    print_unanswerable(records, miss_k)

    out = args.out or RESULTS_DIR / f"{args.collection}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "collection": args.collection,
        "embedding_model": settings.embedding_model,
        "n_chunks": len(chunks),
        "avg_chunk_chars": round(avg_len, 1),
        "k": args.k,
        "ks": list(ks),
        "summary": summaries,
        "unfindable_evidence": [{"id": qid, "quote": quote} for qid, quote in unfindable],
        "per_question": records,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nRészletes eredmény: {out}")


if __name__ == "__main__":
    main()