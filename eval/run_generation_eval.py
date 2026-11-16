#!/usr/bin/env python3
"""Generálás-kiértékelés: RAG-os és kontextus nélküli válaszok helyi Ollama-modellel.

Minden kérdésre két választ kér a modelltől:
    rag     a kereső top-k chunkja a promptban, "csak a forrásból válaszolj" utasítással
    norag   ugyanaz a kérdés kontextus nélkül (baseline: mennyit tud a modell magától)

Használat a repó gyökeréből (futó Qdrant és Ollama mellett):

    python eval/run_generation_eval.py --limit 5          # gyors próba az első 5 kérdéssel
    python eval/run_generation_eval.py                    # mind az 50 kérdés
    python eval/run_generation_eval.py --model qwen2.5:3b --retriever dense --k 3
    python eval/run_generation_eval.py --prompt balanced  # enyhébb RAG-prompt (külön fájlba ment)
    python eval/run_generation_eval.py --report-only      # csak az összesítő a mentett válaszokból

A futás folytatható: a már megválaszolt (kérdés, mód) párokat nem kéri le újra.
Kimenet (experiments/generation/ mappa):
    <címke>.jsonl   minden válasz (a nyers adat)
    <címke>.csv     kézi értékeléshez (Excelben megnyitható, a "kézi" oszlopot töltsd ki)

Az automatikus jelek (szótő-egyezés, elutasítás-felismerés) durva közelítések; a végső
helyességi arány a kézi értékelésből számolandó.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

QUESTIONS = ROOT / "eval" / "questions.jsonl"
OUT_DIR = ROOT / "experiments" / "generation"
COVERAGE_THRESHOLD = 0.6  # ennyi tartalmas szót kell az elvárt válaszból tartalmaznia
MODES = ("rag", "norag")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", text).strip("-")


def context_has_evidence(question: dict, hits: list[dict]) -> bool:
    """Szerepel-e a lekért chunkok között a bizonyítékot tartalmazó szöveg."""
    if not question.get("evidence"):
        return False
    quotes = [(ev["title"], norm(ev["quote"])) for ev in question["evidence"]]
    return all(
        any(title == h["title"] and quote in norm(h["text"]) for h in hits)
        for title, quote in quotes
    )


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "-"


def summarize(records: list[dict]) -> str:
    from rag.generate import answer_coverage

    lines = []
    for mode in MODES:
        rows = [r for r in records if r["mode"] == mode]
        if not rows:
            continue
        answerable = [r for r in rows if r["answerable"]]
        unanswerable = [r for r in rows if not r["answerable"]]
        for r in answerable:
            r["_correct"] = answer_coverage(r["gold_answer"], r["answer"]) >= COVERAGE_THRESHOLD
        lines.append(f"== {mode} ({len(rows)} válasz, átlag {mean(r['seconds'] for r in rows):.1f} mp/válasz)")
        lines.append(
            f"  válaszolható kérdések, automatikus 'helyes' jel : "
            f"{pct(sum(r['_correct'] for r in answerable), len(answerable))}"
        )
        lines.append(
            f"  válaszolható kérdések, elutasított válasz       : "
            f"{pct(sum(r['refusal'] for r in answerable), len(answerable))}"
        )
        if mode == "rag":
            with_ev = [r for r in answerable if r["context_has_evidence"]]
            without = [r for r in answerable if not r["context_has_evidence"]]
            lines.append(
                f"  - ha a kontextusban benne volt a bizonyíték     : "
                f"{pct(sum(r['_correct'] for r in with_ev), len(with_ev))} helyes"
            )
            lines.append(
                f"  - ha a kontextusból hiányzott a bizonyíték      : "
                f"{pct(sum(r['_correct'] for r in without), len(without))} helyes"
            )
        lines.append(
            f"  'nincs válasz' kérdések, helyesen elutasítva    : "
            f"{pct(sum(r['refusal'] for r in unanswerable), len(unanswerable))}"
        )
        lines.append(
            f"  'nincs válasz' kérdések, kitalált válasz        : "
            f"{pct(sum(not r['refusal'] for r in unanswerable), len(unanswerable))}"
        )
        lines.append("")
    lines.append(
        f"Az automatikus 'helyes' jel: az elvárt válasz tartalmas szavainak legalább "
        f"{COVERAGE_THRESHOLD:.0%}-a (5 karakteres szótő-egyezéssel) szerepel a válaszban."
    )
    return "\n".join(lines)


def write_csv(records: list[dict], path: Path) -> None:
    from rag.generate import answer_coverage

    manual: dict[tuple[str, str], str] = {}
    if path.exists():  # a már kitöltött kézi értékelést ne vesszük el
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                manual[(row["id"], row["mód"])] = row.get("kézi", "")
    fields = ["id", "típus", "kérdés", "elvárt válasz", "mód", "modell válasza",
              "elutasítás", "szó-egyezés", "van bizonyíték a kontextusban", "kézi"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for r in sorted(records, key=lambda r: (r["id"], r["mode"])):
            writer.writerow({
                "id": r["id"], "típus": r["type"], "kérdés": r["question"],
                "elvárt válasz": r["gold_answer"], "mód": r["mode"],
                "modell válasza": r["answer"].replace("\n", " "),
                "elutasítás": "igen" if r["refusal"] else "nem",
                "szó-egyezés": f"{answer_coverage(r['gold_answer'], r['answer']):.2f}",
                "van bizonyíték a kontextusban":
                    ("igen" if r["context_has_evidence"] else "nem") if r["mode"] == "rag" else "",
                "kézi": manual.get((r["id"], r["mode"]), ""),
            })


def main() -> None:
    from rag.config import get_settings
    from rag.generate import (
        DEFAULT_MODEL,
        DEFAULT_OLLAMA_URL,
        RAG_PROMPTS,
        OllamaError,
        build_plain_messages,
        build_rag_messages,
        is_refusal,
        ollama_chat,
    )
    from rag.retrieval import RETRIEVER_NAMES

    settings = get_settings()
    parser = argparse.ArgumentParser(description="RAG és baseline válaszgenerálás kiértékelése.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama-modell (alapból: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument("--retriever", choices=RETRIEVER_NAMES, default="hybrid_stem5")
    parser.add_argument("--k", type=int, default=5, help="ennyi chunk kerül a promptba (alapból: 5)")
    parser.add_argument("--prompt", choices=sorted(RAG_PROMPTS), default="strict",
                        help="a RAG-prompt változata: strict (alap) vagy balanced")
    parser.add_argument("--num-ctx", type=int, default=4096, help="a modell kontextusablaka tokenben")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--limit", type=int, default=None, help="csak az első N kérdés (próbához)")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--tag", default=None, help="a kimeneti fájlok neve (alapból a beállításokból)")
    args = parser.parse_args()

    base_tag = f"{args.model}__{args.collection}__{args.retriever}__k{args.k}"
    if args.prompt != "strict":
        base_tag += f"__{args.prompt}"
    tag = args.tag or safe_name(base_tag)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path, csv_path = OUT_DIR / f"{tag}.jsonl", OUT_DIR / f"{tag}.csv"

    with args.questions.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        questions = questions[: args.limit]

    records = load_records(jsonl_path)

    if not args.report_only:
        done = {(r["id"], r["mode"]) for r in records}
        todo = [(q, m) for q in questions for m in args.modes if (q["id"], m) not in done]
        print(f"Modell: {args.model} | gyűjtemény: {args.collection} | kereső: {args.retriever} | k={args.k}")
        print(f"Megválaszolandó: {len(todo)} (már kész: {len(done)})")

        retrieve = None
        if todo and "rag" in args.modes:
            retrieve = open_retriever(args, settings)

        try:
            with jsonl_path.open("a", encoding="utf-8") as out:
                for number, (q, mode) in enumerate(todo, 1):
                    hits: list[dict] = []
                    if mode == "rag":
                        hits = retrieve(q["question"])
                        messages = build_rag_messages(q["question"], hits_as_objects(hits), args.prompt)
                    else:
                        messages = build_plain_messages(q["question"])
                    gen = ollama_chat(messages, args.model, args.ollama_url, num_ctx=args.num_ctx)
                    record = {
                        "id": q["id"], "type": q["type"], "question": q["question"],
                        "gold_answer": q["answer"], "answerable": q["type"] != "nincs válasz",
                        "mode": mode, "answer": gen.text, "refusal": is_refusal(gen.text),
                        "seconds": round(gen.seconds, 2), "output_tokens": gen.output_tokens,
                        "context_has_evidence": context_has_evidence(q, hits) if mode == "rag" else False,
                        "retrieved": [{"title": h["title"], "section": h["section"],
                                       "chunk_index": h["chunk_index"]} for h in hits],
                        "model": args.model,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    records.append(record)
                    print(f"[{number}/{len(todo)}] {q['id']} {mode:<5} ({gen.seconds:.0f} mp): "
                          f"{gen.text[:90].replace(chr(10), ' ')}")
        except OllamaError as exc:
            sys.exit(f"\n{exc}\n(A kész válaszok mentve vannak, újraindítással folytatható.)")

    wanted = {q["id"] for q in questions}
    records = [r for r in records if r["id"] in wanted]
    if not records:
        sys.exit("Még nincs mentett válasz.")
    write_csv(records, csv_path)
    print("\n" + summarize(records))
    print(f"\nFájlok: {jsonl_path}\n        {csv_path}")


def hits_as_objects(hits: list[dict]):
    from types import SimpleNamespace

    return [SimpleNamespace(**h) for h in hits]


def open_retriever(args, settings):
    """Összeállítja a keresőt, és egy `kérdés -> találatok (dict)` függvényt ad vissza."""
    from qdrant_client import QdrantClient

    from rag.retrieval import DenseRetriever, build_retriever, fetch_chunks

    client = QdrantClient(url=settings.qdrant_url)
    try:
        exists = client.collection_exists(args.collection)
    except Exception as exc:
        sys.exit(
            f"Nem érem el a Qdrantot ({settings.qdrant_url}): {type(exc).__name__}.\n"
            "Indítsd el a Docker Desktopot, majd a repó gyökeréből: docker compose up -d"
        )
    if not exists:
        sys.exit(f"A '{args.collection}' gyűjtemény nem létezik. Előbb: python -m rag.ingest --recreate")

    indexed = fetch_chunks(client, args.collection)

    def dense_factory():
        from rag.embedding import Embedder

        return DenseRetriever(client, Embedder(settings.embedding_model), args.collection)

    retriever = build_retriever(args.retriever, indexed, dense_factory)

    def retrieve(question: str) -> list[dict]:
        return [
            {"title": h.title, "section": h.section, "chunk_index": h.chunk_index, "text": h.text}
            for h in retriever.search(question, args.k)
        ]

    return retrieve


if __name__ == "__main__":
    main()