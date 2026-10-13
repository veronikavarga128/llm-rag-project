#!/usr/bin/env python3
"""A kérdéskészlet (eval/questions.jsonl) ellenőrzése a korpuszhoz képest.

Csak standard könyvtárat használ. Használat a repó gyökeréből:

    python eval/validate_questions.py

Ellenőrzi, hogy
  - az azonosítók egyediek, a kötelező mezők megvannak, a típusok érvényesek;
  - minden hivatkozott cikk létezik a korpuszban;
  - minden `evidence.quote` SZÓ SZERINT szerepel a megfelelő cikk szövegében;
  - a `gold_titles` megegyezik az evidence cikkeivel;
  - a "nincs válasz" kérdéseknek nincs evidence-ük, a többinek van.

Ha a korpuszt újra letöltöd, és a Wikipédia közben változott, ez a szkript jelzi, mely
kérdések bizonyítéka nem található meg többé.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "raw" / "corpus.jsonl"
QUESTIONS = ROOT / "eval" / "questions.jsonl"

TYPES = {"tény", "több cikk", "nincs válasz"}
POPULARITY = {"ismert", "kevésbé ismert"}
REQUIRED = ("id", "question", "answer", "type", "popularity", "gold_titles", "evidence")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Nem találom: {path}")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    corpus = {doc["title"]: doc["text"] for doc in load_jsonl(CORPUS)}
    questions = load_jsonl(QUESTIONS)
    errors: list[str] = []
    seen_ids: set[str] = set()

    for item in questions:
        qid = item.get("id", "?")
        for field in REQUIRED:
            if field not in item:
                errors.append(f"{qid}: hiányzó mező: {field}")
        if qid in seen_ids:
            errors.append(f"{qid}: duplikált azonosító")
        seen_ids.add(qid)

        if item.get("type") not in TYPES:
            errors.append(f"{qid}: érvénytelen típus: {item.get('type')!r}")
        if item.get("popularity") not in POPULARITY:
            errors.append(f"{qid}: érvénytelen popularity: {item.get('popularity')!r}")

        evidence = item.get("evidence", [])
        if item.get("type") == "nincs válasz":
            if evidence or item.get("gold_titles"):
                errors.append(f"{qid}: 'nincs válasz' kérdésnek nem lehet evidence-e vagy gold_titles-e")
            for title in item.get("related_titles", []):
                if title not in corpus:
                    errors.append(f"{qid}: a related cikk nincs a korpuszban: {title}")
        else:
            if not evidence:
                errors.append(f"{qid}: nincs evidence")
            for ev in evidence:
                title, quote = ev["title"], ev["quote"]
                if title not in corpus:
                    errors.append(f"{qid}: a cikk nincs a korpuszban: {title}")
                elif quote not in corpus[title]:
                    errors.append(f"{qid}: az idézet nem található a(z) '{title}' cikkben: {quote!r}")
            if sorted(item.get("gold_titles", [])) != sorted({ev["title"] for ev in evidence}):
                errors.append(f"{qid}: a gold_titles nem egyezik az evidence cikkeivel")

    by_type: dict[str, int] = {}
    for item in questions:
        by_type[item.get("type", "?")] = by_type.get(item.get("type", "?"), 0) + 1
    print(f"{len(questions)} kérdés, típusok szerint: {by_type}")

    if errors:
        print(f"\n{len(errors)} hiba:")
        for err in errors:
            print(" -", err)
        sys.exit(1)
    print("Minden rendben: a bizonyítékok szó szerint megvannak a korpuszban.")


if __name__ == "__main__":
    main()