#!/usr/bin/env python3
"""Kísérletfuttató: gyűjteményeket épít, méri őket, és összesítő táblát készít.

Használat a repó gyökeréből (futó Qdrant mellett):

    python experiments/run_experiments.py --dry-run          # csak kiírja, mit futtatna
    python experiments/run_experiments.py                    # minden hiányzó kísérlet
    python experiments/run_experiments.py --only retrievers  # csak a keresési módszerek
    python experiments/run_experiments.py --only chunking prefix
    python experiments/run_experiments.py --force --only retrievers   # mérés újra
    python experiments/run_experiments.py summary            # csak az összesítő tábla

Kísérletcsoportok (--only):
    baseline    a kiinduló konfiguráció: chunk 500/100, cikkcím-előtaggal, dense keresés
    retrievers  BM25, BM25 szótövezéssel, hibrid (dense + BM25, RRF) a baseline gyűjteményen
    chunking    chunkméret és átfedés változtatása (300/50, 800/150, 1000/200, 500/0, 500/200)
    prefix      cikkcím-előtag nélküli embedding
    rerank      cross-encoder reranker a dense és a hibrid kereső tetején (nagy modellt tölt le)

A futtató már meglévő eredményt nem számol újra (--force-szal igen), és csak akkor épít
gyűjteményt, ha az még nem létezik (--reingest-tel újraépíti). Így megszakítás után
folytatható. Minden lépés a megszokott parancssori eszközöket hívja (python -m rag.ingest,
eval/run_retrieval_eval.py), a mérések naplója az experiments/runs/ mappába kerül.

Eredmények: experiments/results/<név>.json (kérdésenként is), és az összesítők:
experiments/results/summary.md és summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
RUNS_DIR = ROOT / "experiments" / "runs"
EVAL_SCRIPT = ROOT / "eval" / "run_retrieval_eval.py"
BASELINE = "corpus"
GROUP_ORDER = ("baseline", "retrievers", "chunking", "prefix", "rerank")


@dataclass(frozen=True)
class Experiment:
    group: str
    collection: str
    retriever: str = "dense"
    rerank: bool = False
    chunk_size: int = 500
    overlap: int = 100
    title_prefix: bool = True

    @property
    def result_path(self) -> Path:
        if self.retriever == "dense" and not self.rerank:
            return RESULTS_DIR / f"{self.collection}.json"
        return RESULTS_DIR / f"{self.collection}__{self.retriever}{'+rerank' if self.rerank else ''}.json"

    @property
    def label(self) -> str:
        label = f"{self.collection} / {self.retriever}"
        return label + (" + rerank" if self.rerank else "")


EXPERIMENTS: list[Experiment] = [
    Experiment("baseline", BASELINE),
    Experiment("retrievers", BASELINE, retriever="bm25"),
    Experiment("retrievers", BASELINE, retriever="bm25_stem5"),
    Experiment("retrievers", BASELINE, retriever="hybrid"),
    Experiment("retrievers", BASELINE, retriever="hybrid_stem5"),
    Experiment("chunking", "corpus_c300_o50", chunk_size=300, overlap=50),
    Experiment("chunking", "corpus_c800_o150", chunk_size=800, overlap=150),
    Experiment("chunking", "corpus_c1000_o200", chunk_size=1000, overlap=200),
    Experiment("chunking", "corpus_c500_o0", chunk_size=500, overlap=0),
    Experiment("chunking", "corpus_c500_o200", chunk_size=500, overlap=200),
    Experiment("prefix", "corpus_c500_o100_notitle", title_prefix=False),
    Experiment("rerank", BASELINE, retriever="dense", rerank=True),
    Experiment("rerank", BASELINE, retriever="hybrid_stem5", rerank=True),
]


# ---------------------------------------------------------------------- futtatás


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def fmt_time(seconds: float) -> str:
    return f"{int(seconds // 60)} p {int(seconds % 60):02d} mp"


def ingest(exp: Experiment) -> None:
    cmd = [
        sys.executable, "-m", "rag.ingest", "--recreate",
        "--collection", exp.collection,
        "--chunk-size", str(exp.chunk_size),
        "--overlap", str(exp.overlap),
    ]
    if not exp.title_prefix:
        cmd.append("--no-title-prefix")
    started = time.monotonic()
    print(f"  gyűjtemény építése: chunk {exp.chunk_size}/{exp.overlap}"
          f"{'' if exp.title_prefix else ', cím-előtag nélkül'} (néhány perc)...")
    subprocess.run(cmd, cwd=ROOT, env=child_env(), check=True)
    print(f"  gyűjtemény kész ({fmt_time(time.monotonic() - started)})")


def evaluate(exp: Experiment) -> None:
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--collection", exp.collection,
        "--retriever", exp.retriever,
        "--out", str(exp.result_path),
    ]
    if exp.rerank:
        cmd.append("--rerank")
    started = time.monotonic()
    print("  mérés...")
    proc = subprocess.run(
        cmd, cwd=ROOT, env=child_env(), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log = RUNS_DIR / f"{exp.result_path.stem}.log"
    log.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join(((proc.stderr or "") + (proc.stdout or "")).strip().splitlines()[-15:])
        sys.exit(f"  HIBA a mérésben (teljes napló: {log}):\n{tail}")
    row = load_row(exp.result_path)
    print(f"  kész ({fmt_time(time.monotonic() - started)}): chunk R@1 {row['chunk_R@1']}, "
          f"R@5 {row['chunk_R@5']}, MRR {row['chunk_MRR']}")


def qdrant_client():
    from qdrant_client import QdrantClient

    from_env = os.environ.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=from_env)
    try:
        client.get_collections()
    except Exception as exc:
        sys.exit(
            f"Nem érem el a Qdrantot ({from_env}): {type(exc).__name__}.\n"
            "Indítsd el a Docker Desktopot, majd a repó gyökeréből: docker compose up -d"
        )
    return client


def run(args: argparse.Namespace) -> None:
    selected = [e for e in EXPERIMENTS if e.group in args.only]
    selected.sort(key=lambda e: GROUP_ORDER.index(e.group))
    todo = [e for e in selected if args.force or not e.result_path.exists()]
    print(f"Kísérletek: {len(selected)} a kiválasztott csoportokban, ebből futtatandó: {len(todo)}")

    if args.dry_run:
        client = None
        for exp in selected:
            state = "kész, kihagyva" if exp not in todo else "FUT"
            print(f"  [{exp.group:<10}] {exp.label:<46} {state}")
        return

    if not todo:
        print("Nincs futtatandó kísérlet (--force-szal újramérhető).")
    else:
        client = qdrant_client()
        for number, exp in enumerate(todo, 1):
            print(f"\n[{number}/{len(todo)}] {exp.label}")
            if args.reingest or not client.collection_exists(exp.collection):
                ingest(exp)
            evaluate(exp)
    write_summary(RESULTS_DIR)


# ---------------------------------------------------------------------- összesítő


def load_row(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    overall = data["summary"]["összes (válaszolható)"]
    chunk, article = overall["chunk"], overall["article"]
    retriever = data.get("retriever", "dense")
    rerank = bool(data.get("rerank", False))
    config = f"{data['collection']} / {retriever}" + (" + rerank" if rerank else "")
    if data["collection"] == BASELINE and retriever == "dense" and not rerank:
        config += " (baseline)"

    def value(block: dict, key: str) -> str:
        return f"{block[key]:.3f}" if key in block else ""

    return {
        "konfiguráció": config,
        "chunkok": data["n_chunks"],
        "chunk_R@1": value(chunk, "R@1"),
        "chunk_R@3": value(chunk, "R@3"),
        "chunk_R@5": value(chunk, "R@5"),
        "chunk_R@10": value(chunk, "R@10"),
        "chunk_MRR": value(chunk, "MRR"),
        "cikk_R@1": value(article, "R@1"),
        "cikk_MRR": value(article, "MRR"),
    }


COLUMNS = [
    "konfiguráció", "chunkok", "chunk_R@1", "chunk_R@3", "chunk_R@5", "chunk_R@10",
    "chunk_MRR", "cikk_R@1", "cikk_MRR",
]


def write_summary(results_dir: Path) -> None:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            rows.append(load_row(path))
        except (KeyError, json.JSONDecodeError):
            continue
    if not rows:
        print("Még nincs mérési eredmény.")
        return
    rows.sort(key=lambda r: (-(float(r["chunk_MRR"]) if r["chunk_MRR"] else 0.0), r["konfiguráció"]))

    with (results_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "|".join("---" for _ in COLUMNS) + "|"]
    lines += ["| " + " | ".join(str(r[c]).replace("|", "/") for c in COLUMNS) + " |" for r in rows]
    table = "\n".join(lines)
    (results_dir / "summary.md").write_text(
        "# Kísérletek összesítése\n\nA 44 válaszolható kérdésen mért érték; chunkszinten a "
        "bizonyítékot tartalmazó chunk, cikkszinten a helyes cikk találata számít. "
        "Rendezés: chunk MRR szerint csökkenően.\n\n" + table + "\n",
        encoding="utf-8",
    )
    print(f"\nÖsszesítő ({len(rows)} konfiguráció), chunk MRR szerint rendezve:\n")
    print(table)
    print(f"\nFájlok: {results_dir / 'summary.md'}, {results_dir / 'summary.csv'}")


# ---------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-kísérletek futtatása és összesítése.")
    parser.add_argument("command", nargs="?", choices=["run", "summary"], default="run")
    parser.add_argument("--only", nargs="+", choices=GROUP_ORDER, default=list(GROUP_ORDER),
                        help="csak ezek a kísérletcsoportok (alapból az összes)")
    parser.add_argument("--force", action="store_true", help="a már meglévő mérések újrafuttatása")
    parser.add_argument("--reingest", action="store_true", help="a gyűjtemények újraépítése is")
    parser.add_argument("--dry-run", action="store_true", help="csak kiírja, mit futtatna")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.command == "summary":
        write_summary(args.results_dir)
    else:
        run(args)


if __name__ == "__main__":
    main()