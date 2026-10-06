#!/usr/bin/env python3
"""A korpusz letöltése a magyar Wikipédiáról kategóriák alapján.

Csak a Python standard könyvtárát használja (nincs külön függőség).

Három lépés:

  1. Kategórianevek keresése:
       python scripts/download_corpus.py find-categories "kastély"

  2. Próbafuttatás: csak összegyűjti a cikkeket, nem tölt le szöveget:
       python scripts/download_corpus.py download --dry-run \\
           --categories "Magyarország kastélyai" "Magyarország várai" --depth 2

  3. Éles letöltés:
       python scripts/download_corpus.py download \\
           --categories "Magyarország kastélyai" "Magyarország várai" --depth 2

Kimenet:
  data/raw/corpus.jsonl       egy sor = egy cikk
  data/corpus_manifest.json   a letöltés paraméterei, dátuma, cikklista revid-ekkel (git-be kerülhet)

A Wikipédia tartalma folyamatosan változik, ezért a manifest rögzíti a letöltés
időpontját és a cikkek revízióazonosítóit.

Licenc: a Wikipédia szövege CC BY-SA 4.0 alatt áll, a forrásmegjelölés a
rekordokban (url, title) és a manifestben megtalálható.

Környezeti változók:
  WIKI_USER_AGENT   A Wikimedia az általános User-Agentet szigorúbban korlátozza (429-es hiba).
                    PowerShell: $env:WIKI_USER_AGENT = "llm-rag-project/0.1 (nev@example.com)"
  WIKI_DELAY        minimális várakozás két kérés között másodpercben (alapértelmezés: 0.6)
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://hu.wikipedia.org/w/api.php"
USER_AGENT = os.environ.get("WIKI_USER_AGENT", "llm-rag-project/0.1 (university coursework)")
DELAY = float(os.environ.get("WIKI_DELAY", "0.6"))
CATEGORY_PREFIX = "Kategória:"
LICENSE = "CC BY-SA 4.0"
SOURCE = "hu.wikipedia.org"

STOP_SECTIONS = {
    "jegyzetek",
    "források",
    "forrás",
    "irodalom",
    "külső hivatkozások",
    "lásd még",
    "hivatkozások",
    "megjegyzések",
    "képek",
    "képgaléria",
    "galéria",
}

HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# API


_last_request_at = 0.0


def _throttle() -> None:
    """Minden kérés előtt gondoskodik róla, hogy legalább DELAY idő teljen el az előzőtől."""
    global _last_request_at
    wait = _last_request_at + DELAY - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def api_get(params: dict, retries: int = 8) -> dict:
    query = {**params, "format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        _throttle()
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise  # pl. 400/403/404: újrapróbálással nem javul
            retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
            wait = int(retry_after) if retry_after.isdigit() else min(5 * 2**attempt, 120)
            log(f"  API hiba (HTTP {exc.code}); újrapróbálom {wait} mp múlva... ({attempt + 1}/{retries})")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            wait = min(5 * 2**attempt, 120)
            log(f"  API hiba ({exc}); újrapróbálom {wait} mp múlva... ({attempt + 1}/{retries})")
            time.sleep(wait)
    raise RuntimeError("Az API többszöri próbálkozás után sem válaszolt.")


def api_get_all(params: dict):
    """Végigmegy az API `continue` lapozásán, és minden választ visszaad."""
    cont: dict = {}
    while True:
        data = api_get({**params, **cont})
        yield data
        if "continue" not in data:
            return
        cont = data["continue"]


# kategóriák keresése


def find_categories(query: str, limit: int) -> None:
    data = api_get(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": 14,
            "srlimit": min(limit, 50),
        }
    )
    titles = [hit["title"] for hit in data.get("query", {}).get("search", [])]
    if not titles:
        print("Nincs találat.")
        return
    info = api_get({"action": "query", "prop": "categoryinfo", "titles": "|".join(titles)})
    rows = []
    for page in info.get("query", {}).get("pages", []):
        ci = page.get("categoryinfo", {})
        rows.append((page["title"], ci.get("pages", 0), ci.get("subcats", 0)))
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"{'Cikkek':>7} {'Alkat.':>7}  Kategória")
    for title, pages, subcats in rows:
        print(f"{pages:>7} {subcats:>7}  {title.removeprefix(CATEGORY_PREFIX)}")


# cikkek gyűjtése


def normalize_category(name: str) -> str:
    name = name.strip()
    return name if name.startswith(CATEGORY_PREFIX) else CATEGORY_PREFIX + name


def collect_articles(
    categories: list[str], depth: int, exclude: list[str]
) -> tuple[dict[int, str], list[tuple[str, int, int]]]:
    """Szélességi bejárás a kategóriafán. Visszaad: {pageid: cím}, [(kategória, mélység, cikkszám)]."""
    exclude_lower = [e.lower() for e in exclude]
    articles: dict[int, str] = {}
    stats: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    queue: collections.deque[tuple[str, int]] = collections.deque()
    for cat in categories:
        name = normalize_category(cat)
        seen.add(name)
        queue.append((name, 0))

    while queue:
        cat, level = queue.popleft()
        direct = 0
        for data in api_get_all(
            {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": cat,
                "cmtype": "page|subcat",
                "cmnamespace": "0|14",
                "cmlimit": "500",
            }
        ):
            for member in data.get("query", {}).get("categorymembers", []):
                if member["ns"] == 14:
                    sub = member["title"]
                    if level < depth and sub not in seen:
                        if any(e in sub.lower() for e in exclude_lower):
                            log(f"  kihagyva (exclude): {sub}")
                            continue
                        seen.add(sub)
                        queue.append((sub, level + 1))
                else:
                    if member["pageid"] not in articles:
                        direct += 1
                    articles.setdefault(member["pageid"], member["title"])
        stats.append((cat, level, direct))
        log(f"  [{level}] {cat.removeprefix(CATEGORY_PREFIX)}: +{direct} új cikk (összesen {len(articles)})")
    return articles, stats


# szöveg letöltése


def clean_text(text: str) -> str:
    """Levágja a hivatkozás-jellegű szakaszokat, a címsorokat markdown formára hozza."""
    out: list[str] = []
    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            title = match.group(2).strip()
            if title.lower() in STOP_SECTIONS:
                break
            out.append("#" * len(match.group(1)) + " " + title)
        else:
            out.append(line.rstrip())
    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def fetch_batch(pageids: list[int]) -> dict[int, dict]:
    pages: dict[int, dict] = {}
    for data in api_get_all(
        {
            "action": "query",
            "pageids": "|".join(str(p) for p in pageids),
            "prop": "extracts|info|revisions",
            "explaintext": 1,
            "exsectionformat": "wiki",
            "exlimit": "max",
            "inprop": "url",
            "rvprop": "ids|timestamp",
        }
    ):
        for page in data.get("query", {}).get("pages", []):
            pid = page.get("pageid")
            if pid is None:
                continue
            current = pages.setdefault(pid, {})
            for key, value in page.items():
                if value is not None:
                    current[key] = value
    return pages


# download


def download(args: argparse.Namespace) -> None:
    log("Cikkek gyűjtése a kategóriákból...")
    articles, stats = collect_articles(args.categories, args.depth, args.exclude)
    log(f"\nÖsszesen {len(articles)} egyedi cikk, {len(stats)} kategóriából.")

    if args.dry_run:
        print("\nPróbafuttatás: nem történt szöveglétöltés.")
        print(f"{'Mélység':>8} {'Új cikk':>8}  Kategória")
        for cat, level, direct in stats[:40]:
            print(f"{level:>8} {direct:>8}  {cat.removeprefix(CATEGORY_PREFIX)}")
        if len(stats) > 40:
            print(f"... és még {len(stats) - 40} kategória")
        print(f"\nÖsszesen: {len(articles)} cikk. Ha ez nem a várt tartomány, módosítsd a")
        print("--depth értékét, vagy zárj ki kategóriákat az --exclude kapcsolóval.")
        return

    pageids = sorted(articles)
    if args.max_articles and len(pageids) > args.max_articles:
        log(f"--max-articles: {args.max_articles} cikkre vágva (pageid szerinti sorrendben).")
        pageids = pageids[: args.max_articles]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    skipped_short = 0
    batch_size = 20
    for start in range(0, len(pageids), batch_size):
        batch = pageids[start : start + batch_size]
        pages = fetch_batch(batch)
        for pid in batch:
            page = pages.get(pid)
            if not page or not page.get("extract"):
                continue
            text = clean_text(page["extract"])
            if len(text) < args.min_chars:
                skipped_short += 1
                continue
            revision = (page.get("revisions") or [{}])[0]
            records.append(
                {
                    "id": f"hu-{pid}",
                    "pageid": pid,
                    "title": page["title"],
                    "url": page.get("fullurl", ""),
                    "revid": revision.get("revid"),
                    "revision_timestamp": revision.get("timestamp"),
                    "n_chars": len(text),
                    "source": SOURCE,
                    "license": LICENSE,
                    "text": text,
                }
            )
        done = min(start + batch_size, len(pageids))
        log(f"  letöltve: {done}/{len(pageids)} (megtartva: {len(records)})")
        time.sleep(DELAY)

    records.sort(key=lambda r: r["title"])
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": SOURCE,
        "license": LICENSE,
        "attribution": "Szöveg: Wikipédia, a szabad enciklopédia (hu.wikipedia.org), "
        "a cikkek szerzői; CC BY-SA 4.0. Az egyes cikkek URL-je a corpus.jsonl rekordjaiban.",
        "categories": args.categories,
        "depth": args.depth,
        "exclude": args.exclude,
        "min_chars": args.min_chars,
        "max_articles": args.max_articles,
        "n_articles": len(records),
        "n_chars_total": sum(r["n_chars"] for r in records),
        "skipped_too_short": skipped_short,
        "articles": [
            {"title": r["title"], "pageid": r["pageid"], "revid": r["revid"]} for r in records
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log(f"\nKész: {len(records)} cikk -> {out_path}")
    log(f"Manifest: {manifest_path}")
    log(f"Kihagyva (túl rövid, < {args.min_chars} karakter): {skipped_short}")


# main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Magyar Wikipédia korpusz letöltése kategóriák alapján.")
    sub = parser.add_subparsers(dest="command", required=True)

    find = sub.add_parser("find-categories", help="kategóriák keresése név alapján")
    find.add_argument("query", help='keresőkifejezés, pl. "kastély"')
    find.add_argument("--limit", type=int, default=30)

    dl = sub.add_parser("download", help="cikkek letöltése")
    dl.add_argument("--categories", nargs="+", required=True, help="kiinduló kategóriák (a 'Kategória:' előtag nélkül)")
    dl.add_argument("--depth", type=int, default=2, help="alkategóriák bejárási mélysége (alapértelmezés: 2)")
    dl.add_argument("--exclude", nargs="*", default=[], help="alkategória-névrészletek, amelyeket kihagy a bejárás")
    dl.add_argument("--max-articles", type=int, default=0, help="felső korlát a cikkek számára (0 = nincs)")
    dl.add_argument("--min-chars", type=int, default=300, help="ennél rövidebb cikkeket eldobja (alapértelmezés: 300)")
    dl.add_argument("--out", default="data/raw/corpus.jsonl")
    dl.add_argument("--manifest", default="data/corpus_manifest.json")
    dl.add_argument("--dry-run", action="store_true", help="csak összegyűjti és összesíti a cikkeket")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "find-categories":
        find_categories(args.query, args.limit)
    else:
        download(args)


if __name__ == "__main__":
    main()