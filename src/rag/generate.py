"""Válaszgenerálás helyi Ollama-modellel: RAG-os és kontextus nélküli (baseline) változat.

Az Ollama natív HTTP API-ját hívja (csak a standard könyvtárral), nem kell hozzá extra csomag.

Gyors próba a repó gyökeréből (futó Ollama és letöltött modell mellett):

    python -m rag.generate "Hány helyiségből áll a keszthelyi Festetics-kastély?" --model gemma3:4b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"
REFUSAL_TEXT = "A megadott forrásokból nem derül ki."
PLAIN_REFUSAL_TEXT = "Nem tudom."

RAG_SYSTEM_PROMPT = (
    "Magyar nyelvű kérdésmegválaszoló asszisztens vagy. A kérdésre kizárólag a megadott "
    "forrásszövegek alapján válaszolj, magyarul, tömören (egy-két mondatban). "
    "Ne használj a forrásokon kívüli tudást. "
    f"Ha a válasz nem derül ki a forrásokból, pontosan ezt a mondatot írd: „{REFUSAL_TEXT}”"
)
PLAIN_SYSTEM_PROMPT = (
    "Magyar nyelvű kérdésmegválaszoló asszisztens vagy. A kérdésre magyarul, tömören "
    "(egy-két mondatban) válaszolj. "
    f"Ha nem tudod biztosan a választ, pontosan ezt a mondatot írd: „{PLAIN_REFUSAL_TEXT}”"
)

_REFUSAL_PATTERNS = (
    "nem derül ki",
    "nem derülnek ki",
    "nem található",
    "nem szerepel",
    "nem tartalmaz",
    "nincs információ",
    "nincs adat",
    "nem áll rendelkezésre",
    "nem tudom",
    "nem tudok válaszolni",
    "nem állapítható meg",
    "nem megállapítható",
)


class OllamaError(RuntimeError):
    """Az Ollama nem érhető el, vagy hibát adott."""


@dataclass(frozen=True)
class Generation:
    text: str
    seconds: float
    output_tokens: int


# ------------------------------------------------------------------ prompt


def format_context(hits: Sequence) -> str:
    """A találatok számozott forrásblokkokká alakítva (cím › szakasz + szöveg)."""
    blocks = []
    for number, hit in enumerate(hits, 1):
        where = hit.title + (f" › {hit.section}" if getattr(hit, "section", "") else "")
        blocks.append(f"[{number}] {where}\n{hit.text.strip()}")
    return "\n\n".join(blocks)


def build_rag_messages(question: str, hits: Sequence) -> list[dict]:
    user = f"Források:\n\n{format_context(hits)}\n\nKérdés: {question}\n\nVálasz:"
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_plain_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": PLAIN_SYSTEM_PROMPT},
        {"role": "user", "content": f"Kérdés: {question}\n\nVálasz:"},
    ]


# ------------------------------------------------------------------ Ollama


def ollama_chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.0,
    seed: int = 42,
    num_ctx: int = 4096,
    num_predict: int = 256,
    timeout: float = 900.0,
) -> Generation:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            raise OllamaError(
                f"Az Ollama nem ismeri a(z) '{model}' modellt. Töltsd le: ollama pull {model}"
            ) from exc
        raise OllamaError(f"Az Ollama hibát adott (HTTP {exc.code}): {body[:300]}") from exc
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        raise OllamaError(
            f"Nem érem el az Ollamát ({base_url}): {type(exc).__name__}. "
            "Indítsd el az Ollama alkalmazást (vagy: ollama serve), és próbáld újra."
        ) from exc

    seconds = float(data.get("total_duration", 0)) / 1e9
    return Generation(
        text=(data.get("message", {}).get("content") or "").strip(),
        seconds=seconds,
        output_tokens=int(data.get("eval_count", 0)),
    )


# ------------------------------------------------------------------ automatikus jelek


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_refusal(answer: str) -> bool:
    """A válasz "nem tudom / nem derül ki" jellegű-e."""
    text = normalize_answer(answer)
    return not text or any(pattern in text for pattern in _REFUSAL_PATTERNS)


def content_stems(text: str, stem: int = 5, min_len: int = 3) -> set[str]:
    """A szöveg tartalmas szavai az első `stem` karakterre vágva (a toldalékok miatt)."""
    words = re.findall(r"\w+", text.lower())
    return {w[:stem] for w in words if len(w) >= min_len}


def answer_coverage(gold_answer: str, answer: str) -> float:
    """Az elvárt válasz tartalmas szavainak (szótő-egyezéssel mért) hányada a modell válaszában.

    Durva, automatikus közelítés: a végső értékeléshez a kézi ellenőrzés a mérvadó.
    """
    gold = content_stems(gold_answer)
    if not gold:
        return 0.0
    return len(gold & content_stems(answer)) / len(gold)


# ------------------------------------------------------------------ parancssor


def main() -> None:
    parser = argparse.ArgumentParser(description="Egyetlen kérdés megválaszolása kontextus nélkül.")
    parser.add_argument("question")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_OLLAMA_URL)
    args = parser.parse_args()
    try:
        result = ollama_chat(build_plain_messages(args.question), args.model, args.url)
    except OllamaError as exc:
        sys.exit(str(exc))
    print(f"{result.text}\n\n({result.seconds:.1f} mp, {result.output_tokens} token)")


if __name__ == "__main__":
    main()