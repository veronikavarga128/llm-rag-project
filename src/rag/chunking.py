"""Szövegdarabolás (chunkolás) a RAG-pipeline-hoz.

A méretek KARAKTERBEN értendők (nem tokenben).

Működés:
  - a cikket mondatokra bontja, és azokat gyűjti össze egy chunkba, amíg a méretkorlát engedi;
  - bekezdés- és szakaszhatárokat megőrzi (a bekezdések között üres sor marad);
  - a szakaszcímet (## ...) sosem hagyja a chunk végén "árván": átviszi a következő chunk elejére;
  - az átfedés (overlap) mondatszintű: a következő chunk az előző utolsó mondatai közül annyit
    ismétel meg, amennyi belefér az átfedés karakterkeretébe (lehet, hogy egyet sem, ha a
    mondatok hosszabbak a keretnél);
  - minden chunkhoz rögzíti, melyik szakaszba tartozik (a chunk utolsó mondatának szakasza).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

HEADING_RE = re.compile(r"^#{2,6}\s+(.+?)\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰ0-9„\"(\[])")
PARAGRAPH_SEP = "\n\n"
HEADING_SEP = "\n"
SENTENCE_SEP = " "


@dataclass(frozen=True)
class Chunk:
    index: int
    section: str
    text: str


@dataclass(frozen=True)
class _Unit:
    text: str
    section: str
    sep: str
    heading: bool = False


def _split_sentence(sentence: str, size: int) -> list[str]:
    """A méretkorlátnál hosszabb "mondatot" karakterhatáron darabolja."""
    if len(sentence) <= size:
        return [sentence]
    return [sentence[i : i + size] for i in range(0, len(sentence), size)]


def _paragraph_units(text: str, section: str, size: int, first_sep: str) -> Iterator[_Unit]:
    text = text.strip()
    if not text:
        return
    first = True
    for sentence in SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for piece in _split_sentence(sentence, size):
            yield _Unit(piece, section, first_sep if first else SENTENCE_SEP)
            first = False


def _units(text: str, size: int) -> Iterator[_Unit]:
    section = ""
    after_heading = False
    for block in re.split(r"\n\s*\n", text):
        buffer: list[str] = []
        for line in block.splitlines():
            match = HEADING_RE.match(line.strip())
            if match:
                body = "\n".join(buffer).strip()
                if body:
                    yield from _paragraph_units(
                        body, section, size, HEADING_SEP if after_heading else PARAGRAPH_SEP
                    )
                    after_heading = False
                buffer = []
                section = match.group(1)
                yield _Unit(line.strip(), section, PARAGRAPH_SEP, heading=True)
                after_heading = True
            else:
                buffer.append(line)
        body = "\n".join(buffer).strip()
        if body:
            yield from _paragraph_units(
                body, section, size, HEADING_SEP if after_heading else PARAGRAPH_SEP
            )
            after_heading = False


def _joined(units: list[_Unit]) -> str:
    if not units:
        return ""
    parts = [units[0].text]
    for unit in units[1:]:
        parts.append(unit.sep)
        parts.append(unit.text)
    return "".join(parts)


def _joined_len(units: list[_Unit]) -> int:
    if not units:
        return 0
    return len(units[0].text) + sum(len(u.sep) + len(u.text) for u in units[1:])


def chunk_document(text: str, size: int = 500, overlap: int = 100) -> list[Chunk]:
    """A cikk szövegét legfeljebb `size` karakteres chunkokra bontja `overlap` átfedéssel."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Kell: size > 0 és 0 <= overlap < size.")

    chunks: list[Chunk] = []
    current: list[_Unit] = []
    new_units = 0

    def fits(units: list[_Unit], unit: _Unit) -> bool:
        return not units or _joined_len(units) + len(unit.sep) + len(unit.text) <= size

    def emit(units: list[_Unit]) -> tuple[list[_Unit], list[_Unit]]:
        body = list(units)
        dangling: list[_Unit] = []
        while body and body[-1].heading:
            dangling.insert(0, body.pop())
        if body:
            chunks.append(Chunk(len(chunks), body[-1].section, _joined(body)))
        return body, dangling

    def overlap_tail(body: list[_Unit]) -> list[_Unit]:
        tail: list[_Unit] = []
        total = 0
        for unit in reversed(body):
            add = len(unit.text) + (len(tail[0].sep) if tail else 0)
            if total + add > overlap:
                break
            tail.insert(0, unit)
            total += add
        return tail

    for unit in _units(text, size):
        if not fits(current, unit):
            if new_units:
                body, dangling = emit(current)
                current = overlap_tail(body) + dangling
                new_units = 0
                if not fits(current, unit):
                    current = dangling
                if not fits(current, unit):
                    current = []
            else:
                current = []
        current.append(unit)
        new_units += 1

    if new_units:
        emit(current)
    return chunks
