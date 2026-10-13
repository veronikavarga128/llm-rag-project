import pytest

from rag.chunking import chunk_document


def make_article() -> str:
    intro = " ".join(f"Ez a {i}. bevezető mondat a várról." for i in range(1, 8))
    history = " ".join(f"A vár építése {1300 + i} körül folytatódott." for i in range(40))
    layout = " ".join(f"A {i}. szoba a keleti szárnyban található." for i in range(40))
    return f"{intro}\n\n## Története\n{history}\n\n## Leírása\n{layout}"


def test_short_text_is_single_chunk():
    chunks = chunk_document("Egy rövid mondat. Még egy.", size=500, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "Egy rövid mondat. Még egy."
    assert chunks[0].index == 0


def test_empty_text_gives_no_chunks():
    assert chunk_document("", size=500, overlap=100) == []
    assert chunk_document("\n\n  \n", size=500, overlap=100) == []


def test_chunks_respect_size_limit():
    for size, overlap in [(300, 50), (500, 100), (800, 150)]:
        chunks = chunk_document(make_article(), size=size, overlap=overlap)
        assert chunks
        assert all(len(c.text) <= size for c in chunks)


def test_chunk_indices_are_sequential():
    chunks = chunk_document(make_article(), size=300, overlap=50)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_repeats_text_between_neighbours():
    chunks = chunk_document(make_article(), size=400, overlap=120)
    overlapping = 0
    for previous, following in zip(chunks, chunks[1:], strict=False):
        first_sentence = following.text.split(". ")[0]
        if first_sentence and first_sentence in previous.text:
            overlapping += 1
    assert overlapping > len(chunks) // 2


def test_no_overlap_when_zero():
    text = " ".join(f"Mondat száma {i}." for i in range(200))
    chunks = chunk_document(text, size=200, overlap=0)
    joined = " ".join(c.text for c in chunks)
    assert joined.count("Mondat száma 100.") == 1


def test_section_is_tracked():
    chunks = chunk_document(make_article(), size=200, overlap=40)
    assert chunks[0].section == ""
    assert any(c.section == "Története" for c in chunks)
    assert any(c.section == "Leírása" for c in chunks)
    for chunk in chunks:
        if "keleti szárnyban" in chunk.text and "építése" not in chunk.text:
            assert chunk.section == "Leírása"


def test_heading_never_dangles_at_chunk_end():
    chunks = chunk_document(make_article(), size=300, overlap=50)
    for chunk in chunks:
        last_line = chunk.text.strip().splitlines()[-1]
        assert not last_line.startswith("## ")


def test_very_long_sentence_is_hard_split():
    text = "A" * 1200
    chunks = chunk_document(text, size=500, overlap=100)
    assert all(len(c.text) <= 500 for c in chunks)
    assert sum(len(c.text) for c in chunks) >= 1200


def test_invalid_parameters():
    with pytest.raises(ValueError):
        chunk_document("szöveg", size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_document("szöveg", size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_document("szöveg", size=100, overlap=-1)