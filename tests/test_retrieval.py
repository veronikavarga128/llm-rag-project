import pytest

from rag.retrieval import (
    BM25Retriever,
    HybridRetriever,
    Hit,
    IndexedChunk,
    RerankedRetriever,
    build_retriever,
    rrf_fuse,
    tokenize,
)


def make_chunks() -> list[IndexedChunk]:
    return [
        IndexedChunk("d1", "Egri vár", "Története", 0, "Az egri vár 1552-ben ostromot állt ki."),
        IndexedChunk("d2", "Diósgyőri vár", "", 0, "A diósgyőri vár Nagy Lajos idején élte fénykorát."),
        IndexedChunk("d3", "Festetics-kastély", "Története", 0, "A keszthelyi kastély 101 helyiségből áll."),
    ]


def hit(doc_id: str, score: float = 0.0, index: int = 0) -> Hit:
    return Hit(score, doc_id, f"cím {doc_id}", "", index, f"szöveg {doc_id}")


def test_tokenize_lowercases_and_keeps_numbers():
    assert tokenize("Az Egri vár 1552-ben épült.") == ["az", "egri", "vár", "1552", "ben", "épült"]


def test_tokenize_stem_truncates():
    assert tokenize("kastélyt kastély vár", stem=5) == ["kasté", "kasté", "vár"]


def test_bm25_finds_matching_chunk():
    retriever = BM25Retriever(make_chunks())
    hits = retriever.search("Hány helyiségből áll a kastély?", 3)
    assert hits[0].doc_id == "d3"
    assert len(hits) == 3


def test_bm25_uses_title_text():
    retriever = BM25Retriever(make_chunks())
    assert retriever.search("Diósgyőri", 1)[0].doc_id == "d2"


def test_stemming_matches_inflected_forms():
    chunks = make_chunks()
    plain = BM25Retriever(chunks, stem=0).search("kastélyt", 3)
    stemmed = BM25Retriever(chunks, stem=5).search("kastélyt", 3)
    assert stemmed[0].doc_id == "d3"
    assert stemmed[0].score > 0
    assert all(h.score == 0 for h in plain)


def test_bm25_ties_are_deterministic():
    retriever = BM25Retriever(make_chunks())
    first = [h.doc_id for h in retriever.search("nincsilyenszó", 3)]
    second = [h.doc_id for h in retriever.search("nincsilyenszó", 3)]
    assert first == second == ["d1", "d2", "d3"]


def test_rrf_prefers_items_in_both_rankings():
    dense = [hit("a"), hit("b"), hit("c")]
    sparse = [hit("c"), hit("b"), hit("d")]
    fused = rrf_fuse([dense, sparse], k=4)
    # c: 1/63 + 1/61 = 0,032266; b: 1/62 + 1/62 = 0,032258; a: 1/61; d: 1/63
    assert [h.doc_id for h in fused] == ["c", "b", "a", "d"]


def test_rrf_truncates_to_k_and_scores_by_rank():
    fused = rrf_fuse([[hit("a"), hit("b")]], k=1, rrf_k=60)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / 61)


def test_hybrid_combines_dense_and_sparse():
    class FakeDense:
        def search(self, query, k):
            return [hit("x"), hit("y")][:k]

    sparse = BM25Retriever(make_chunks())
    fused = HybridRetriever(FakeDense(), sparse, pool=10).search("kastély helyiség", 3)
    ids = [h.doc_id for h in fused]
    assert "d3" in ids and "x" in ids
    assert len(ids) == 3


def test_reranked_retriever_reorders_and_truncates():
    class FakeBase:
        def search(self, query, k):
            assert k >= 3  # a top_n-et kéri le
            return [hit("a"), hit("b"), hit("c")]

    class FakeReranker:
        def rerank(self, query, hits, k):
            return list(reversed(hits))[:k]

    result = RerankedRetriever(FakeBase(), FakeReranker(), top_n=3).search("kérdés", 2)
    assert [h.doc_id for h in result] == ["c", "b"]


def test_build_retriever_names():
    chunks = make_chunks()
    sentinel = object()
    assert build_retriever("dense", chunks, lambda: sentinel) is sentinel
    assert isinstance(build_retriever("bm25", chunks, lambda: sentinel), BM25Retriever)
    assert build_retriever("bm25_stem5", chunks, lambda: sentinel).stem == 5
    hybrid = build_retriever("hybrid_stem5", chunks, lambda: sentinel)
    assert isinstance(hybrid, HybridRetriever) and hybrid.sparse.stem == 5
    with pytest.raises(ValueError):
        build_retriever("ismeretlen", chunks, lambda: sentinel)