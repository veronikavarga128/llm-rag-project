"""Embedding-modell burkoló (sentence-transformers).

Az E5-családú modellek (pl. intfloat/multilingual-e5-small) elvárják a "query: " és
"passage: " előtagot; más modelleknél (pl. BAAI/bge-m3) nincs szükség rá.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

MAX_SEQ_LENGTH = 512


class Embedder:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        # Egyes modellek (pl. bge-m3) alapértelmezett hossza 8192, ami CPU-n felesleges és lassú.
        current = self.model.max_seq_length or MAX_SEQ_LENGTH
        self.model.max_seq_length = min(current, MAX_SEQ_LENGTH)
        uses_e5_prefixes = "e5" in model_name.lower()
        self.query_prefix = "query: " if uses_e5_prefixes else ""
        self.passage_prefix = "passage: " if uses_e5_prefixes else ""

    @property
    def dim(self) -> int:
        # Az újabb sentence-transformers átnevezte a metódust; a régebbi névvel is működik.
        getter = getattr(self.model, "get_embedding_dimension", None)
        if getter is None:
            getter = self.model.get_sentence_embedding_dimension
        return int(getter())

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        prefixed = [self.passage_prefix + t for t in texts]
        vectors = self.model.encode(
            prefixed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(
            self.query_prefix + query, normalize_embeddings=True, show_progress_bar=False
        )
        return vector.tolist()