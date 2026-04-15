from __future__ import annotations

import re
from typing import Any

from sklearn.feature_extraction.text import HashingVectorizer

from src.core.config import AppConfig


class EmbeddingService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.backend = "hashing"
        self.model_name = "hashing"
        self.dimension = config.embedding_dimensions
        self._encoder: Any | None = None
        self._vectorizer: HashingVectorizer | None = None
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        preferred = (self.config.embedding_backend or "auto").lower()
        wants_transformer = preferred in {"auto", "sentence-transformers", "sentence_transformers"}

        if wants_transformer:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(
                    self.config.embedding_model,
                    cache_folder=str(self.config.model_cache_dir),
                )
                self.backend = "sentence-transformers"
                self.model_name = self.config.embedding_model
                detected_dimension = self._encoder.get_sentence_embedding_dimension()
                if isinstance(detected_dimension, int) and detected_dimension > 0:
                    self.dimension = detected_dimension
                return
            except Exception:
                pass

        self._vectorizer = HashingVectorizer(
            n_features=self.config.embedding_dimensions,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )
        self.backend = "hashing"
        self.model_name = "hashing"
        self.dimension = self.config.embedding_dimensions

    @property
    def collection_suffix(self) -> str:
        raw = f"{self.backend}_{self.model_name}"
        return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        normalized = [text or "" for text in texts]
        if self._encoder is not None:
            embeddings = self._encoder.encode(
                normalized,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.tolist()

        if self._vectorizer is None:
            raise RuntimeError("Embedding backend is not initialized.")
        matrix = self._vectorizer.transform(normalized)
        return matrix.toarray().tolist()

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        normalized = [self._prepare_query_text(text or "") for text in texts]
        if self._encoder is not None:
            embeddings = self._encoder.encode(
                normalized,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.tolist()

        if self._vectorizer is None:
            raise RuntimeError("Embedding backend is not initialized.")
        matrix = self._vectorizer.transform(normalized)
        return matrix.toarray().tolist()

    def _prepare_query_text(self, text: str) -> str:
        model_name = self.model_name.lower()
        if "bge" in model_name:
            return f"Represent this sentence for searching relevant passages: {text}"
        return text
