from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from typing import Any

import chromadb

from src.core.config import AppConfig
from src.services.embedding_service import EmbeddingService


class DocumentStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embedding_service = EmbeddingService(config)
        self.client = chromadb.PersistentClient(path=str(config.vector_store_dir))
        collection_name = self._build_collection_name(config.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Policy knowledge base",
                "embedding_backend": self.embedding_service.backend,
                "embedding_model": self.embedding_service.model_name,
            },
        )

    def _build_collection_name(self, base_name: str) -> str:
        suffix = self.embedding_service.collection_suffix[:36]
        combined = f"{base_name}_{suffix}"
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", combined)[:63]

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_service.embed_documents(texts)

    def _embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_service.embed_query(texts)

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = json.dumps(value, ensure_ascii=False)
        return cleaned

    @staticmethod
    def _decode_metadata_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return value
        if stripped[0] not in "[{\"" and stripped not in {"true", "false", "null"}:
            return value
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value

    def _decode_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            decoded[key] = self._decode_metadata_value(value)
        return decoded

    def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            return

        ids = [item["chunk_id"] for item in documents]
        search_texts = [item["search_text"] for item in documents]
        metadatas = [
            self._clean_metadata({key: value for key, value in item.items() if key not in {"chunk_id", "search_text"}})
            for item in documents
        ]
        embeddings = self._embed_documents(search_texts)
        self.collection.upsert(
            ids=ids,
            documents=search_texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def _run_query(
        self,
        query: str,
        top_k: int,
        allowed_document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        if allowed_document_ids:
            return self._run_filtered_query(query, top_k, allowed_document_ids)

        response = self.collection.query(
            query_embeddings=[self._embed_queries([query])[0]],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        ids = response.get("ids", [[]])[0]

        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            hit = dict(metadata or {})
            if allowed_document_ids and hit.get("document_id") not in allowed_document_ids:
                continue
            hit["chunk_id"] = chunk_id
            hit["search_text"] = document
            semantic_score = float(1 / (1 + distance)) if distance is not None else 0.0
            lexical_score = self._lexical_score(query, document)
            hit["semantic_score"] = semantic_score
            hit["lexical_score"] = lexical_score
            hit["score"] = (0.8 * semantic_score) + (0.2 * lexical_score)
            hits.append(hit)
        return hits

    def _run_filtered_query(
        self,
        query: str,
        top_k: int,
        allowed_document_ids: list[str],
    ) -> list[dict[str, Any]]:
        selected_ids = {document_id for document_id in allowed_document_ids if document_id}
        if not selected_ids:
            return []

        response = self.collection.get(include=["documents", "metadatas", "embeddings"])
        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])
        embeddings = response.get("embeddings", [])
        ids = response.get("ids", [])
        query_embedding = self._embed_queries([query])[0]

        hits: list[dict[str, Any]] = []
        for chunk_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            hit = dict(metadata or {})
            if hit.get("document_id") not in selected_ids:
                continue
            hit["chunk_id"] = chunk_id
            hit["search_text"] = document
            semantic_score = self._cosine_similarity(query_embedding, embedding)
            lexical_score = self._lexical_score(query, document)
            hit["semantic_score"] = semantic_score
            hit["lexical_score"] = lexical_score
            hit["score"] = (0.8 * semantic_score) + (0.2 * lexical_score)
            hits.append(hit)

        ranked = sorted(hits, key=lambda item: item.get("score", 0.0), reverse=True)
        return ranked[:top_k]

    def hybrid_search(
        self,
        original_query: str,
        translated_query: str,
        top_k: int,
        allowed_document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        ordered_hits: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        query_limit = max(top_k * 6, 24) if allowed_document_ids else max(top_k * 2, 8)
        for hit in self._run_query(original_query, query_limit, allowed_document_ids=allowed_document_ids):
            ordered_hits.setdefault(hit["chunk_id"], hit)
        if translated_query and translated_query != original_query:
            for hit in self._run_query(translated_query, query_limit, allowed_document_ids=allowed_document_ids):
                existing = ordered_hits.get(hit["chunk_id"])
                if existing is None or hit["score"] > existing["score"]:
                    ordered_hits[hit["chunk_id"]] = hit
        ranked = sorted(ordered_hits.values(), key=lambda item: item.get("score", 0.0), reverse=True)
        return ranked[:top_k]

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[dict[str, Any]]:
        response = self.collection.get(include=["metadatas"])
        sources: dict[str, dict[str, Any]] = {}
        for metadata in response.get("metadatas", []):
            if not metadata:
                continue
            document_id = metadata.get("document_id")
            if not document_id:
                continue
            record = sources.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "source_name": metadata.get("source_name", "Unknown source"),
                    "extraction_method": metadata.get("extraction_method", "unknown"),
                    "language_code": metadata.get("language_code", "n/a"),
                    "file_size_bytes": int(metadata.get("file_size_bytes") or 0),
                    "chunk_count": 0,
                },
            )
            record["chunk_count"] += 1
        return list(sources.values())

    def get_source_records(self, document_id: str) -> list[dict[str, Any]]:
        response = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        ids = response.get("ids", [])
        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])
        records: list[dict[str, Any]] = []
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            item = self._decode_metadata(metadata)
            item["chunk_id"] = chunk_id
            item["search_text"] = document
            records.append(item)
        records.sort(key=lambda item: int(item.get("chunk_index") or 0))
        return records

    def delete_source(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.split(r"\W+", text.lower()) if len(token) > 1}

    def _lexical_score(self, query: str, document: str) -> float:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0
        document_tokens = self._tokenize(document)
        overlap = len(query_tokens & document_tokens)
        return overlap / len(query_tokens)

    @staticmethod
    def _cosine_similarity(query_embedding: list[float], document_embedding: list[float] | None) -> float:
        if query_embedding is None or document_embedding is None:
            return 0.0
        if len(query_embedding) == 0 or len(document_embedding) == 0:
            return 0.0
        dot = sum(float(left) * float(right) for left, right in zip(query_embedding, document_embedding))
        query_norm = math.sqrt(sum(float(value) * float(value) for value in query_embedding))
        document_norm = math.sqrt(sum(float(value) * float(value) for value in document_embedding))
        if not query_norm or not document_norm:
            return 0.0
        cosine = dot / (query_norm * document_norm)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))
