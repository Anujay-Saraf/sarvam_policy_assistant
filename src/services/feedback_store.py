from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.config import AppConfig
from src.services.embedding_service import EmbeddingService


class FeedbackStore:
    def __init__(self, config: AppConfig, embedding_service: EmbeddingService) -> None:
        self.config = config
        self.embedding_service = embedding_service
        self.db_path = Path(config.feedback_db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    response_id TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    query_language_code TEXT,
                    actor_scope TEXT,
                    session_id TEXT,
                    retrieval_query TEXT,
                    answer_text TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    response_mode TEXT NOT NULL,
                    similarity_score REAL,
                    reused_from_response_id TEXT,
                    parent_response_id TEXT,
                    feedback_status TEXT NOT NULL DEFAULT 'pending',
                    expectation_text TEXT,
                    improvement_context_json TEXT,
                    query_embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    response_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(response_id) REFERENCES responses(response_id)
                );

                CREATE INDEX IF NOT EXISTS idx_responses_feedback_status
                    ON responses(feedback_status, created_at DESC);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(responses)").fetchall()
            }
            if "actor_scope" not in columns:
                connection.execute("ALTER TABLE responses ADD COLUMN actor_scope TEXT")
            if "session_id" not in columns:
                connection.execute("ALTER TABLE responses ADD COLUMN session_id TEXT")
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_responses_actor_scope
                    ON responses(actor_scope, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_responses_parent
                    ON responses(parent_response_id);

                CREATE INDEX IF NOT EXISTS idx_feedback_events_response
                    ON feedback_events(response_id, created_at DESC);
                """
            )

    def save_response(
        self,
        *,
        response_id: str,
        query_text: str,
        query_language_code: str,
        actor_scope: str,
        session_id: str,
        retrieval_query: str,
        answer_text: str,
        sources: list[dict[str, Any]],
        response_mode: str,
        similarity_score: float | None = None,
        reused_from_response_id: str | None = None,
        parent_response_id: str | None = None,
        improvement_context: dict[str, Any] | None = None,
    ) -> None:
        timestamp = self._utc_now()
        embedding = self.embedding_service.embed_query([query_text])[0]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO responses (
                    response_id,
                    query_text,
                    normalized_query,
                    query_language_code,
                    actor_scope,
                    session_id,
                    retrieval_query,
                    answer_text,
                    sources_json,
                    response_mode,
                    similarity_score,
                    reused_from_response_id,
                    parent_response_id,
                    feedback_status,
                    expectation_text,
                    improvement_context_json,
                    query_embedding_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response_id,
                    query_text,
                    self._normalize_query(query_text),
                    query_language_code,
                    actor_scope,
                    session_id,
                    retrieval_query,
                    answer_text,
                    json.dumps(sources, ensure_ascii=False),
                    response_mode,
                    similarity_score,
                    reused_from_response_id,
                    parent_response_id,
                    "pending",
                    None,
                    json.dumps(improvement_context or {}, ensure_ascii=False),
                    json.dumps(embedding),
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO feedback_events (
                    response_id,
                    event_type,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    response_id,
                    "response_created",
                    json.dumps(
                        {
                            "response_mode": response_mode,
                            "similarity_score": similarity_score,
                            "reused_from_response_id": reused_from_response_id,
                            "parent_response_id": parent_response_id,
                        },
                        ensure_ascii=False,
                    ),
                    timestamp,
                ),
            )

    def set_feedback(
        self,
        response_id: str,
        *,
        feedback_status: str,
        expectation_text: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        timestamp = self._utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE responses
                SET feedback_status = ?,
                    expectation_text = COALESCE(?, expectation_text),
                    updated_at = ?
                WHERE response_id = ?
                """,
                (feedback_status, expectation_text, timestamp, response_id),
            )
            connection.execute(
                """
                INSERT INTO feedback_events (
                    response_id,
                    event_type,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    response_id,
                    f"feedback_{feedback_status}",
                    json.dumps(
                        {
                            "expectation_text": expectation_text,
                            **(extra_payload or {}),
                        },
                        ensure_ascii=False,
                    ),
                    timestamp,
                ),
            )

    def get_response(self, response_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM responses
                WHERE response_id = ?
                """,
                (response_id,),
            ).fetchone()
        return self._row_to_response(row) if row else None

    def find_best_reusable_response(
        self,
        query_text: str,
        *,
        actor_scope: str | None = None,
        min_similarity: float = 0.9,
        blocked_similarity: float = 0.82,
        limit: int = 25,
    ) -> dict[str, Any] | None:
        scope_clause = " AND actor_scope = ?" if actor_scope else ""
        candidates = self._load_candidate_rows(
            f"""
            SELECT *
            FROM responses
            WHERE feedback_status = 'liked'
              AND response_mode != 'fallback'
              {scope_clause}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            ((actor_scope, limit) if actor_scope else (limit,)),
        )
        disliked_candidates = self._load_candidate_rows(
            f"""
            SELECT *
            FROM responses
            WHERE feedback_status = 'disliked'
              AND response_mode != 'fallback'
              {scope_clause}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            ((actor_scope, limit) if actor_scope else (limit,)),
        )
        return self._best_similarity_match(
            query_text,
            candidates,
            min_similarity=min_similarity,
            blocked_candidates=disliked_candidates,
            blocked_similarity=blocked_similarity,
        )

    def find_learning_signals(
        self,
        query_text: str,
        *,
        actor_scope: str | None = None,
        min_similarity: float = 0.62,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        scope_clause = " AND actor_scope = ?" if actor_scope else ""
        candidates = self._load_candidate_rows(
            f"""
            SELECT *
            FROM responses
            WHERE feedback_status IN ('liked', 'disliked')
              AND response_mode != 'fallback'
              {scope_clause}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            ((actor_scope, limit) if actor_scope else (limit,)),
        )

        query_embedding = self.embedding_service.embed_query([query_text])[0]
        signals: list[dict[str, Any]] = []
        for candidate in candidates:
            similarity = self._cosine_similarity(query_embedding, candidate.get("query_embedding") or [])
            if similarity < min_similarity:
                continue
            if candidate.get("feedback_status") == "liked":
                signals.append(
                    {
                        "type": "liked",
                        "query_text": candidate.get("query_text", ""),
                        "answer_text": candidate.get("answer_text", ""),
                        "similarity": similarity,
                    }
                )
            expectation_text = (candidate.get("expectation_text") or "").strip()
            if candidate.get("feedback_status") == "disliked" and expectation_text:
                signals.append(
                    {
                        "type": "expected",
                        "query_text": candidate.get("query_text", ""),
                        "expectation_text": expectation_text,
                        "similarity": similarity,
                    }
                )

        ranked = sorted(signals, key=lambda item: item.get("similarity", 0.0), reverse=True)
        return ranked[:4]

    def _load_candidate_rows(self, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_response(row) for row in rows]

    def _best_similarity_match(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        *,
        min_similarity: float,
        blocked_candidates: list[dict[str, Any]] | None = None,
        blocked_similarity: float = 0.82,
    ) -> dict[str, Any] | None:
        query_embedding = self.embedding_service.embed_query([query_text])[0]
        blocked_answers: set[str] = set()
        for candidate in blocked_candidates or []:
            similarity = self._cosine_similarity(query_embedding, candidate.get("query_embedding") or [])
            if similarity < blocked_similarity:
                continue
            blocked_answers.add(self._normalize_answer(candidate.get("answer_text", "")))

        best_match: dict[str, Any] | None = None
        best_score = min_similarity
        for candidate in candidates:
            if self._normalize_answer(candidate.get("answer_text", "")) in blocked_answers:
                continue
            similarity = self._cosine_similarity(query_embedding, candidate.get("query_embedding") or [])
            if similarity < best_score:
                continue
            best_score = similarity
            best_match = {
                **candidate,
                "similarity_score": similarity,
            }
        return best_match

    @staticmethod
    def _normalize_query(query_text: str) -> str:
        return " ".join(query_text.lower().split())

    @staticmethod
    def _normalize_answer(answer_text: str) -> str:
        return " ".join((answer_text or "").lower().split())

    @staticmethod
    def _row_to_response(row: sqlite3.Row) -> dict[str, Any]:
        response = dict(row)
        response["sources"] = json.loads(response.pop("sources_json") or "[]")
        response["query_embedding"] = json.loads(response.pop("query_embedding_json") or "[]")
        response["improvement_context"] = json.loads(response.pop("improvement_context_json") or "{}")
        return response

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat()
