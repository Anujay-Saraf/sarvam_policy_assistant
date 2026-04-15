from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from src.core.config import AppConfig


class StructuredDataStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.db_path = Path(config.structured_db_path)
        self._ensure_schema()
        self.seed_demo_data()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    ingestion_mode TEXT NOT NULL,
                    notes TEXT,
                    imported_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tables_meta (
                    table_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    page_hint TEXT,
                    schema_json TEXT NOT NULL,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    extraction_mode TEXT NOT NULL,
                    raw_preview TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE TABLE IF NOT EXISTS table_rows (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_id INTEGER NOT NULL,
                    row_index INTEGER NOT NULL,
                    row_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(table_id) REFERENCES tables_meta(table_id)
                );

                CREATE TABLE IF NOT EXISTS query_logs (
                    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_id INTEGER,
                    user_query TEXT NOT NULL,
                    proposed_sql TEXT,
                    answer_text TEXT NOT NULL,
                    matched_rows INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(table_id) REFERENCES tables_meta(table_id)
                );

                CREATE INDEX IF NOT EXISTS idx_tables_document ON tables_meta(document_id);
                CREATE INDEX IF NOT EXISTS idx_rows_table ON table_rows(table_id, row_index);
                """
            )

    def seed_demo_data(self) -> None:
        with self._connect() as connection:
            existing = connection.execute("SELECT COUNT(*) AS total FROM documents").fetchone()
            if existing and int(existing["total"]) > 0:
                return

        rows = [
            {
                "serial_no": "1",
                "name": "Ravi Kumar",
                "village": "Bhopal",
                "scheme": "Kisan Support",
                "amount": 12500,
                "status": "Approved",
                "year": 2025,
            },
            {
                "serial_no": "2",
                "name": "Sita Devi",
                "village": "Indore",
                "scheme": "Kisan Support",
                "amount": 9800,
                "status": "Pending",
                "year": 2025,
            },
            {
                "serial_no": "3",
                "name": "Rahim Ali",
                "village": "Bhopal",
                "scheme": "Scholarship",
                "amount": 15000,
                "status": "Approved",
                "year": 2024,
            },
            {
                "serial_no": "4",
                "name": "Meena Bai",
                "village": "Jabalpur",
                "scheme": "Scholarship",
                "amount": 15000,
                "status": "Approved",
                "year": 2025,
            },
            {
                "serial_no": "5",
                "name": "Anita Sharma",
                "village": "Indore",
                "scheme": "Widow Pension",
                "amount": 6000,
                "status": "Rejected",
                "year": 2025,
            },
        ]
        self._insert_table(
            document_id="demo_handwritten_register",
            source_name="demo_handwritten_register.jpeg",
            source_type="demo",
            ingestion_mode="seeded-example",
            table_name="beneficiary_register",
            rows=rows,
            extraction_mode="manual-demo-seed",
            page_hint="1",
            notes="Default example showing how OCR or handwritten table data looks after normalization.",
            raw_preview="Serial | Name | Village | Scheme | Amount | Status | Year",
        )

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.document_id, d.source_name, d.source_type, d.ingestion_mode, d.notes, d.imported_at,
                       COUNT(t.table_id) AS table_count,
                       COALESCE(SUM(t.row_count), 0) AS total_rows
                FROM documents d
                LEFT JOIN tables_meta t ON t.document_id = d.document_id
                GROUP BY d.document_id, d.source_name, d.source_type, d.ingestion_mode, d.notes, d.imported_at
                ORDER BY d.imported_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_tables(self, document_id: str | None = None) -> list[dict[str, Any]]:
        params: tuple[Any, ...] = ()
        query = """
            SELECT t.table_id, t.document_id, d.source_name, t.table_name, t.page_hint, t.schema_json, t.row_count,
                   t.extraction_mode, t.raw_preview, t.created_at
            FROM tables_meta t
            JOIN documents d ON d.document_id = t.document_id
        """
        if document_id:
            query += " WHERE t.document_id = ?"
            params = (document_id,)
        query += " ORDER BY t.created_at DESC, t.table_id DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["schema"] = json.loads(item.pop("schema_json") or "[]")
            results.append(item)
        return results

    def get_table_rows(self, table_id: int, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT row_index, row_json
                FROM table_rows
                WHERE table_id = ?
                ORDER BY row_index ASC
                LIMIT ?
                """,
                (table_id, limit),
            ).fetchall()
        return [json.loads(row["row_json"]) for row in rows]

    def import_document_records(
        self,
        *,
        document_id: str,
        source_name: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        rows = self._extract_rows_from_structured_metadata(records)
        extraction_mode = "ocr-metadata-table"
        if not rows:
            chunks = [str(record.get("original_text") or record.get("search_text") or "").strip() for record in records]
            combined_text = "\n".join(chunk for chunk in chunks if chunk)
            rows = self._extract_rows_from_text(combined_text)
            extraction_mode = "ocr-heuristic-table" if rows else "ocr-heuristic-staging"
            if not rows:
                rows = self._build_fallback_rows(combined_text)
            raw_preview = combined_text[:600]
        else:
            raw_preview = self._build_metadata_preview(records)

        if not rows:
            raise ValueError("Structured import ke liye usable row-like OCR content nahi mila.")

        page_values = sorted({str(record.get("page_start")) for record in records if record.get("page_start")})
        page_hint = ",".join(page_values[:8]) if page_values else "n/a"
        self._delete_document_if_exists(document_id)
        table_id = self._insert_table(
            document_id=document_id,
            source_name=source_name,
            source_type="ocr-import",
            ingestion_mode="structured-staging",
            table_name=f"{Path(source_name).stem}_table",
            rows=rows,
            extraction_mode=extraction_mode,
            page_hint=page_hint,
            notes="Best-effort structured import generated from OCR/vector chunks and detected table metadata.",
            raw_preview=raw_preview,
        )
        return {
            "document_id": document_id,
            "table_id": table_id,
            "row_count": len(rows),
            "extraction_mode": extraction_mode,
        }

    def import_tabular_file(
        self,
        *,
        source_name: str,
        file_bytes: bytes,
    ) -> dict[str, Any]:
        suffix = Path(source_name).suffix.lower()
        document_id = f"structured_{uuid4().hex[:12]}"
        tables_to_insert: list[tuple[str, list[dict[str, Any]], str]] = []

        if suffix == ".csv":
            rows = self._parse_csv_bytes(file_bytes)
            if rows:
                tables_to_insert.append((Path(source_name).stem or "csv_table", rows, "csv-direct-import"))
        elif suffix == ".json":
            rows = self._parse_json_bytes(file_bytes)
            if rows:
                tables_to_insert.append((Path(source_name).stem or "json_table", rows, "json-direct-import"))
        elif suffix == ".xlsx":
            workbook_tables = self._parse_xlsx_bytes(file_bytes)
            for sheet_name, rows in workbook_tables:
                if rows:
                    tables_to_insert.append((sheet_name, rows, "xlsx-direct-import"))
        else:
            raise ValueError("Only CSV, JSON, and XLSX files can be directly imported into Structured Data.")

        if not tables_to_insert:
            raise ValueError("Uploaded tabular file me usable rows nahi mile.")

        self._delete_document_if_exists(document_id)
        table_ids: list[int] = []
        total_rows = 0
        for table_name, rows, extraction_mode in tables_to_insert:
            table_ids.append(
                self._insert_table(
                    document_id=document_id,
                    source_name=source_name,
                    source_type="tabular-upload",
                    ingestion_mode="structured-direct",
                    table_name=self._safe_identifier(table_name),
                    rows=rows,
                    extraction_mode=extraction_mode,
                    page_hint="n/a",
                    notes="Direct structured import from uploaded tabular file.",
                    raw_preview=json.dumps(rows[:3], ensure_ascii=False)[:600],
                )
            )
            total_rows += len(rows)

        return {
            "document_id": document_id,
            "table_id": table_ids[0],
            "table_ids": table_ids,
            "table_count": len(table_ids),
            "row_count": total_rows,
            "extraction_mode": tables_to_insert[0][2],
        }

    def answer_query(self, question: str, table_id: int | None = None) -> dict[str, Any]:
        target_table = self._resolve_table(table_id=table_id, question=question)
        if target_table is None:
            return {
                "answer": "Structured DB me koi table available nahi hai.",
                "proposed_sql": "",
                "matched_rows": [],
                "table": None,
            }

        rows = self.get_table_rows(int(target_table["table_id"]), limit=1000)
        query_plan = self._build_query_plan(
            question=question,
            rows=rows,
            table_name=str(target_table["table_name"]),
        )
        display_rows = list(query_plan["display_rows"])
        proposed_sql = str(query_plan["proposed_sql"])
        answer = self._build_answer_from_plan(query_plan)
        self._log_query(
            table_id=int(target_table["table_id"]),
            user_query=question,
            proposed_sql=proposed_sql,
            answer_text=answer,
            matched_rows=len(display_rows),
        )
        return {
            "answer": answer,
            "proposed_sql": proposed_sql,
            "matched_rows": display_rows,
            "table": target_table,
        }

    def clear(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink(missing_ok=True)
        self._ensure_schema()
        self.seed_demo_data()

    def _resolve_table(self, *, table_id: int | None, question: str) -> dict[str, Any] | None:
        tables = self.list_tables()
        if not tables:
            return None
        if table_id is not None:
            for table in tables:
                if int(table["table_id"]) == int(table_id):
                    return table

        normalized_query = self._normalize_text(question)
        best_table = None
        best_score = -1
        for table in tables:
            haystack = self._normalize_text(
                f"{table.get('table_name', '')} {table.get('source_name', '')} {' '.join(table.get('schema', []))}"
            )
            score = sum(1 for token in self._tokenize(normalized_query) if token and token in haystack)
            if score > best_score:
                best_score = score
                best_table = table
        return best_table or tables[0]

    def _filter_rows_by_tokens(self, question: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized_query = self._normalize_text(question)
        tokens = [token for token in self._tokenize(normalized_query) if len(token) > 2]
        if not tokens:
            return []

        skip_tokens = {
            "show",
            "list",
            "rows",
            "record",
            "records",
            "table",
            "data",
            "find",
            "give",
            "whose",
            "which",
            "kitne",
            "kitni",
            "kitna",
            "count",
            "total",
            "sum",
            "kul",
            "jod",
            "hai",
            "hain",
            "ka",
            "ke",
            "ki",
            "for",
            "with",
            "from",
            "all",
            "english",
            "hindi",
            "query",
        }
        filtered_tokens = [token for token in tokens if token not in skip_tokens]
        if not filtered_tokens:
            return []

        matches: list[dict[str, Any]] = []
        for row in rows:
            haystack = self._normalize_text(json.dumps(row, ensure_ascii=False))
            if all(token in haystack for token in filtered_tokens):
                matches.append(row)
        return matches

    def _pick_numeric_column(self, question: str, rows: list[dict[str, Any]]) -> str | None:
        if not rows:
            return None
        normalized_query = self._normalize_text(question)
        columns = list(rows[0].keys())
        numeric_columns = [
            column
            for column in columns
            if sum(1 for row in rows if self._is_number_like(row.get(column))) >= max(1, len(rows) // 2)
        ]
        if not numeric_columns:
            return None
        amount_query_hints = {"amount", "pese", "paise", "paisa", "money", "rupees", "rupaye", "rs", "inr", "fund"}
        amount_column_hints = {"amount", "amt", "value", "payment", "fund", "money", "sanction"}
        if any(token in normalized_query for token in amount_query_hints):
            for column in numeric_columns:
                normalized_column = self._normalize_text(column)
                if any(hint in normalized_column for hint in amount_column_hints):
                    return column
        for column in numeric_columns:
            if self._normalize_text(column) in normalized_query:
                return column
        preferred = [
            column
            for column in numeric_columns
            if self._normalize_text(column) in {"amount", "year", "value", "total"}
        ]
        return preferred[0] if preferred else numeric_columns[0]

    def _build_query_plan(self, *, question: str, rows: list[dict[str, Any]], table_name: str) -> dict[str, Any]:
        operation = self._detect_operation(question, rows)
        value_filters = self._extract_value_filters(question, rows)
        group_column = self._detect_group_column(question, rows, value_filters)
        comparison_column = self._detect_comparison_column(question, value_filters)
        if comparison_column and not group_column:
            group_column = comparison_column

        applied_filters = list(value_filters)
        if group_column:
            applied_filters = [item for item in applied_filters if item["column"] != group_column]

        matched_rows = self._apply_value_filters(rows, applied_filters)
        used_fallback = False
        if not matched_rows:
            token_matches = self._filter_rows_by_tokens(question, rows)
            if token_matches:
                matched_rows = token_matches
                used_fallback = True
        if not matched_rows:
            matched_rows = list(rows)

        if group_column and operation == "rows":
            operation = "count"

        numeric_column = self._pick_numeric_column(question, matched_rows or rows) if operation in {"sum", "avg", "min", "max"} else None
        limit = self._detect_limit(question)
        row_sort = self._detect_row_sort(question, rows, numeric_column)

        aggregated_rows: list[dict[str, Any]] = []
        display_rows: list[dict[str, Any]]
        order_clause = ""
        group_clause = ""
        if group_column:
            aggregated_rows = self._aggregate_by_column(
                matched_rows,
                group_column=group_column,
                operation=operation,
                numeric_column=numeric_column,
            )
            aggregated_rows, order_clause = self._order_grouped_rows(question, aggregated_rows, operation, group_column)
            if limit is not None:
                aggregated_rows = aggregated_rows[:limit]
            display_rows = aggregated_rows[:10]
            group_clause = f" GROUP BY {self._safe_identifier(group_column)}"
        else:
            working_rows = list(matched_rows)
            if row_sort is not None:
                working_rows = self._sort_rows(working_rows, row_sort["column"], row_sort["direction"])
                order_clause = (
                    f" ORDER BY {self._safe_identifier(str(row_sort['column']))} "
                    f"{'DESC' if row_sort['direction'] == 'desc' else 'ASC'}"
                )
            if limit is not None and operation == "rows":
                working_rows = working_rows[:limit]
                if not order_clause:
                    order_clause = " ORDER BY row_index ASC"
            display_rows = working_rows[:10]

        select_sql = self._build_select_sql(
            operation=operation,
            group_column=group_column,
            numeric_column=numeric_column,
        )
        where_clauses = self._filters_to_sql(applied_filters)
        if used_fallback and not where_clauses:
            where_clauses.append("-- token-match fallback")

        proposed_sql = f"{select_sql} FROM {self._safe_identifier(table_name)}"
        if where_clauses:
            proposed_sql += " WHERE " + " AND ".join(where_clauses)
        proposed_sql += group_clause
        proposed_sql += order_clause
        if limit is not None:
            proposed_sql += f" LIMIT {limit}"
        return {
            "operation": operation,
            "numeric_column": numeric_column,
            "matched_rows": matched_rows,
            "display_rows": display_rows,
            "group_column": group_column,
            "aggregated_rows": aggregated_rows,
            "filter_clauses": where_clauses,
            "proposed_sql": proposed_sql,
            "limit": limit,
            "row_sort": row_sort,
            "used_fallback": used_fallback,
        }

    def _detect_operation(self, question: str, rows: list[dict[str, Any]]) -> str:
        normalized_query = self._normalize_text(question)
        row_context = {"show", "list", "record", "records", "beneficiary", "beneficiaries", "name", "which", "who", "dikhao"}
        sort_context = {"top", "highest", "lowest", "largest", "smallest", "maximum", "minimum"}
        if any(token in normalized_query for token in sort_context) and any(token in normalized_query for token in row_context):
            return "rows"

        avg_intents = {"average", "avg", "mean", "aosat", "ausat"}
        amount_intents = {
            "amount",
            "pese",
            "paise",
            "paisa",
            "money",
            "rupees",
            "rupaye",
            "rs",
            "inr",
            "sanction amount",
            "approved amount",
        }
        sum_intents = {"sum", "total", "kul", "jod"}
        max_intents = {"maximum", "max", "highest", "largest", "sabse zyada"}
        min_intents = {"minimum", "min", "lowest", "smallest", "sabse kam"}
        count_intents = {"kitne", "kitni", "kitna", "how many", "count", "sankhya", "number of"}

        if any(token in normalized_query for token in avg_intents):
            return "avg"
        if any(token in normalized_query for token in max_intents) and self._pick_numeric_column(question, rows):
            return "max"
        if any(token in normalized_query for token in min_intents) and self._pick_numeric_column(question, rows):
            return "min"
        if any(token in normalized_query for token in amount_intents | sum_intents):
            return "sum"
        if any(token in normalized_query for token in count_intents):
            return "count"
        return "rows"

    def _extract_value_filters(self, question: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        normalized_query = self._normalize_text(question)
        matched_filters: list[dict[str, Any]] = []
        seen_columns: set[str] = set()
        for column in rows[0].keys():
            distinct_values = self._distinct_text_values(rows, column)
            if not distinct_values or len(distinct_values) > 40:
                continue
            matched_values: list[tuple[int, str]] = []
            for value in distinct_values:
                score = self._score_query_value_match(normalized_query, value)
                if score > 0:
                    matched_values.append((score, value))
            matched_values.sort(key=lambda item: item[0], reverse=True)
            selected_values: list[str] = []
            for _, value in matched_values:
                normalized_value = self._normalize_text(value)
                if normalized_value not in {self._normalize_text(item) for item in selected_values}:
                    selected_values.append(value)
                if len(selected_values) >= 3:
                    break
            if selected_values and column not in seen_columns:
                matched_filters.append({"column": str(column), "values": selected_values, "kind": "text"})
                seen_columns.add(str(column))
        matched_filters.extend(self._extract_numeric_filters(normalized_query, rows, seen_columns))
        return matched_filters

    def _extract_numeric_filters(
        self,
        normalized_query: str,
        rows: list[dict[str, Any]],
        seen_columns: set[str],
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        filters: list[dict[str, Any]] = []
        year_values = re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", normalized_query)
        if not year_values:
            return filters
        for column in rows[0].keys():
            normalized_column = self._normalize_text(column)
            if column in seen_columns:
                continue
            if "year" not in normalized_column and "saal" not in normalized_query:
                continue
            distinct_numbers = sorted(
                {
                    int(float(str(row.get(column)).replace(",", "").strip()))
                    for row in rows
                    if self._is_number_like(row.get(column))
                }
            )
            selected_values = [int(value) for value in year_values if int(value) in distinct_numbers]
            if selected_values:
                filters.append({"column": str(column), "values": selected_values, "kind": "number"})
                seen_columns.add(str(column))
        return filters

    def _apply_value_filters(self, rows: list[dict[str, Any]], filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not filters:
            return []
        matches: list[dict[str, Any]] = []
        for row in rows:
            include = True
            for filter_item in filters:
                cell_value = row.get(filter_item["column"])
                if not self._cell_matches_filter(cell_value, filter_item):
                    include = False
                    break
            if include:
                matches.append(row)
        return matches

    def _cell_matches_filter(self, cell_value: Any, filter_item: dict[str, Any]) -> bool:
        values = filter_item.get("values") or []
        if filter_item.get("kind") == "number":
            if not self._is_number_like(cell_value):
                return False
            numeric_value = int(float(str(cell_value).replace(",", "").strip()))
            return numeric_value in {int(value) for value in values}
        normalized_cell = self._normalize_text(cell_value)
        normalized_values = {self._normalize_text(value) for value in values}
        return normalized_cell in normalized_values

    def _distinct_text_values(self, rows: list[dict[str, Any]], column: str) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for row in rows:
            value = row.get(column)
            if value is None or self._is_number_like(value):
                continue
            normalized = self._normalize_text(value)
            if len(normalized) < 2 or normalized in seen:
                continue
            seen.add(normalized)
            values.append(str(value))
        values.sort(key=lambda item: len(self._normalize_text(item)), reverse=True)
        return values

    def _score_query_value_match(self, normalized_query: str, value: str) -> int:
        normalized_value = self._normalize_text(value)
        if not normalized_value:
            return -1
        if normalized_value in normalized_query:
            return len(normalized_value)

        value_tokens = [token for token in self._tokenize(normalized_value) if len(token) > 1]
        if value_tokens and all(token in normalized_query for token in value_tokens):
            return len(value_tokens) * 10

        status_synonyms = {
            "approved": ["approved", "approve", "sanctioned", "sanction", "pass", "passed"],
            "pending": ["pending", "awaiting", "rukha", "ruka", "under process"],
            "rejected": ["rejected", "reject", "denied", "cancelled", "canceled"],
        }
        for canonical, synonyms in status_synonyms.items():
            if normalized_value == canonical and any(synonym in normalized_query for synonym in synonyms):
                return 25
        return -1

    def _detect_group_column(
        self,
        question: str,
        rows: list[dict[str, Any]],
        filters: list[dict[str, Any]],
    ) -> str | None:
        if not rows:
            return None
        normalized_query = self._normalize_text(question)
        for column in rows[0].keys():
            for alias in self._column_aliases(str(column)):
                if (
                    f"{alias} wise" in normalized_query
                    or f"{alias}-wise" in normalized_query
                    or f"by {alias}" in normalized_query
                    or f"per {alias}" in normalized_query
                    or f"group by {alias}" in normalized_query
                    or f"{alias} ke hisab" in normalized_query
                    or f"{alias} ke hisaab" in normalized_query
                ):
                    return str(column)
        if any(token in normalized_query for token in {"compare", "comparison", "vs", "versus"}):
            comparison_column = self._detect_comparison_column(question, filters)
            if comparison_column:
                return comparison_column
        return None

    def _detect_comparison_column(self, question: str, filters: list[dict[str, Any]]) -> str | None:
        normalized_query = self._normalize_text(question)
        if not any(token in normalized_query for token in {"compare", "comparison", "vs", "versus", "breakup", "split"}):
            return None
        for item in filters:
            if len(item.get("values") or []) > 1:
                return str(item["column"])
        return None

    def _aggregate_by_column(
        self,
        rows: list[dict[str, Any]],
        *,
        group_column: str,
        operation: str,
        numeric_column: str | None,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get(group_column) or "Unknown")
            grouped.setdefault(key, []).append(row)

        results: list[dict[str, Any]] = []
        metric_key = self._metric_label(operation, numeric_column)
        for key, items in grouped.items():
            result: dict[str, Any] = {group_column: key, "row_count": len(items)}
            if operation == "count":
                result[metric_key] = len(items)
            elif operation == "sum" and numeric_column:
                result[metric_key] = round(sum(self._to_number(item.get(numeric_column)) for item in items), 2)
            elif operation == "avg" and numeric_column:
                values = [self._to_number(item.get(numeric_column)) for item in items]
                result[metric_key] = round(sum(values) / len(values), 2) if values else 0.0
            elif operation == "max" and numeric_column:
                values = [self._to_number(item.get(numeric_column)) for item in items]
                result[metric_key] = max(values) if values else 0.0
            elif operation == "min" and numeric_column:
                values = [self._to_number(item.get(numeric_column)) for item in items]
                result[metric_key] = min(values) if values else 0.0
            else:
                result[metric_key] = len(items)
            results.append(result)
        return results

    def _detect_row_sort(
        self,
        question: str,
        rows: list[dict[str, Any]],
        numeric_column: str | None,
    ) -> dict[str, str] | None:
        normalized_query = self._normalize_text(question)
        direction = None
        if any(token in normalized_query for token in {"highest", "largest", "top", "maximum", "most", "sabse zyada"}):
            direction = "desc"
        elif any(token in normalized_query for token in {"lowest", "smallest", "bottom", "minimum", "least", "sabse kam"}):
            direction = "asc"
        if direction is None:
            return None
        sort_column = numeric_column or self._pick_numeric_column(question, rows)
        if not sort_column:
            return None
        return {"column": str(sort_column), "direction": direction}

    def _detect_limit(self, question: str) -> int | None:
        normalized_query = self._normalize_text(question)
        match = re.search(r"\b(?:top|first|last|bottom)\s+(\d+)\b", normalized_query)
        if match:
            return max(1, min(50, int(match.group(1))))
        if any(token in normalized_query for token in {"top", "highest", "lowest", "bottom", "first", "last"}):
            return 5
        return None

    def _order_grouped_rows(
        self,
        question: str,
        grouped_rows: list[dict[str, Any]],
        operation: str,
        group_column: str,
    ) -> tuple[list[dict[str, Any]], str]:
        if not grouped_rows:
            return grouped_rows, ""
        metric_key = next((key for key in grouped_rows[0].keys() if key not in {group_column, "row_count"}), "row_count")
        normalized_query = self._normalize_text(question)
        direction = "desc"
        if any(token in normalized_query for token in {"lowest", "smallest", "minimum", "least", "bottom", "sabse kam"}):
            direction = "asc"
        ordered = sorted(
            grouped_rows,
            key=lambda item: self._to_number(item.get(metric_key)),
            reverse=(direction == "desc"),
        )
        return ordered, f" ORDER BY {self._safe_identifier(metric_key)} {'DESC' if direction == 'desc' else 'ASC'}"

    def _sort_rows(self, rows: list[dict[str, Any]], column: str, direction: str) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda item: self._to_number(item.get(column)),
            reverse=(direction == "desc"),
        )

    def _build_select_sql(self, *, operation: str, group_column: str | None, numeric_column: str | None) -> str:
        metric_label = self._metric_label(operation, numeric_column)
        if group_column:
            if operation == "count":
                return f"SELECT {self._safe_identifier(group_column)}, COUNT(*) AS {self._safe_identifier(metric_label)}"
            if operation in {"sum", "avg", "min", "max"} and numeric_column:
                return (
                    f"SELECT {self._safe_identifier(group_column)}, "
                    f"{operation.upper()}({self._safe_identifier(numeric_column)}) AS {self._safe_identifier(metric_label)}"
                )
            return f"SELECT {self._safe_identifier(group_column)}, COUNT(*) AS {self._safe_identifier(metric_label)}"
        if operation == "count":
            return "SELECT COUNT(*) AS matching_rows"
        if operation in {"sum", "avg", "min", "max"} and numeric_column:
            return f"SELECT {operation.upper()}({self._safe_identifier(numeric_column)}) AS {self._safe_identifier(metric_label)}"
        return "SELECT *"

    def _filters_to_sql(self, filters: list[dict[str, Any]]) -> list[str]:
        clauses: list[str] = []
        for filter_item in filters:
            column = self._safe_identifier(str(filter_item["column"]))
            values = filter_item.get("values") or []
            if not values:
                continue
            if filter_item.get("kind") == "number":
                normalized_values = ", ".join(str(int(value)) for value in values)
                if len(values) == 1:
                    clauses.append(f"{column} = {int(values[0])}")
                else:
                    clauses.append(f"{column} IN ({normalized_values})")
            else:
                normalized_values = [
                    f"'{self._escape_sql_literal(self._normalize_text(str(value)))}'"
                    for value in values
                ]
                if len(normalized_values) == 1:
                    clauses.append(f"LOWER({column}) = {normalized_values[0]}")
                else:
                    clauses.append(f"LOWER({column}) IN ({', '.join(normalized_values)})")
        return clauses

    def _build_answer_from_plan(self, plan: dict[str, Any]) -> str:
        operation = str(plan["operation"])
        group_column = plan.get("group_column")
        numeric_column = plan.get("numeric_column")
        matched_rows = list(plan.get("matched_rows") or [])
        display_rows = list(plan.get("display_rows") or [])
        row_sort = plan.get("row_sort")
        limit = plan.get("limit")

        if group_column:
            metric_label = self._metric_label(operation, numeric_column)
            if not display_rows:
                return f"`{group_column}` wise summary generate hui, but matching rows nahi mile."
            top_entry = display_rows[0]
            return (
                f"`{group_column}` wise `{self._humanize_metric_label(metric_label)}` summary generate hui. "
                f"Top result: {top_entry.get(group_column)} -> {top_entry.get(metric_label)}."
            )

        if operation == "sum":
            total = sum(self._to_number(row.get(numeric_column)) for row in matched_rows) if numeric_column else 0.0
            if numeric_column:
                return f"Column `{numeric_column}` ka total {total:.2f} hai across {len(matched_rows)} matching rows."
            return "Query me koi clear numeric column identify nahi hua."
        if operation == "avg":
            values = [self._to_number(row.get(numeric_column)) for row in matched_rows] if numeric_column else []
            average = (sum(values) / len(values)) if values else 0.0
            if numeric_column:
                return f"Column `{numeric_column}` ka average {average:.2f} hai across {len(matched_rows)} matching rows."
            return "Average ke liye koi clear numeric column identify nahi hua."
        if operation == "max":
            values = [self._to_number(row.get(numeric_column)) for row in matched_rows] if numeric_column else []
            maximum = max(values) if values else 0.0
            if numeric_column:
                return f"Column `{numeric_column}` ki maximum value {maximum:.2f} hai across {len(matched_rows)} matching rows."
            return "Maximum ke liye koi clear numeric column identify nahi hua."
        if operation == "min":
            values = [self._to_number(row.get(numeric_column)) for row in matched_rows] if numeric_column else []
            minimum = min(values) if values else 0.0
            if numeric_column:
                return f"Column `{numeric_column}` ki minimum value {minimum:.2f} hai across {len(matched_rows)} matching rows."
            return "Minimum ke liye koi clear numeric column identify nahi hua."
        if operation == "count":
            return f"Total matching records: {len(matched_rows)}."
        if row_sort:
            direction = "descending" if row_sort.get("direction") == "desc" else "ascending"
            prefix = f"Top {limit} sorted rows" if limit else "Sorted rows"
            return f"{prefix} `{row_sort.get('column')}` ke hisaab se {direction} order me dikhayi ja rahi hain."
        if plan.get("filter_clauses"):
            return f"{len(display_rows)} rows preview me dikh rahi hain. Query ke filters apply kiye gaye hain."
        return "Exact filter detect nahi hua, isliye representative rows dikhaye ja rahe hain."

    def _metric_label(self, operation: str, numeric_column: str | None) -> str:
        if operation == "count":
            return "matching_rows"
        if operation in {"sum", "avg", "min", "max"} and numeric_column:
            prefix_map = {
                "sum": "total",
                "avg": "average",
                "min": "minimum",
                "max": "maximum",
            }
            return f"{prefix_map.get(operation, operation)}_{self._safe_identifier(numeric_column)}"
        return "matching_rows"

    @staticmethod
    def _humanize_metric_label(metric_label: str) -> str:
        return metric_label.replace("_", " ")

    def _column_aliases(self, column: str) -> set[str]:
        normalized = self._normalize_text(column)
        aliases = {normalized}
        aliases.update(token for token in re.split(r"[_\s]+", normalized) if token)
        if "scheme" in normalized:
            aliases.update({"scheme", "yojana", "program"})
        if "status" in normalized:
            aliases.update({"status", "approval", "state"})
        if "village" in normalized:
            aliases.update({"village", "gaon", "city", "location", "district"})
        if "year" in normalized:
            aliases.update({"year", "saal"})
        if "amount" in normalized:
            aliases.update({"amount", "pese", "paise", "rupaye", "money", "fund"})
        if "name" in normalized:
            aliases.update({"name", "beneficiary", "applicant"})
        return aliases

    def _extract_rows_from_structured_metadata(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        extracted_rows: list[dict[str, Any]] = []
        for record in records:
            tables = record.get("structured_tables") or []
            if isinstance(tables, str):
                try:
                    tables = json.loads(tables)
                except json.JSONDecodeError:
                    tables = []
            if not isinstance(tables, list):
                continue
            for table in tables:
                if not isinstance(table, dict):
                    continue
                headers = table.get("headers") or []
                rows = table.get("rows") or []
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if isinstance(row, dict):
                        normalized = {self._slugify_header(str(key), index): value for index, (key, value) in enumerate(row.items(), start=1)}
                        if normalized:
                            extracted_rows.append(normalized)
                    elif isinstance(row, list) and headers and len(row) == len(headers):
                        extracted_rows.append(
                            {
                                self._slugify_header(str(header), index): row[index - 1]
                                for index, header in enumerate(headers, start=1)
                            }
                        )
        return extracted_rows

    def _build_metadata_preview(self, records: list[dict[str, Any]]) -> str:
        previews: list[str] = []
        for record in records[:5]:
            structured_summary = str(record.get("structured_summary") or "").strip()
            if structured_summary:
                previews.append(structured_summary)
            original_text = str(record.get("original_text") or "").strip()
            if original_text:
                previews.append(original_text[:220])
        return "\n\n".join(previews)[:600]

    def _parse_csv_bytes(self, file_bytes: bytes) -> list[dict[str, Any]]:
        decoded = file_bytes.decode("utf-8-sig", errors="ignore")
        reader = csv.reader(StringIO(decoded))
        rows = [row for row in reader if any(str(cell).strip() for cell in row)]
        return self._rows_to_dicts(rows)

    def _parse_json_bytes(self, file_bytes: bytes) -> list[dict[str, Any]]:
        payload = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            items = payload["rows"]
        else:
            items = [payload] if isinstance(payload, dict) else []

        rows: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                rows.append({self._slugify_header(str(key), index): value for index, (key, value) in enumerate(item.items(), start=1)})
        return rows

    def _parse_xlsx_bytes(self, file_bytes: bytes) -> list[tuple[str, list[dict[str, Any]]]]:
        workbook_tables: list[tuple[str, list[dict[str, Any]]]] = []
        with ZipFile(BytesIO(file_bytes)) as archive:
            shared_strings = self._read_xlsx_shared_strings(archive)
            sheet_map = self._read_xlsx_sheet_map(archive)
            for sheet_name, sheet_path in sheet_map:
                if sheet_path not in archive.namelist():
                    continue
                sheet_rows = self._read_xlsx_sheet_rows(archive.read(sheet_path), shared_strings)
                parsed_rows = self._rows_to_dicts(sheet_rows)
                if parsed_rows:
                    workbook_tables.append((sheet_name, parsed_rows))
        return workbook_tables

    def _rows_to_dicts(self, rows: list[list[Any]]) -> list[dict[str, Any]]:
        normalized_rows = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in rows
            if any(str(cell).strip() for cell in row)
        ]
        if len(normalized_rows) < 2:
            return []
        header_row = normalized_rows[0]
        if self._looks_like_header(header_row):
            headers = [self._slugify_header(cell, index) for index, cell in enumerate(header_row, start=1)]
            data_rows = normalized_rows[1:]
        else:
            headers = [f"col_{index}" for index in range(1, len(header_row) + 1)]
            data_rows = normalized_rows
        return [
            {headers[index]: row[index] if index < len(row) else "" for index in range(len(headers))}
            for row in data_rows
            if any(str(cell).strip() for cell in row)
        ]

    def _read_xlsx_shared_strings(self, archive: ZipFile) -> list[str]:
        shared_path = "xl/sharedStrings.xml"
        if shared_path not in archive.namelist():
            return []
        root = ET.fromstring(archive.read(shared_path))
        namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        values: list[str] = []
        for item in root.findall("main:si", namespace):
            text_parts = [node.text or "" for node in item.findall(".//main:t", namespace)]
            values.append("".join(text_parts))
        return values

    def _read_xlsx_sheet_map(self, archive: ZipFile) -> list[tuple[str, str]]:
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        workbook_ns = {
            "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
            "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        }
        rels_ns = {"pkg": "http://schemas.openxmlformats.org/package/2006/relationships"}
        rel_target_by_id = {
            relation.attrib.get("Id", ""): relation.attrib.get("Target", "")
            for relation in rels_root.findall("pkg:Relationship", rels_ns)
        }
        sheet_map: list[tuple[str, str]] = []
        for sheet in workbook_root.findall("main:sheets/main:sheet", workbook_ns):
            sheet_name = sheet.attrib.get("name", "sheet")
            relation_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "")
            target = rel_target_by_id.get(relation_id, "")
            if not target:
                continue
            normalized_target = target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"
            sheet_map.append((sheet_name, normalized_target))
        return sheet_map

    def _read_xlsx_sheet_rows(self, sheet_bytes: bytes, shared_strings: list[str]) -> list[list[str]]:
        root = ET.fromstring(sheet_bytes)
        namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows: list[list[str]] = []
        for row_node in root.findall(".//main:sheetData/main:row", namespace):
            cell_map: dict[int, str] = {}
            max_index = 0
            for cell in row_node.findall("main:c", namespace):
                ref = cell.attrib.get("r", "")
                col_index = self._xlsx_column_index(ref)
                max_index = max(max_index, col_index)
                value = self._xlsx_cell_value(cell, shared_strings, namespace)
                cell_map[col_index] = value
            if max_index == 0:
                continue
            rows.append([cell_map.get(index, "") for index in range(1, max_index + 1)])
        return rows

    @staticmethod
    def _xlsx_column_index(cell_reference: str) -> int:
        letters = "".join(char for char in cell_reference if char.isalpha()).upper()
        index = 0
        for char in letters:
            index = (index * 26) + (ord(char) - 64)
        return index

    @staticmethod
    def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str], namespace: dict[str, str]) -> str:
        cell_type = cell.attrib.get("t", "")
        if cell_type == "inlineStr":
            text_parts = [node.text or "" for node in cell.findall(".//main:t", namespace)]
            return "".join(text_parts).strip()
        value_node = cell.find("main:v", namespace)
        raw_value = (value_node.text or "").strip() if value_node is not None else ""
        if cell_type == "s":
            try:
                return str(shared_strings[int(raw_value)])
            except (ValueError, IndexError):
                return raw_value
        return raw_value

    def _insert_table(
        self,
        *,
        document_id: str,
        source_name: str,
        source_type: str,
        ingestion_mode: str,
        table_name: str,
        rows: list[dict[str, Any]],
        extraction_mode: str,
        page_hint: str,
        notes: str,
        raw_preview: str,
    ) -> int:
        schema = list(rows[0].keys()) if rows else []
        timestamp = self._utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO documents (
                    document_id, source_name, source_type, ingestion_mode, notes, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (document_id, source_name, source_type, ingestion_mode, notes, timestamp),
            )
            cursor = connection.execute(
                """
                INSERT INTO tables_meta (
                    document_id, table_name, page_hint, schema_json, row_count, extraction_mode, raw_preview, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    table_name,
                    page_hint,
                    json.dumps(schema, ensure_ascii=False),
                    len(rows),
                    extraction_mode,
                    raw_preview,
                    timestamp,
                ),
            )
            table_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO table_rows (table_id, row_index, row_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (table_id, index, json.dumps(row, ensure_ascii=False), timestamp)
                    for index, row in enumerate(rows, start=1)
                ],
            )
        return table_id

    def _delete_document_if_exists(self, document_id: str) -> None:
        with self._connect() as connection:
            table_ids = [
                int(row["table_id"])
                for row in connection.execute(
                    "SELECT table_id FROM tables_meta WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
            ]
            for table_id in table_ids:
                connection.execute("DELETE FROM table_rows WHERE table_id = ?", (table_id,))
                connection.execute("DELETE FROM query_logs WHERE table_id = ?", (table_id,))
            connection.execute("DELETE FROM tables_meta WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    def _extract_rows_from_text(self, text: str) -> list[dict[str, Any]]:
        candidates: list[list[str]] = []
        for raw_line in text.splitlines():
            line = " ".join(raw_line.split()).strip(" |")
            if not line:
                continue
            if "|" in line:
                cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            elif "\t" in raw_line:
                cells = [cell.strip() for cell in raw_line.split("\t") if cell.strip()]
            else:
                cells = [cell.strip() for cell in re.split(r"\s{2,}", raw_line) if cell.strip()]
            if len(cells) >= 2:
                candidates.append(cells)

        if len(candidates) < 2:
            return []

        width_counter = Counter(len(candidate) for candidate in candidates)
        target_width = width_counter.most_common(1)[0][0]
        normalized_rows = [row for row in candidates if len(row) == target_width]
        if len(normalized_rows) < 2:
            return []

        header_row = normalized_rows[0]
        if self._looks_like_header(header_row):
            headers = [self._slugify_header(cell, index) for index, cell in enumerate(header_row, start=1)]
            data_rows = normalized_rows[1:]
        else:
            headers = [f"col_{index}" for index in range(1, target_width + 1)]
            data_rows = normalized_rows
        return [dict(zip(headers, row, strict=False)) for row in data_rows if len(row) == len(headers)]

    def _build_fallback_rows(self, text: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, raw_line in enumerate(text.splitlines(), start=1):
            line = " ".join(raw_line.split()).strip()
            if len(line) < 8:
                continue
            rows.append({"row_no": index, "raw_text": line})
            if len(rows) >= 25:
                break
        return rows

    def _log_query(self, *, table_id: int, user_query: str, proposed_sql: str, answer_text: str, matched_rows: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO query_logs (table_id, user_query, proposed_sql, answer_text, matched_rows, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (table_id, user_query, proposed_sql, answer_text, matched_rows, self._utc_now()),
            )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text or "").lower().split())

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in re.split(r"\W+", text.lower()) if token]

    @staticmethod
    def _safe_identifier(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_") or "table_data"

    @staticmethod
    def _escape_sql_literal(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _slugify_header(text: str, index: int) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
        return slug or f"col_{index}"

    @staticmethod
    def _looks_like_header(row: list[str]) -> bool:
        alpha_cells = sum(1 for cell in row if re.search(r"[A-Za-z\u0900-\u097F]", cell))
        return alpha_cells >= max(1, len(row) - 1)

    @staticmethod
    def _is_number_like(value: Any) -> bool:
        try:
            float(str(value).replace(",", "").strip())
            return True
        except ValueError:
            return False

    @staticmethod
    def _to_number(value: Any) -> float:
        try:
            return float(str(value).replace(",", "").strip())
        except ValueError:
            return 0.0

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat()
