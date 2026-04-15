from __future__ import annotations

import csv
import json
import re
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from src.core.config import AppConfig
from src.services.document_store import DocumentStore
from src.services.sarvam_service import SarvamService


class IngestionService:
    OCR_ONLY_SUFFIXES = {".png", ".jpg", ".jpeg"}
    OCR_CAPABLE_SUFFIXES = OCR_ONLY_SUFFIXES | {".pdf", ".zip"}
    LOCAL_PARSE_SUFFIXES = {".txt", ".md", ".csv", ".json", ".pdf"}
    ZIP_PASS_THROUGH_IMAGE_SUFFIXES = OCR_ONLY_SUFFIXES

    def __init__(self, config: AppConfig, store: DocumentStore) -> None:
        self.config = config
        self.store = store

    def ingest_uploaded_file(
        self,
        uploaded_file: Any,
        sarvam: SarvamService,
        language_code: str,
        use_ocr: bool,
        build_translation_index: bool = True,
    ) -> dict[str, Any]:
        document_id = uuid.uuid4().hex
        saved_path = self._save_upload(uploaded_file, document_id)
        file_suffix = saved_path.suffix.lower()
        ingested_at = datetime.now(timezone.utc).isoformat()
        extraction_language_code = "en-IN" if language_code == "auto" else language_code

        extraction_warnings: list[str] = []
        ocr_failure_message: str | None = None
        local_segments_fallback: list[dict[str, Any]] | None = None
        ocr_enabled = use_ocr or file_suffix in self.OCR_ONLY_SUFFIXES
        if language_code == "auto":
            extraction_warnings.append(
                "Auto language detect enabled: OCR ran with a neutral English hint, then language was auto-detected for indexing."
            )
        if file_suffix in self.OCR_ONLY_SUFFIXES and not use_ocr:
            extraction_warnings.append(
                "OCR was auto-enabled because image uploads require OCR extraction."
            )
        if file_suffix == ".zip":
            segments, extraction_method, zip_warnings = self._extract_zip_segments(
                archive_path=saved_path,
                sarvam=sarvam,
                language_code=extraction_language_code,
                use_ocr=use_ocr,
            )
            extraction_warnings.extend(zip_warnings)
        elif file_suffix == ".pdf" and not ocr_enabled:
            local_segments_fallback = self._extract_local_segments(saved_path)
            if self._looks_like_scanned_pdf(local_segments_fallback) and sarvam.is_configured:
                ocr_enabled = True
                extraction_warnings.append(
                    "Selectable text was not detected in this PDF, so OCR was auto-enabled for scanned/handwritten pages."
                )

            if ocr_enabled and file_suffix in self.OCR_CAPABLE_SUFFIXES:
                try:
                    segments, extraction_method = sarvam.extract_document_segments(
                        file_path=saved_path,
                        language_code=extraction_language_code,
                        output_dir=self.config.ocr_dir,
                    )
                except Exception as exc:
                    ocr_failure_message = str(exc)
                    segments = local_segments_fallback or self._extract_local_segments(saved_path)
                    extraction_method = "local-parser-fallback"
                    extraction_warnings.append(f"OCR fallback applied: {exc}")
            else:
                segments = local_segments_fallback or self._extract_local_segments(saved_path)
                extraction_method = "local-parser"
        elif ocr_enabled and file_suffix in self.OCR_CAPABLE_SUFFIXES:
            try:
                segments, extraction_method = sarvam.extract_document_segments(
                    file_path=saved_path,
                    language_code=extraction_language_code,
                    output_dir=self.config.ocr_dir,
                )
            except Exception as exc:
                ocr_failure_message = str(exc)
                raise
        else:
            segments = local_segments_fallback or self._extract_local_segments(saved_path)
            extraction_method = "local-parser"

        if not any((segment.get("text") or "").strip() for segment in segments):
            if ocr_enabled and file_suffix == ".pdf":
                if ocr_failure_message:
                    raise ValueError(
                        "Sarvam OCR could not extract usable text from this PDF, and the local PDF parser also found no selectable text. "
                        f"Underlying OCR error: {ocr_failure_message}"
                    )
                raise ValueError(
                    "Sarvam OCR did not return usable text, and the local PDF parser also found no selectable text. "
                    "This PDF appears to be image-only or scanned."
                )
            raise ValueError("No text could be extracted from this file.")

        effective_language_code = self._resolve_document_language(
            requested_language_code=language_code,
            segments=segments,
            sarvam=sarvam,
            translation_enabled=build_translation_index,
        )
        if effective_language_code != language_code:
            extraction_warnings.append(
                f"Detected document language override applied: requested={language_code}, detected={effective_language_code}"
            )

        chunk_payloads = self._build_chunk_payloads(
            segments=segments,
            language_code=effective_language_code,
            sarvam=sarvam,
            enabled=build_translation_index,
            extraction_method=extraction_method,
        )

        records: list[dict[str, Any]] = []
        file_size_bytes = saved_path.stat().st_size
        original_filename = self._original_filename(saved_path)
        for idx, payload in enumerate(chunk_payloads):
            chunk = payload["original_text"]
            english_text = payload["translated_text"]
            page_start = payload["page_start"]
            page_end = payload["page_end"]
            archive_entry_name = payload.get("archive_entry_name")
            structured_tables = payload.get("structured_tables") or []
            structured_summary = payload.get("structured_summary") or ""
            location_prefix = self._build_location_prefix(
                original_filename,
                page_start,
                page_end,
                archive_entry_name=archive_entry_name,
            )
            search_text_parts = [location_prefix, english_text]
            search_text = "\n".join(part.strip() for part in search_text_parts if part and part.strip()).strip()
            records.append(
                {
                    "chunk_id": f"{document_id}:{idx}",
                    "document_id": document_id,
                    "chunk_index": idx,
                    "source_name": original_filename,
                    "stored_filename": saved_path.name,
                    "file_extension": saved_path.suffix.lower(),
                    "file_size_bytes": file_size_bytes,
                    "language_code": effective_language_code,
                    "extraction_method": extraction_method,
                    "ingested_at": ingested_at,
                    "page_start": page_start,
                    "page_end": page_end,
                    "char_count": len(chunk),
                    "word_count": len(chunk.split()),
                    "original_text": chunk,
                    "translated_text": english_text,
                    "retrieval_text": english_text,
                    "archive_entry_name": archive_entry_name,
                    "structured_tables": structured_tables,
                    "structured_table_count": len(structured_tables),
                    "structured_summary": structured_summary,
                    "translation_index_language": "en-IN",
                    "retrieval_language_code": "en-IN",
                    "search_text": search_text,
                }
            )

        if not records:
            raise ValueError(
                "The document was read, but no meaningful chunks remained after OCR cleanup and chunking. "
                "Try a clearer scan or upload a higher-resolution copy."
            )

        self.store.upsert_documents(records)
        return {
            "document_id": document_id,
            "source_name": original_filename,
            "chunk_count": len(records),
            "extraction_method": extraction_method,
            "language_code": effective_language_code,
            "index_language_code": "en-IN",
            "warnings": extraction_warnings,
        }

    def _save_upload(self, uploaded_file: Any, document_id: str) -> Path:
        safe_name = uploaded_file.name.replace(" ", "_")
        destination = self.config.uploads_dir / f"{document_id}_{safe_name}"
        destination.write_bytes(uploaded_file.getvalue())
        return destination

    def _extract_local_segments(self, path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return [{"text": path.read_text(encoding="utf-8", errors="ignore"), "page_start": None, "page_end": None}]
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            return [{"text": json.dumps(data, ensure_ascii=False, indent=2), "page_start": None, "page_end": None}]
        if suffix == ".csv":
            rows: list[str] = []
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    rows.append(" | ".join(row))
            return [{"text": "\n".join(rows), "page_start": None, "page_end": None}]
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            pages: list[dict[str, Any]] = []
            for page_number, page in enumerate(reader.pages, start=1):
                pages.append(
                    {
                        "text": page.extract_text() or "",
                        "page_start": page_number,
                        "page_end": page_number,
                    }
                )
            return pages
        raise ValueError(f"Unsupported file type for local parsing: {suffix}")

    def _extract_zip_segments(
        self,
        archive_path: Path,
        sarvam: SarvamService,
        language_code: str,
        use_ocr: bool,
    ) -> tuple[list[dict[str, Any]], str, list[str]]:
        warnings: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            member_names = self._extract_zip_members(archive_path, temp_root)
            extracted_files = [
                temp_root / member_name
                for member_name in member_names
                if (temp_root / member_name).is_file()
            ]
            if not extracted_files:
                raise ValueError("ZIP archive is empty or contains no supported files.")

            supported_files = [path for path in extracted_files if self._is_supported_zip_member(path)]
            skipped_files = [path for path in extracted_files if path not in supported_files]
            for skipped_path in skipped_files:
                rel_name = skipped_path.relative_to(temp_root).as_posix()
                warnings.append(f"Skipped unsupported archive entry: {rel_name}")

            if not supported_files:
                raise ValueError("ZIP archive did not contain any supported documents.")

            suffixes = {path.suffix.lower() for path in supported_files}
            if suffixes and suffixes.issubset(self.ZIP_PASS_THROUGH_IMAGE_SUFFIXES):
                if not sarvam.is_configured:
                    raise ValueError("ZIP archive contains image pages, so OCR requires a valid Sarvam API key.")
                archive_segments, archive_method = sarvam.extract_document_segments(
                    file_path=archive_path,
                    language_code=language_code,
                    output_dir=self.config.ocr_dir,
                )
                warnings.append("ZIP archive contained image pages, so OCR ran on the archive directly.")
                return archive_segments, archive_method, warnings

            combined_segments: list[dict[str, Any]] = []
            method_labels: list[str] = []
            for inner_path in sorted(supported_files, key=lambda path: path.relative_to(temp_root).as_posix()):
                rel_name = inner_path.relative_to(temp_root).as_posix()
                inner_segments, inner_method, inner_warnings = self._extract_inner_archive_file(
                    file_path=inner_path,
                    display_name=rel_name,
                    sarvam=sarvam,
                    language_code=language_code,
                    use_ocr=use_ocr,
                )
                method_labels.append(inner_method)
                warnings.extend(inner_warnings)
                combined_segments.extend(inner_segments)

            if not combined_segments:
                raise ValueError("ZIP archive was read, but no extractable text was found inside it.")

            resolved_method = "zip-expanded"
            unique_methods = sorted({label for label in method_labels if label})
            if unique_methods:
                resolved_method = f"zip-expanded[{', '.join(unique_methods)}]"
            return combined_segments, resolved_method, warnings

    def _extract_inner_archive_file(
        self,
        file_path: Path,
        display_name: str,
        sarvam: SarvamService,
        language_code: str,
        use_ocr: bool,
    ) -> tuple[list[dict[str, Any]], str, list[str]]:
        suffix = file_path.suffix.lower()
        warnings: list[str] = []

        if suffix in self.OCR_ONLY_SUFFIXES:
            if not sarvam.is_configured:
                raise ValueError(f"Archive entry '{display_name}' requires OCR, but no Sarvam API key is configured.")
            segments, extraction_method = sarvam.extract_document_segments(
                file_path=file_path,
                language_code=language_code,
                output_dir=self.config.ocr_dir,
            )
        elif suffix == ".pdf":
            local_segments = self._extract_local_segments(file_path)
            should_use_ocr = use_ocr
            if not should_use_ocr and self._looks_like_scanned_pdf(local_segments) and sarvam.is_configured:
                should_use_ocr = True
                warnings.append(
                    f"Archive entry '{display_name}' looked scanned, so OCR was auto-enabled for that PDF."
                )
            if should_use_ocr and sarvam.is_configured:
                try:
                    segments, extraction_method = sarvam.extract_document_segments(
                        file_path=file_path,
                        language_code=language_code,
                        output_dir=self.config.ocr_dir,
                    )
                except Exception as exc:
                    segments = local_segments
                    extraction_method = "local-parser-fallback"
                    warnings.append(f"Archive entry '{display_name}' fell back to local PDF parsing after OCR issue: {exc}")
            else:
                segments = local_segments
                extraction_method = "local-parser"
        else:
            segments = self._extract_local_segments(file_path)
            extraction_method = "local-parser"

        normalized_segments: list[dict[str, Any]] = []
        for segment in segments:
            normalized_segments.append(
                {
                    **segment,
                    "archive_entry_name": display_name,
                }
            )
        return normalized_segments, extraction_method, warnings

    @staticmethod
    def _extract_zip_members(archive_path: Path, temp_root: Path) -> list[Path]:
        extracted_paths: list[Path] = []
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_path = Path(member.filename)
                if member_path.name.startswith(".") or "__MACOSX" in member_path.parts:
                    continue
                destination = (temp_root / member_path).resolve()
                if temp_root.resolve() not in destination.parents and destination != temp_root.resolve():
                    raise ValueError("ZIP archive contains an unsafe file path.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source_handle, destination.open("wb") as target_handle:
                    target_handle.write(source_handle.read())
                extracted_paths.append(member_path)
        return extracted_paths

    def _is_supported_zip_member(self, path: Path) -> bool:
        return path.suffix.lower() in (self.LOCAL_PARSE_SUFFIXES | self.OCR_ONLY_SUFFIXES)

    def _build_chunk_payloads(
        self,
        segments: list[dict[str, Any]],
        language_code: str,
        sarvam: SarvamService,
        enabled: bool,
        extraction_method: str,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        preserve_noisy_ocr = extraction_method.startswith("sarvam-document-intelligence")
        for segment in segments:
            segment_text = (segment.get("text") or "").strip()
            if not segment_text:
                continue
            chunks = self._chunk_text(segment_text)
            translated_chunks = self._build_translation_index(
                chunks=chunks,
                language_code=language_code,
                sarvam=sarvam,
                enabled=enabled,
            )
            for index, chunk in enumerate(chunks):
                english_text = translated_chunks[index] if index < len(translated_chunks) else ""
                if self._is_low_quality_chunk(chunk, preserve_noisy_ocr=preserve_noisy_ocr):
                    continue
                structured_tables = self._extract_structured_tables(chunk)
                payloads.append(
                    {
                        "original_text": chunk,
                        "translated_text": english_text or chunk,
                        "page_start": segment.get("page_start"),
                        "page_end": segment.get("page_end"),
                        "archive_entry_name": segment.get("archive_entry_name"),
                        "structured_tables": structured_tables,
                        "structured_summary": self._summarize_structured_tables(structured_tables),
                    }
                )
        return payloads

    def _chunk_text(self, text: str) -> list[str]:
        paragraphs = self._normalize_paragraphs(text)
        if not paragraphs:
            return []

        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= self.config.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current.strip())
                overlap_text = self._tail_overlap(current)
                current = f"{overlap_text}\n\n{paragraph}".strip() if overlap_text else paragraph
            else:
                chunks.extend(self._split_long_paragraph(paragraph))
                current = ""

            if len(current) > self.config.chunk_size:
                chunks.extend(self._split_long_paragraph(current))
                current = ""

        if current:
            chunks.append(current.strip())
        return [chunk for chunk in chunks if chunk.strip()]

    def _build_translation_index(
        self,
        chunks: list[str],
        language_code: str,
        sarvam: SarvamService,
        enabled: bool,
    ) -> list[str]:
        if not enabled:
            return [" ".join(chunk.split()) for chunk in chunks]
        if language_code == "en-IN":
            return [" ".join(chunk.split()) for chunk in chunks]
        if not sarvam.is_configured:
            raise ValueError("English translation index requires a valid Sarvam API key for non-English documents.")

        translated: list[str] = []
        for chunk in chunks:
            english_text = sarvam.translate_text(
                text=chunk,
                source_language_code=language_code,
                target_language_code="en-IN",
            )
            if not english_text.strip():
                raise ValueError("A non-English document chunk could not be converted into English for vector indexing.")
            translated.append(" ".join(english_text.split()))
        return translated

    def _resolve_document_language(
        self,
        requested_language_code: str,
        segments: list[dict[str, Any]],
        sarvam: SarvamService,
        translation_enabled: bool,
    ) -> str:
        sample_text = "\n".join((segment.get("text") or "")[:1200] for segment in segments[:3])
        heuristic_language_code = self._detect_language_from_text(sample_text)

        if translation_enabled and sarvam.is_configured and sample_text.strip():
            try:
                _, detected_language_code = sarvam.translate_for_retrieval(
                    sample_text,
                    heuristic_language_code or requested_language_code or "auto",
                )
                if detected_language_code and detected_language_code != "auto":
                    return detected_language_code
            except Exception:
                pass

        if requested_language_code not in {"auto", ""}:
            return requested_language_code
        return heuristic_language_code or "en-IN"

    @staticmethod
    def _is_low_quality_chunk(text: str, preserve_noisy_ocr: bool = False) -> bool:
        chunk = (text or "").strip()
        if not chunk:
            return True

        lowered = chunk.lower()
        if "data:image/" in lowered or '"page_num"' in lowered or '"block_id"' in lowered:
            return True

        long_token_matches = re.findall(r"[A-Za-z0-9+/=_-]{80,}", chunk)
        if len(long_token_matches) >= 1 and not preserve_noisy_ocr:
            return True

        words = chunk.split()
        if not words:
            return True

        long_word_count = sum(1 for word in words if len(word) >= 40)
        if long_word_count >= max(2, len(words) // 3) and not preserve_noisy_ocr:
            return True

        alpha_count = sum(1 for char in chunk if char.isalpha())
        digit_count = sum(1 for char in chunk if char.isdigit())
        if alpha_count < 20 and digit_count > alpha_count and not preserve_noisy_ocr:
            return True

        return False

    @staticmethod
    def _detect_language_from_text(text: str) -> str | None:
        sample = (text or "").strip()
        if not sample:
            return None

        script_patterns: list[tuple[str, str]] = [
            (r"[\u0980-\u09FF]", "bn-IN"),
            (r"[\u0A80-\u0AFF]", "gu-IN"),
            (r"[\u0A00-\u0A7F]", "pa-IN"),
            (r"[\u0B00-\u0B7F]", "od-IN"),
            (r"[\u0B80-\u0BFF]", "ta-IN"),
            (r"[\u0C00-\u0C7F]", "te-IN"),
            (r"[\u0C80-\u0CFF]", "kn-IN"),
            (r"[\u0D00-\u0D7F]", "ml-IN"),
            (r"[\u0900-\u097F]", "hi-IN"),
        ]
        for pattern, language_code in script_patterns:
            if re.search(pattern, sample):
                return language_code
        return None

    @staticmethod
    def _looks_like_scanned_pdf(segments: list[dict[str, Any]]) -> bool:
        if not segments:
            return True

        page_texts = [(segment.get("text") or "").strip() for segment in segments]
        non_empty_pages = [text for text in page_texts if text]
        if not non_empty_pages:
            return True

        total_chars = sum(len(text) for text in non_empty_pages)
        short_pages = sum(1 for text in page_texts if len(text) < 40)
        return total_chars < 160 or short_pages >= max(1, len(page_texts) // 2)

    @staticmethod
    def _extract_structured_tables(text: str) -> list[dict[str, Any]]:
        markdown_tables = IngestionService._extract_markdown_tables(text)
        aligned_tables = IngestionService._extract_aligned_tables(text)
        tables = markdown_tables + aligned_tables
        deduped: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        for table in tables:
            signature = json.dumps(table, ensure_ascii=False, sort_keys=True)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            deduped.append(table)
        return deduped[:3]

    @staticmethod
    def _extract_markdown_tables(text: str) -> list[dict[str, Any]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        tables: list[dict[str, Any]] = []
        index = 0
        while index < len(lines) - 2:
            if "|" not in lines[index]:
                index += 1
                continue
            separator = lines[index + 1]
            if not re.fullmatch(r"[\|\-\:\s]+", separator):
                index += 1
                continue
            header_cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
            row_index = index + 2
            rows: list[dict[str, str]] = []
            while row_index < len(lines) and "|" in lines[row_index]:
                row_cells = [cell.strip() for cell in lines[row_index].strip("|").split("|")]
                if len(row_cells) != len(header_cells):
                    break
                rows.append(
                    {
                        header_cells[cell_index] or f"column_{cell_index + 1}": row_cells[cell_index]
                        for cell_index in range(len(header_cells))
                    }
                )
                row_index += 1
            if rows:
                tables.append(
                    {
                        "table_type": "markdown",
                        "headers": header_cells,
                        "rows": rows[:10],
                    }
                )
            index = row_index
        return tables

    @staticmethod
    def _extract_aligned_tables(text: str) -> list[dict[str, Any]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        candidate_groups: list[list[str]] = []
        current_group: list[str] = []
        for line in lines:
            if re.search(r"\S\s{2,}\S", line):
                current_group.append(line)
            else:
                if len(current_group) >= 2:
                    candidate_groups.append(current_group)
                current_group = []
        if len(current_group) >= 2:
            candidate_groups.append(current_group)

        tables: list[dict[str, Any]] = []
        for group in candidate_groups:
            split_rows = [re.split(r"\s{2,}", row) for row in group]
            column_count = max(len(row) for row in split_rows)
            if column_count < 2:
                continue
            normalized_rows = [row for row in split_rows if len(row) == column_count]
            if len(normalized_rows) < 2:
                continue
            headers = normalized_rows[0]
            rows = []
            for row in normalized_rows[1:11]:
                rows.append(
                    {
                        headers[cell_index] or f"column_{cell_index + 1}": row[cell_index]
                        for cell_index in range(column_count)
                    }
                )
            if rows:
                tables.append(
                    {
                        "table_type": "aligned_text",
                        "headers": headers,
                        "rows": rows,
                    }
                )
        return tables

    @staticmethod
    def _summarize_structured_tables(tables: list[dict[str, Any]]) -> str:
        if not tables:
            return ""

        parts: list[str] = []
        for index, table in enumerate(tables[:2], start=1):
            headers = [str(header).strip() for header in table.get("headers", []) if str(header).strip()]
            rows = table.get("rows", [])[:3]
            row_parts = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row_parts.append(", ".join(f"{key}: {value}" for key, value in row.items() if str(value).strip()))
            summary = f"Table {index} columns: {', '.join(headers[:6])}"
            if row_parts:
                summary += " | sample rows: " + " || ".join(row_parts)
            parts.append(summary)
        return "\n".join(parts)

    @staticmethod
    def _normalize_paragraphs(text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n")
        raw_parts = normalized.split("\n\n")
        paragraphs: list[str] = []
        for part in raw_parts:
            cleaned = " ".join(part.split())
            if cleaned:
                paragraphs.append(cleaned)
        if paragraphs:
            return paragraphs
        fallback = " ".join(text.split())
        return [fallback] if fallback else []

    def _split_long_paragraph(self, text: str) -> list[str]:
        compact = " ".join(text.split())
        if len(compact) <= self.config.chunk_size:
            return [compact]

        chunks: list[str] = []
        step = max(1, self.config.chunk_size - self.config.chunk_overlap)
        start = 0
        while start < len(compact):
            end = min(start + self.config.chunk_size, len(compact))
            chunk = compact[start:end]
            if end < len(compact):
                boundary = chunk.rfind(" ")
                if boundary > self.config.chunk_size // 2:
                    chunk = chunk[:boundary]
                    end = start + boundary
            chunks.append(chunk.strip())
            if end >= len(compact):
                break
            start = max(end - self.config.chunk_overlap, start + step)
        return [chunk for chunk in chunks if chunk]

    def _tail_overlap(self, text: str) -> str:
        compact = " ".join(text.split())
        if len(compact) <= self.config.chunk_overlap:
            return compact
        tail = compact[-self.config.chunk_overlap :]
        boundary = tail.find(" ")
        return tail[boundary + 1 :].strip() if boundary != -1 else tail.strip()

    @staticmethod
    def _original_filename(path: Path) -> str:
        parts = path.name.split("_", 1)
        return parts[1] if len(parts) == 2 else path.name

    @staticmethod
    def _build_location_prefix(
        source_name: str,
        page_start: int | None,
        page_end: int | None,
        archive_entry_name: str | None = None,
    ) -> str:
        parts = [f"Source: {source_name}"]
        if archive_entry_name:
            parts.append(f"Archive entry: {archive_entry_name}")
        if page_start is None:
            return "\n".join(parts)
        if page_start == page_end:
            parts.append(f"Page: {page_start}")
            return "\n".join(parts)
        parts.append(f"Pages: {page_start}-{page_end}")
        return "\n".join(parts)
