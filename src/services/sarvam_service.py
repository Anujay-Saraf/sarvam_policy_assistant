from __future__ import annotations

import asyncio
import base64
import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterator

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader, PdfWriter

from src.core.constants import (
    DEFAULT_STT_MODEL,
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_TTS_MODEL,
)


class SarvamService:
    CHAT_COMPLETIONS_URL = "https://api.sarvam.ai/v1/chat/completions"
    DEFAULT_DOCUMENT_LANGUAGE = "en-IN"

    def __init__(self, api_key: str | None) -> None:
        self.api_key = (api_key or "").strip()
        self._client: Any | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self) -> Any:
        if not self.is_configured:
            raise ValueError("Sarvam API key is missing.")
        if self._client is None:
            from sarvamai import SarvamAI

            self._client = SarvamAI(api_subscription_key=self.api_key)
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int = 900,
    ) -> dict[str, Any]:
        response = self._get_client().chat.completions(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return {"content": content, "raw": response}

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int = 900,
    ) -> Iterator[str]:
        if not self.is_configured:
            raise ValueError("Sarvam API key is missing.")

        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        with requests.post(
            self.CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=300,
        ) as response:
            if response.status_code >= 400:
                self._raise_stream_error(response)

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                chunk = self._extract_stream_text(event)
                if chunk:
                    yield chunk

    def translate_text(
        self,
        text: str,
        source_language_code: str,
        target_language_code: str,
        model: str = DEFAULT_TRANSLATION_MODEL,
    ) -> str:
        segments = self._split_for_translation(text)
        translated_segments: list[str] = []
        for segment in segments:
            response = self._get_client().text.translate(
                input=segment,
                source_language_code=source_language_code or "auto",
                target_language_code=target_language_code,
                model=model,
            )
            translated_segments.append(response.translated_text)
        return " ".join(part.strip() for part in translated_segments if part.strip())

    def translate_for_retrieval(self, text: str, input_language_code: str) -> tuple[str, str]:
        response = self._get_client().text.translate(
            input=text[:2000],
            source_language_code=input_language_code or "auto",
            target_language_code="en-IN",
            model=DEFAULT_TRANSLATION_MODEL,
        )
        translated_text = response.translated_text
        detected_language = getattr(response, "source_language_code", None) or input_language_code or "auto"
        return translated_text, detected_language

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        language_code: str | None,
        translate_to_english: bool,
    ) -> dict[str, Any]:
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(audio_bytes)
            temp_path = Path(handle.name)

        try:
            with temp_path.open("rb") as audio_handle:
                if translate_to_english:
                    response = self._get_client().speech_to_text.translate(
                        file=audio_handle,
                        model=DEFAULT_STT_MODEL,
                        mode="translate",
                    )
                else:
                    kwargs: dict[str, Any] = {
                        "file": audio_handle,
                        "model": DEFAULT_STT_MODEL,
                        "mode": "transcribe",
                    }
                    if language_code and language_code != "auto":
                        kwargs["language_code"] = language_code
                    response = self._get_client().speech_to_text.transcribe(**kwargs)
        finally:
            temp_path.unlink(missing_ok=True)

        return {
            "transcript": response.transcript,
            "language_code": getattr(response, "language_code", None),
            "raw": response,
        }

    def synthesize_speech(
        self,
        text: str,
        target_language_code: str,
        speaker: str,
        model: str = DEFAULT_TTS_MODEL,
        output_audio_codec: str | None = None,
    ) -> bytes:
        response = self._get_client().text_to_speech.convert(
            text=text[:2500],
            target_language_code=target_language_code,
            model=model,
            speaker=speaker,
            output_audio_codec=output_audio_codec,
        )
        return self._merge_tts_audio_chunks(getattr(response, "audios", None))

    def synthesize_speech_streaming(
        self,
        text: str,
        target_language_code: str,
        speaker: str,
        model: str = DEFAULT_TTS_MODEL,
        chunk_callback: Callable[[bytes, bool], None] | None = None,
    ) -> bytes:
        if not text.strip():
            return b""
        try:
            return asyncio.run(
                self._synthesize_speech_streaming_async(
                    text=text,
                    target_language_code=target_language_code,
                    speaker=speaker,
                    model=model,
                    chunk_callback=chunk_callback,
                )
            )
        except RuntimeError:
            return self.synthesize_speech(
                text=text,
                target_language_code=target_language_code,
                speaker=speaker,
                model=model,
                output_audio_codec="mp3",
            )

    async def _synthesize_speech_streaming_async(
        self,
        text: str,
        target_language_code: str,
        speaker: str,
        model: str,
        chunk_callback: Callable[[bytes, bool], None] | None,
    ) -> bytes:
        from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse

        client = AsyncSarvamAI(api_subscription_key=self.api_key)
        combined_audio = bytearray()

        async with client.text_to_speech_streaming.connect(
            model=model,
            send_completion_event=True,
        ) as ws:
            await ws.configure(
                target_language_code=target_language_code,
                speaker=speaker,
                min_buffer_size=20,
                max_chunk_length=120,
                output_audio_codec="mp3",
                output_audio_bitrate="128k",
            )
            for text_chunk in self._split_for_tts_streaming(text[:2500]):
                await ws.convert(text_chunk)
            await ws.flush()

            async for message in ws:
                if isinstance(message, AudioOutput):
                    audio_chunk = base64.b64decode(message.data.audio)
                    combined_audio.extend(audio_chunk)
                    if chunk_callback is not None:
                        chunk_callback(bytes(combined_audio), False)
                elif isinstance(message, EventResponse):
                    event_type = getattr(message.data, "event_type", None)
                    if event_type == "final":
                        break

        final_audio = bytes(combined_audio)
        if not final_audio:
            raise ValueError("Sarvam streaming TTS returned an empty audio payload.")
        if chunk_callback is not None and final_audio:
            chunk_callback(final_audio, True)
        return final_audio

    @staticmethod
    def _split_for_tts_streaming(text: str, max_chars: int = 260) -> list[str]:
        normalized = " ".join(text.split())
        if len(normalized) <= max_chars:
            return [normalized] if normalized else []

        sentences = re.split(r"(?<=[.!?।])\s+", normalized)
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
                current = sentence
            else:
                words = sentence.split()
                piece = ""
                for word in words:
                    next_piece = f"{piece} {word}".strip() if piece else word
                    if len(next_piece) <= max_chars:
                        piece = next_piece
                    else:
                        chunks.append(piece)
                        piece = word
                if piece:
                    current = piece
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _merge_tts_audio_chunks(encoded_chunks: Any) -> bytes:
        chunks = [str(chunk) for chunk in (encoded_chunks or []) if chunk]
        if not chunks:
            raise ValueError("Sarvam TTS returned an empty audio payload.")

        decoded_chunks = [base64.b64decode(chunk) for chunk in chunks]
        if not decoded_chunks:
            raise ValueError("Sarvam TTS returned an empty audio payload.")

        if len(decoded_chunks) == 1:
            return decoded_chunks[0]

        first_chunk = decoded_chunks[0]
        if len(first_chunk) >= 12 and first_chunk[:4] == b"RIFF" and first_chunk[8:12] == b"WAVE":
            combined_audio = bytearray(first_chunk)
            for chunk in decoded_chunks[1:]:
                data_pos = chunk.find(b"data")
                if data_pos != -1 and len(chunk) >= data_pos + 8:
                    combined_audio.extend(chunk[data_pos + 8 :])
                else:
                    combined_audio.extend(chunk)

            total_size = len(combined_audio) - 8
            combined_audio[4:8] = total_size.to_bytes(4, "little", signed=False)

            data_pos = combined_audio.find(b"data")
            if data_pos != -1 and len(combined_audio) >= data_pos + 8:
                data_size = len(combined_audio) - data_pos - 8
                combined_audio[data_pos + 4 : data_pos + 8] = data_size.to_bytes(4, "little", signed=False)
            return bytes(combined_audio)

        combined_audio = bytearray()
        for chunk in decoded_chunks:
            combined_audio.extend(chunk)
        return bytes(combined_audio)

    def extract_document_segments(
        self,
        file_path: Path,
        language_code: str,
        output_dir: Path,
    ) -> tuple[list[dict[str, Any]], str]:
        normalized_language_code = self._normalize_document_language(language_code)
        if file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)
            if total_pages > 10:
                return self._extract_large_pdf_segments(
                    file_path=file_path,
                    language_code=normalized_language_code,
                    output_dir=output_dir,
                    total_pages=total_pages,
                )
            text = self._extract_single_document_text(file_path, normalized_language_code, output_dir)
            return (
                [
                    {
                        "text": text,
                        "page_start": 1,
                        "page_end": total_pages,
                    }
                ],
                "sarvam-document-intelligence",
            )

        text = self._extract_single_document_text(file_path, normalized_language_code, output_dir)
        return ([{"text": text, "page_start": None, "page_end": None}], "sarvam-document-intelligence")

    def _extract_single_document_text(
        self,
        file_path: Path,
        language_code: str,
        output_dir: Path,
    ) -> str:
        job = self._get_client().document_intelligence.create_job(
            language=self._normalize_document_language(language_code),
            output_format="md",
        )
        job.upload_file(str(file_path))
        job.start()
        status = job.wait_until_complete()
        if status.job_state not in {"Completed", "PartiallyCompleted"}:
            raise RuntimeError(f"Document OCR failed with state {status.job_state}")

        output_path = output_dir / f"{file_path.stem}_{job.job_id}.zip"
        job.download_output(str(output_path))
        return self._read_ocr_zip(output_path)

    def _extract_large_pdf_segments(
        self,
        file_path: Path,
        language_code: str,
        output_dir: Path,
        total_pages: int,
    ) -> tuple[list[dict[str, Any]], str]:
        reader = PdfReader(str(file_path))
        segments: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            for start_page in range(0, total_pages, 10):
                end_page = min(start_page + 10, total_pages)
                split_path = temp_root / f"{file_path.stem}_pages_{start_page + 1}_{end_page}.pdf"

                writer = PdfWriter()
                for page_index in range(start_page, end_page):
                    writer.add_page(reader.pages[page_index])
                with split_path.open("wb") as handle:
                    writer.write(handle)

                text = self._extract_single_document_text(split_path, language_code, output_dir)
                segments.append(
                    {
                        "text": text,
                        "page_start": start_page + 1,
                        "page_end": end_page,
                    }
                )

        return segments, "sarvam-document-intelligence-batched"

    def _read_ocr_zip(self, zip_path: Path) -> str:
        parts: list[str] = []
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                lowered = member.lower()
                if lowered.endswith(".json"):
                    continue
                data = archive.read(member).decode("utf-8", errors="ignore")
                if lowered.endswith(".html"):
                    parts.append(self._clean_ocr_text(BeautifulSoup(data, "html.parser").get_text(separator=" ", strip=True)))
                else:
                    parts.append(self._clean_ocr_text(data))
        return "\n\n".join(part.strip() for part in parts if part.strip())

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        cleaned = text or ""
        cleaned = re.sub(r"!\[[^\]]*\]\(data:image/[^)]*\)", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\{[\s\r\n]*\"page_num\"[\s\S]*", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()

    @classmethod
    def _normalize_document_language(cls, language_code: str | None) -> str:
        normalized = (language_code or "").strip()
        if not normalized or normalized == "auto":
            return cls.DEFAULT_DOCUMENT_LANGUAGE
        return normalized

    @staticmethod
    def _extract_stream_text(event: dict[str, Any]) -> str:
        choices = event.get("choices") or []
        if not choices:
            return ""
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content
        message = choice.get("message") or {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        content = choice.get("content")
        return content if isinstance(content, str) else ""

    @staticmethod
    def _raise_stream_error(response: requests.Response) -> None:
        try:
            body = response.json()
        except ValueError:
            body = response.text
        raise RuntimeError(
            f"Streaming chat failed with status {response.status_code}: {body}"
        )

    @staticmethod
    def _split_for_translation(text: str, max_chars: int = 1800) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        words = text.split()
        chunks: list[str] = []
        current: list[str] = []
        current_length = 0
        for word in words:
            extra = len(word) + (1 if current else 0)
            if current and current_length + extra > max_chars:
                chunks.append(" ".join(current))
                current = [word]
                current_length = len(word)
            else:
                current.append(word)
                current_length += extra
        if current:
            chunks.append(" ".join(current))
        return chunks
