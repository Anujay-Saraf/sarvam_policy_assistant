from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    data_dir: Path
    uploads_dir: Path
    ocr_dir: Path
    audio_dir: Path
    feedback_dir: Path
    feedback_db_path: Path
    structured_dir: Path
    structured_db_path: Path
    model_cache_dir: Path
    vector_store_dir: Path
    collection_name: str
    sarvam_api_key: str
    admin_access_code: str
    chunk_size: int
    chunk_overlap: int
    embedding_backend: str
    embedding_model: str
    embedding_dimensions: int


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    root_dir = Path(__file__).resolve().parents[2]
    load_dotenv(root_dir / ".env")

    data_dir = root_dir / "data"
    uploads_dir = data_dir / "uploads"
    ocr_dir = data_dir / "ocr"
    audio_dir = data_dir / "audio"
    feedback_dir = data_dir / "feedback"
    feedback_db_path = feedback_dir / "feedback.db"
    structured_dir = data_dir / "structured"
    structured_db_path = structured_dir / "structured_demo.db"
    model_cache_dir = data_dir / "models"
    vector_store_dir = data_dir / "vector_store"

    for path in (data_dir, uploads_dir, ocr_dir, audio_dir, feedback_dir, structured_dir, model_cache_dir, vector_store_dir):
        path.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        root_dir=root_dir,
        data_dir=data_dir,
        uploads_dir=uploads_dir,
        ocr_dir=ocr_dir,
        audio_dir=audio_dir,
        feedback_dir=feedback_dir,
        feedback_db_path=feedback_db_path,
        structured_dir=structured_dir,
        structured_db_path=structured_db_path,
        model_cache_dir=model_cache_dir,
        vector_store_dir=vector_store_dir,
        collection_name=os.getenv("VECTOR_COLLECTION", "policy_knowledge_base"),
        sarvam_api_key=os.getenv("SARVAM_API_KEY", ""),
        admin_access_code=os.getenv("ADMIN_ACCESS_CODE", ""),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "auto"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "BAAI/bge-large-en-v1.5",
        ),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
    )
