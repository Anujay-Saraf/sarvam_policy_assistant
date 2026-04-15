# Sarvam Policy Assistant

This project is a Streamlit app for:

- multilingual policy Q&A over a persistent local vector store
- OCR ingestion from the UI using Sarvam Document Intelligence
- chat in Indian languages with Sarvam chat models
- text and voice interaction using Sarvam speech APIs
- storing OCR output in the same searchable knowledge base used by chat
- capturing response feedback in a structured local DB for reuse and continuous answer improvement

## What is included

- `app.py`: main Streamlit application
- `src/services/sarvam_service.py`: Sarvam SDK wrapper for chat, translation, OCR, STT, and TTS
- `src/services/document_store.py`: persistent Chroma-backed vector store
- `src/services/feedback_store.py`: structured SQLite feedback memory with similarity-based response reuse
- `src/services/ingestion_service.py`: upload, OCR/local parsing, chunking, and indexing pipeline
- `data/`: local storage for uploaded files, OCR artifacts, audio, and vector data
- `Admin` view inside the app: governance metrics, guardrails, token estimates, and heuristic security health checks

## Architecture

1. User uploads policy files from the Streamlit UI.
2. Scanned PDFs and images go through Sarvam Document Intelligence OCR when possible.
3. Extracted text is chunked and each chunk is translated to English for retrieval using Sarvam Translate.
4. English search chunks are embedded with `BAAI/bge-large-en-v1.5` when available, with a hashing fallback if the model cannot be loaded.
5. Each chunk is saved with metadata such as document id, filename, page range, ingestion time, and text statistics.
6. User asks questions through text or voice input.
7. Before generating a fresh answer, the app checks the local feedback DB for a highly similar previously liked response and can reuse it instantly.
8. If no approved cache hit is found, the app translates the user query to English for retrieval, reranks results, and uses the top 5 matches.
9. The answer is shown as text and can also be synthesized to speech.
10. Every answer is saved in a structured SQLite table with metadata, embeddings, sources, and user feedback state.
11. The UI asks for like/dislike on every assistant response; dislikes can collect expectation text and trigger an immediate improved re-run.

## Setup

### 1. Create and activate a virtual environment

```powershell
cd c:\Users\vmadmin\Downloads\sarvam_policy_assistant
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment

```powershell
Copy-Item .env.example .env
```

Then set your Sarvam key in `.env`:

```env
SARVAM_API_KEY=your_key_here
```

Optional embedding settings:

```env
EMBEDDING_BACKEND=auto
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

Optional admin access control:

```env
ADMIN_ACCESS_CODE=your_admin_code_here
```

### 4. Run the app

```powershell
streamlit run app.py
```

## Usage notes

- Paste the Sarvam API key in the sidebar or keep it in `.env`.
- Use the `Ingestion` tab to upload PDFs, images, or text files.
- Enable OCR for scanned policy documents. Large PDFs are automatically split into OCR-safe batches.
- Every ingested document is prepared with an English retrieval index so mixed-language documents work better with English embeddings.
- Streamlit navigation now separates the app into `/user` and `/admin` views.
- In `/user`, the main screen is only the chat workspace, while the left sidebar is reserved for document upload and available-document selection.
- In `/admin`, ingestion, library management, governance details, and model/runtime controls are separated from the user chat experience.
- `Admin` gives the business owner a governance console for response-level confidence, relevance, token estimates, and health checks.
- Chat session memory is stored in Streamlit `session_state`, so 4-5 follow-up text/audio turns can continue in the same conversational thread.
- Use `Upload audio (Recommended)` for the most reliable voice-query flow.
- Even if microphone/audio-input streaming is unreliable on a VM, assistant output can still be played from the built-in speaker player for the reply language.
- Audio reply output format now supports both `MP3` and `WAV`, with `MP3` recommended for best browser compatibility.
- Enable `Stream chat responses` for token-like text streaming and `Stream audio reply (Beta)` for progressive audio updates.
- Streaming audio uses sentence-sized TTS chunks and a custom browser audio player for lower perceived latency.
- The app retrieves and shows the top 5 grounded matches with page-aware metadata.
- Every assistant answer includes a like/dislike feedback loop.
- Confidence score and relevance score are shown in chat for each assistant response.
- Liked responses are stored in `data/feedback/feedback.db` and can be reused for future similar queries to reduce latency.
- Disliked responses ask the user what they expected, save that expectation in the feedback DB, and trigger an improved answer in the same chat flow.
- Similar historical feedback is also injected as lightweight preference signals when generating new answers.
- Short follow-up queries such as `aur`, `uske liye`, or `what about this` are treated as conversational continuations, so the app prefers session context over one-shot cache reuse.
- Assistant replies expose a `Play response` path so Hindi, Hinglish-inferred Hindi, and other supported Indian-language replies can be spoken back through the browser audio player.
- Generated replies can also be downloaded in the selected `MP3` or `WAV` format.
- Admin health checks surface heuristic green/yellow flags for outbound endpoints, dangerous code primitives, masked secret entry, and guardrail enablement.
- If the selected audio language is unsupported by TTS, the app falls back to English audio.
- Streaming audio is delivered as progressive MP3 updates in the current UI. Depending on the browser, playback may restart from the latest chunk as the audio element refreshes.

## Important implementation note

On the first run, `sentence-transformers` may download the `BAAI/bge-large-en-v1.5` model. If that is unavailable, the app still works using a hashing fallback, but retrieval quality will be lower than the transformer-backed path.
