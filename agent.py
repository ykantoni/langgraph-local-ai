import os
import logging
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pypdf
import docx as _docx
try:
    import fitz  # PyMuPDF (much faster text extraction for many large PDFs)
except Exception:
    fitz = None
from sentence_transformers import SentenceTransformer
from langchain_classic.agents import AgentType, initialize_agent
from langchain_community.vectorstores import FAISS 
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter


logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# https://huggingface.co/blog/static-embeddings
class SentenceTransformerEmbeddings(Embeddings):
    """Local embedding via sentence-transformers — in-process, no HTTP, GPU-aware.

    For retrieval-optimised models (e.g. mxbai-embed-large-v1) a query prefix is
    prepended automatically on embed_query() while documents are embedded as-is.
    """

    # Used by mxbai-embed-large-v1 and compatible models; ignored for others.
    _QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        normalize: bool = True,
        encode_batch_size: int = 32,
    ) -> None:
        import torch

        self._model_name = model_name
        self._normalize = normalize
        self._encode_batch_size = encode_batch_size

        requested_device = (device or "auto").lower()
        if requested_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "ST_DEVICE=cuda requested, but CUDA is unavailable in this PyTorch build. Falling back to CPU."
            )
            resolved_device = "cpu"
        else:
            resolved_device = requested_device

        self._device = resolved_device
        logger.info("Loading SentenceTransformer '%s' on %s ...", model_name, self._device)
        self._model = SentenceTransformer(model_name, device=self._device)
        logger.info("SentenceTransformer model ready")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._encode_batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        prompted = self._QUERY_PROMPT + text
        vector = self._model.encode(
            [prompted],
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return vector[0].tolist()


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def _load_txt(file_path: str) -> str:
    """Load a plain-text file with multi-encoding fallback."""
    encodings_to_try = ["utf-8", "cp1252", "latin-1", "ascii"]
    for encoding in encodings_to_try:
        try:
            with open(file_path, encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"Could not decode {file_path} with any of {encodings_to_try}")


def _load_pdf(file_path: str) -> str:
    """Extract text from a PDF with fast backend + lightweight cache."""
    path = Path(file_path)
    max_pages = int(os.environ.get("PDF_MAX_PAGES", "0"))
    extractor = os.environ.get("PDF_EXTRACTOR", "auto").lower()
    cache_dir = Path(os.environ.get("PDF_CACHE_DIR", "./.pdf_text_cache"))

    stat = path.stat()
    cache_key_raw = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{max_pages}|{extractor}"
    cache_key = hashlib.sha256(cache_key_raw.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{cache_key}.txt"

    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    # Prefer PyMuPDF when present; fallback to pypdf.
    if extractor in {"auto", "pymupdf"} and fitz is not None:
        with fitz.open(file_path) as doc:
            pages: list[str] = []
            page_count = len(doc)
            limit = page_count if max_pages <= 0 else min(max_pages, page_count)
            for i in range(limit):
                pages.append(doc[i].get_text("text") or "")
            text = "\n".join(pages)
    else:
        reader = pypdf.PdfReader(file_path)
        pages: list[str] = []
        page_count = len(reader.pages)
        limit = page_count if max_pages <= 0 else min(max_pages, page_count)
        for i in range(limit):
            pages.append(reader.pages[i].extract_text() or "")
        text = "\n".join(pages)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text, encoding="utf-8")
    return text


def _load_docx(file_path: str) -> str:
    """Extract paragraph text from a DOCX file."""
    doc = _docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)


def load_documents(docs_dir: str):
    """Load .txt, .pdf, and .docx documents with fallback encoding for text files."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_path.resolve()}")

    source_files = sorted([
        p for p in docs_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    logger.info(
        "Scanning %d supported files (.txt/.pdf/.docx) in %s",
        len(source_files), docs_path.resolve(),
    )

    docs = []
    for index, path in enumerate(source_files, start=1):
        file_path = str(path)
        ext = path.suffix.lower()
        try:
            if ext == ".txt":
                text = _load_txt(file_path)
            elif ext == ".pdf":
                text = _load_pdf(file_path)
            elif ext == ".docx":
                text = _load_docx(file_path)
            else:
                continue  # should not happen given the filter above

            docs.append(Document(
                page_content=text,
                metadata={"source": file_path, "type": ext.lstrip(".")},
            ))
            logger.info("Loaded [%d/%d] %s (%s)", index, len(source_files), path.name, ext)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {file_path}: {exc}") from exc

    if not docs:
        raise ValueError(
            f"No supported files ({', '.join(SUPPORTED_EXTENSIONS)}) found in {docs_path.resolve()}"
        )
    logger.info("Loaded %d documents successfully", len(docs))
    return docs


def split_and_sanitize_documents(docs: list[Document]) -> list[Document]:
    """Split docs and enforce an upper bound to avoid Ollama context overflows."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Hard cap is kept intentionally conservative for embedding models.
    max_chars = int(os.environ.get("EMBED_MAX_CHARS", "1800"))
    safe_chunks: list[Document] = []
    dropped = 0
    for chunk in chunks:
        text = chunk.page_content.strip()
        if not text:
            dropped += 1
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        safe_chunks.append(Document(page_content=text, metadata=chunk.metadata))

    if dropped:
        logger.warning("Dropped %d empty chunks before embedding", dropped)
    if not safe_chunks:
        raise ValueError("No valid chunks after splitting/sanitization")
    return safe_chunks


def build_faiss_batched(chunks: list[Document], embeddings: SentenceTransformerEmbeddings) -> FAISS:
    """Embed batches in parallel threads, then build FAISS index from pre-computed vectors.

    Ollama server handles each /api/embed request independently, so parallel calls
    are safe. FAISS index construction is kept sequential (not thread-safe for writes).
    """
    if not chunks:
        raise ValueError("Cannot build vector index from empty chunk list")

    batch_size = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
    num_threads = int(os.environ.get("EMBED_THREADS", "8"))

    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    total_batches = len(batches)
    logger.info(
        "Embedding %d chunks in %d batches | batch_size=%d  threads=%d",
        len(chunks), total_batches, batch_size, num_threads,
    )

    def embed_batch(idx: int, batch: list[Document]):
        texts = [doc.page_content for doc in batch]
        sizes = [len(t) for t in texts]
        total_chars = sum(sizes)
        avg_chars = total_chars // len(sizes) if sizes else 0
        min_chars = min(sizes) if sizes else 0
        max_chars = max(sizes) if sizes else 0
        logger.info(
            "  Batch %d/%d started  | chunks=%d  total=%d chars  avg=%d  min=%d  max=%d",
            idx + 1, total_batches, len(texts), total_chars, avg_chars, min_chars, max_chars,
        )
        vectors = embeddings.embed_documents(texts)
        logger.info(
            "  Batch %d/%d finished | vectors=%d  total=%d chars embedded",
            idx + 1, total_batches, len(vectors), total_chars,
        )
        return idx, vectors, total_chars

    # --- parallel embedding phase ---
    # OLLAMA_NUM_PARALLEL=8
    # seems does not work for embeddings
    #
    results: dict[int, list[list[float]]] = {}
    completed_batches = 0
    loaded_embeddings = 0
    loaded_chars = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(embed_batch, i, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            idx, vectors, total_chars = future.result()
            results[idx] = vectors
            completed_batches += 1
            loaded_embeddings += len(vectors)
            loaded_chars += total_chars

            if completed_batches % 100 == 0:
                logger.info(
                    "Progress: completed %d/%d batches | embeddings loaded=%d | chars embedded=%d",
                    completed_batches,
                    total_batches,
                    loaded_embeddings,
                    loaded_chars,
                )

    # --- sequential FAISS build phase (not thread-safe) ---
    all_texts = [doc.page_content for doc in chunks]
    all_metas = [doc.metadata for doc in chunks]
    all_vectors: list[list[float]] = []
    for i in range(total_batches):
        all_vectors.extend(results[i])

    logger.info("Building FAISS index from %d vectors ...", len(all_vectors))
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(all_texts, all_vectors)),
        embedding=embeddings,
        metadatas=all_metas,
    )
    logger.info("Vector index build complete  (%d vectors indexed)", len(all_vectors))
    return vectorstore


def collect_docs_state(docs_dir: str) -> dict:
    """Collect lightweight source state using file size + mtime_ns for all supported files."""
    docs_path = Path(docs_dir)
    source_files = sorted([
        p for p in docs_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    files: dict[str, dict[str, int]] = {}
    for path in source_files:
        stat = path.stat()
        files[str(path.resolve())] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    return {
        "docs_dir": str(docs_path.resolve()),
        "file_count": len(files),
        "files": files,
    }


def load_saved_docs_state(state_file: str) -> dict | None:
    if not os.path.exists(state_file):
        return None
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("Failed to read docs state file, forcing rebuild: %s", state_file)
        return None


def save_docs_state(state_file: str, state: dict) -> None:
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def docs_changed(docs_dir: str, state_file: str) -> bool:
    """Return True when current docs metadata differs from the saved metadata."""
    current = collect_docs_state(docs_dir)
    saved = load_saved_docs_state(state_file)
    return saved != current


def get_or_create_vectorstore(embeddings: SentenceTransformerEmbeddings) -> FAISS:
    """Load FAISS index from disk when available, else build and persist it."""
    force_rebuild = os.environ.get("FORCE_REBUILD_INDEX", "0") == "1"

    has_index_files = os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_META_FILE)
    source_changed = docs_changed(DOCS_DIR, FAISS_STATE_FILE)

    if not force_rebuild and has_index_files and not source_changed:
        logger.info("Loading FAISS index from disk: %s", FAISS_INDEX_DIR)
        vectorstore = FAISS.load_local(
            folder_path=FAISS_INDEX_DIR,
            embeddings=embeddings,
            index_name=FAISS_INDEX_NAME,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded FAISS index successfully (docs unchanged)")
        return vectorstore

    if force_rebuild:
        logger.info("FORCE_REBUILD_INDEX=1 set, rebuilding FAISS index")
    elif not has_index_files:
        logger.info("No persisted FAISS index found, creating a new one")
    elif source_changed:
        logger.info("Source documents changed, rebuilding FAISS index")
    else:
        logger.info("Rebuilding FAISS index")

    docs = load_documents(DOCS_DIR)
    chunks = split_and_sanitize_documents(docs)
    logger.info("Prepared %d chunks for embedding", len(chunks))

    vectorstore = build_faiss_batched(chunks, embeddings)
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(folder_path=FAISS_INDEX_DIR, index_name=FAISS_INDEX_NAME)
    save_docs_state(FAISS_STATE_FILE, collect_docs_state(DOCS_DIR))
    logger.info("Saved FAISS index to disk: %s", FAISS_INDEX_DIR)
    return vectorstore

# 1. Vector/Model configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "granite3.3")
# sentence-transformers model (downloaded from HuggingFace on first run)
# mxbai-embed-large-v1  → 335M params, 1024-dim, strong retrieval quality
# nomic-ai/nomic-embed-text-v1  → 137M params, 768-dim, lighter alternative
ST_EMBED_MODEL = os.environ.get("ST_EMBED_MODEL", "sentence-transformers/static-retrieval-mrl-en-v1")
ST_ENCODE_BATCH = int(os.environ.get("ST_ENCODE_BATCH", "32"))
ST_DEVICE = os.environ.get("ST_DEVICE", "cpu")
DOCS_DIR = os.environ.get("DOCS_DIR", "/docs")
FAISS_INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", "./faiss_store")
FAISS_INDEX_NAME = os.environ.get("FAISS_INDEX_NAME", "index")
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, f"{FAISS_INDEX_NAME}.faiss")
FAISS_META_FILE = os.path.join(FAISS_INDEX_DIR, f"{FAISS_INDEX_NAME}.pkl")
FAISS_STATE_FILE = os.path.join(FAISS_INDEX_DIR, f"{FAISS_INDEX_NAME}.source_state.json")

embeddings = SentenceTransformerEmbeddings(
    model_name=ST_EMBED_MODEL,
    encode_batch_size=ST_ENCODE_BATCH,
    device=ST_DEVICE,
)
vectorstore = get_or_create_vectorstore(embeddings)

# 4. Create a search function
def search_docs(query: str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([r.page_content for r in results])


# 5. Wrap as a tool
search_tool = Tool(
    name="LocalDocumentSearch",
    func=search_docs,
    description="Searches local documents for relevant information",
)

# 6. LLM
llm = ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)

# 7. Agent (decides when to use tool)
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# 8. Ask questions
while True:
    try:
        query = input("Ask: ")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break
    response = agent.run(query)
    print("\nAnswer:", response)