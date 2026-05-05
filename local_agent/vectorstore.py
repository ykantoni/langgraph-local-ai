import logging
import os
from pathlib import Path
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from local_agent.config import Settings
from local_agent.ingest import collect_docs_state, docs_changed, load_documents, save_docs_state


logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings(Embeddings):
    _QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
    _RE_SPACES = re.compile(r"[ \t]+")
    _RE_MANY_BLANK_LINES = re.compile(r"\n{3,}")

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
        # Store HF downloads in a repo-local cache so startup doesn't re-fetch.
        # You can force offline usage by setting HF_HUB_OFFLINE=1.
        repo_root = Path(__file__).resolve().parents[1]
        cache_dir = repo_root / ".hf-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading SentenceTransformer '%s' on %s ...", model_name, self._device)
        self._model = SentenceTransformer(
            model_name,
            device=self._device,
            cache_folder=str(cache_dir),
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "1") == "1",
        )
        logger.info("SentenceTransformer model ready")

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Normalize text for stable embeddings (ASCII + UTF-8 safe).

        - Unicode NFKC normalization (compatibility forms)
        - Normalize newlines to \\n
        - Remove NUL bytes
        - Trim whitespace, collapse repeated spaces/tabs
        - Collapse excessive blank lines
        """
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Trim each line and collapse horizontal whitespace
        lines = []
        for line in text.split("\n"):
            line = cls._RE_SPACES.sub(" ", line).strip()
            lines.append(line)
        text = "\n".join(lines).strip()
        text = cls._RE_MANY_BLANK_LINES.sub("\n\n", text)
        return text

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [self.normalize_text(t) for t in texts]
        vectors = self._model.encode(
            texts,
            batch_size=self._encode_batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        text = self.normalize_text(text)
        vector = self._model.encode(
            [self._QUERY_PROMPT + text],
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return vector[0].tolist()


def split_and_sanitize_documents(docs: list[Document], max_chars: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    safe_chunks: list[Document] = []
    dropped = 0
    for chunk in chunks:
        text = SentenceTransformerEmbeddings.normalize_text(chunk.page_content)
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


def build_faiss_batched(
    chunks: list[Document],
    embeddings: SentenceTransformerEmbeddings,
    batch_size: int,
    num_threads: int,
) -> FAISS:
    if not chunks:
        raise ValueError("Cannot build vector index from empty chunk list")

    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    total_batches = len(batches)
    logger.info(
        "Embedding %d chunks in %d batches | batch_size=%d  threads=%d",
        len(chunks),
        total_batches,
        batch_size,
        num_threads,
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
            idx + 1,
            total_batches,
            len(texts),
            total_chars,
            avg_chars,
            min_chars,
            max_chars,
        )
        vectors = embeddings.embed_documents(texts)
        logger.info(
            "  Batch %d/%d finished | vectors=%d  total=%d chars embedded",
            idx + 1,
            total_batches,
            len(vectors),
            total_chars,
        )
        return idx, vectors, total_chars

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


def build_faiss_ivfpq_batched(
    chunks: list[Document],
    embeddings: SentenceTransformerEmbeddings,
    batch_size: int,
    num_threads: int,
    *,
    nlist: int,
    pq_m: int,
    pq_nbits: int,
    nprobe: int,
) -> FAISS:
    """Build a FAISS IVF+PQ index.

    Notes:
    - IVF+PQ must be trained before adding vectors.
    - We keep LangChain's FAISS wrapper (docstore + id mapping), but provide a custom FAISS index.
    """
    if not chunks:
        raise ValueError("Cannot build vector index from empty chunk list")

    # Reuse the existing batching implementation to produce the full embedding matrix.
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    total_batches = len(batches)
    logger.info(
        "Embedding %d chunks in %d batches | batch_size=%d  threads=%d",
        len(chunks),
        total_batches,
        batch_size,
        num_threads,
    )

    def embed_batch(idx: int, batch: list[Document]):
        texts = [doc.page_content for doc in batch]
        vectors = embeddings.embed_documents(texts)
        return idx, vectors

    results: dict[int, list[list[float]]] = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(embed_batch, i, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            idx, vectors = future.result()
            results[idx] = vectors

    all_texts = [doc.page_content for doc in chunks]
    all_metas = [doc.metadata for doc in chunks]
    all_vectors: list[list[float]] = []
    for i in range(total_batches):
        all_vectors.extend(results[i])

    faiss_vectors = np.asarray(all_vectors, dtype=np.float32)
    if faiss_vectors.ndim != 2:
        raise ValueError(f"Unexpected embedding shape: {faiss_vectors.shape}")

    dim = int(faiss_vectors.shape[1])
    if pq_m <= 0 or dim % pq_m != 0:
        raise ValueError(f"FAISS_PQ_M must divide embedding dim. dim={dim} pq_m={pq_m}")

    # Build IVF+PQ index
    from langchain_community.vectorstores.faiss import dependable_faiss_import
    from langchain_community.docstore.in_memory import InMemoryDocstore

    faiss = dependable_faiss_import()
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, int(nlist), int(pq_m), int(pq_nbits))

    logger.info(
        "Training FAISS IndexIVFPQ | vectors=%d dim=%d nlist=%d pq_m=%d pq_nbits=%d",
        faiss_vectors.shape[0],
        dim,
        nlist,
        pq_m,
        pq_nbits,
    )
    index.train(faiss_vectors)
    index.nprobe = int(nprobe)
    index.add(faiss_vectors)
    logger.info("FAISS IVF+PQ index ready (%d vectors indexed)", index.ntotal)

    # Build the LangChain wrapper (docstore + id mapping) the usual way, then swap in the
    # trained IVF+PQ index. The vectors are added in the same order, so the ID mapping remains
    # correct.
    tmp = FAISS.from_embeddings(
        text_embeddings=list(zip(all_texts, all_vectors)),
        embedding=embeddings,
        metadatas=all_metas,
    )
    tmp.index = index
    return tmp


def get_or_create_vectorstore(settings: Settings, embeddings: SentenceTransformerEmbeddings) -> FAISS:
    has_index_files = os.path.exists(settings.faiss_index_file) and os.path.exists(settings.faiss_meta_file)
    source_changed = docs_changed(settings.docs_dir, settings.faiss_state_file)

    if not settings.force_rebuild_index and has_index_files and not source_changed:
        logger.info("Loading FAISS index from disk: %s", settings.faiss_index_dir)
        vectorstore = FAISS.load_local(
            folder_path=settings.faiss_index_dir,
            embeddings=embeddings,
            index_name=settings.faiss_index_name,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded FAISS index successfully (docs unchanged)")
        return vectorstore

    if settings.force_rebuild_index:
        logger.info("FORCE_REBUILD_INDEX=1 set, rebuilding FAISS index")
    elif not has_index_files:
        logger.info("No persisted FAISS index found, creating a new one")
    elif source_changed:
        logger.info("Source documents changed, rebuilding FAISS index")
    else:
        logger.info("Rebuilding FAISS index")

    docs = load_documents(settings)
    chunks = split_and_sanitize_documents(docs, max_chars=settings.embed_max_chars)
    logger.info("Prepared %d chunks for embedding", len(chunks))

    vectorstore = build_faiss_ivfpq_batched(
        chunks,
        embeddings,
        batch_size=settings.embed_batch_size,
        num_threads=settings.embed_threads,
        nlist=settings.faiss_nlist,
        pq_m=settings.faiss_pq_m,
        pq_nbits=settings.faiss_pq_nbits,
        nprobe=settings.faiss_nprobe,
    )
    os.makedirs(settings.faiss_index_dir, exist_ok=True)
    vectorstore.save_local(folder_path=settings.faiss_index_dir, index_name=settings.faiss_index_name)
    save_docs_state(settings.faiss_state_file, collect_docs_state(settings.docs_dir))
    logger.info("Saved FAISS index to disk: %s", settings.faiss_index_dir)
    return vectorstore

