import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import ollama as _ollama
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


class OllamaCleanEmbeddings(Embeddings):
    """Calls Ollama /api/embed directly — no LLM sampling options attached.

    This avoids the 'invalid option: tfs_z' error that langchain_ollama's
    OllamaEmbeddings triggers on pure embedding models like nomic-embed-text.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self._client = _ollama.Client(host=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embed(model=self.model, input=texts)
        return [list(vector) for vector in response.embeddings]

    def embed_query(self, text: str) -> List[float]:
        response = self._client.embed(model=self.model, input=[text])
        return list(response.embeddings[0])


def load_documents(docs_dir: str):
    """Load documents with fallback encoding handling."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_path.resolve()}")

    docs = []
    # Try encodings in order: UTF-8, Windows-1252 (common on Windows), Latin-1, ASCII
    encodings_to_try = ["utf-8", "cp1252", "latin-1", "ascii"]
    
    txt_files = [name for name in os.listdir(docs_path) if name.endswith(".txt")]
    logger.info("Scanning %d .txt files in %s", len(txt_files), docs_path.resolve())

    for index, file in enumerate(txt_files, start=1):
        if file.endswith(".txt"):
            file_path = str(docs_path / file)
            successfully_loaded = False
            
            for encoding in encodings_to_try:
                try:
                    with open(file_path, encoding=encoding) as f:
                        text = f.read()
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file_path}
                    ))
                    successfully_loaded = True
                    logger.info("Loaded [%d/%d] %s with %s", index, len(txt_files), file, encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if not successfully_loaded:
                raise RuntimeError(f"Could not load {file} with any encoding: {encodings_to_try}")

    if not docs:
        raise ValueError(f"No .txt files found in {docs_path.resolve()}")
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


def build_faiss_batched(chunks: list[Document], embeddings: OllamaCleanEmbeddings) -> FAISS:
    """Embed batches in parallel threads, then build FAISS index from pre-computed vectors.

    Ollama server handles each /api/embed request independently, so parallel calls
    are safe. FAISS index construction is kept sequential (not thread-safe for writes).
    """
    if not chunks:
        raise ValueError("Cannot build vector index from empty chunk list")

    batch_size = int(os.environ.get("EMBED_BATCH_SIZE", "256"))
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
        return idx, vectors

    # --- parallel embedding phase ---
    # OLLAMA_NUM_PARALLEL=8
    # seems does not work for embeddings
    #
    results: dict[int, list[list[float]]] = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(embed_batch, i, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            idx, vectors = future.result()
            results[idx] = vectors

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

# 1. Load local documents
DOCS_DIR = os.environ.get("DOCS_DIR", "/docs")
docs = load_documents(DOCS_DIR)

# 2. Split into chunks
chunks = split_and_sanitize_documents(docs)
logger.info("Prepared %d chunks for embedding", len(chunks))

# 3. Create vector index (FAISS)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "granite3.3")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
# mxbai-embed-large
# nomic-embed-text

embeddings = OllamaCleanEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
vectorstore = build_faiss_batched(chunks, embeddings)

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