import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    log_level: str
    ollama_base_url: str
    ollama_chat_model: str
    st_embed_model: str
    st_encode_batch: int
    st_device: str
    docs_dir: str
    faiss_index_dir: str
    faiss_index_name: str
    faiss_nlist: int
    faiss_pq_m: int
    faiss_pq_nbits: int
    faiss_nprobe: int
    embed_max_chars: int
    embed_batch_size: int
    embed_threads: int
    pdf_max_pages: int
    pdf_extractor: str
    pdf_cache_dir: str
    force_rebuild_index: bool

    @property
    def faiss_index_file(self) -> str:
        return os.path.join(self.faiss_index_dir, f"{self.faiss_index_name}.faiss")

    @property
    def faiss_meta_file(self) -> str:
        return os.path.join(self.faiss_index_dir, f"{self.faiss_index_name}.pkl")

    @property
    def faiss_state_file(self) -> str:
        return os.path.join(self.faiss_index_dir, f"{self.faiss_index_name}.source_state.json")


def load_settings() -> Settings:
    return Settings(
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_chat_model=os.environ.get("OLLAMA_CHAT_MODEL", "granite3.3"),
        st_embed_model=os.environ.get("ST_EMBED_MODEL", "sentence-transformers/static-retrieval-mrl-en-v1"),
        st_encode_batch=int(os.environ.get("ST_ENCODE_BATCH", "32")),
        st_device=os.environ.get("ST_DEVICE", "cpu"),
        docs_dir=os.environ.get("DOCS_DIR", "/docs"),
        faiss_index_dir=os.environ.get("FAISS_INDEX_DIR", "./faiss_store"),
        faiss_index_name=os.environ.get("FAISS_INDEX_NAME", "index"),
        faiss_nlist=int(os.environ.get("FAISS_NLIST", "256")),
        faiss_pq_m=int(os.environ.get("FAISS_PQ_M", "64")),
        faiss_pq_nbits=int(os.environ.get("FAISS_PQ_NBITS", "8")),
        faiss_nprobe=int(os.environ.get("FAISS_NPROBE", "16")),
        embed_max_chars=int(os.environ.get("EMBED_MAX_CHARS", "1800")),
        embed_batch_size=int(os.environ.get("EMBED_BATCH_SIZE", "32")),
        embed_threads=int(os.environ.get("EMBED_THREADS", "8")),
        pdf_max_pages=int(os.environ.get("PDF_MAX_PAGES", "0")),
        pdf_extractor=os.environ.get("PDF_EXTRACTOR", "auto").lower(),
        pdf_cache_dir=os.environ.get("PDF_CACHE_DIR", "./.pdf_text_cache"),
        force_rebuild_index=os.environ.get("FORCE_REBUILD_INDEX", "0") == "1",
    )

