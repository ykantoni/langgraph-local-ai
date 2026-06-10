"""On-disk / database size helpers for benchmark.py and benchmark8.py."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GiB ({num_bytes:,} bytes)"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.2f} MiB ({num_bytes:,} bytes)"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KiB ({num_bytes:,} bytes)"
    return f"{num_bytes:,} bytes"


def dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def file_size(path: str) -> int:
    return os.path.getsize(path) if os.path.isfile(path) else 0


def estimate_raw_vector_bytes(n_vectors: int, dim: int) -> int:
    return n_vectors * dim * 4


def pgvector_table_bytes(conn: Any, table: str) -> int:
    """Total PostgreSQL disk for table + indexes + TOAST (pg_total_relation_size)."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT pg_total_relation_size('{table}'::regclass)")
        row = cur.fetchone()
    return int(row[0]) if row else 0


def qdrant_storage_bytes(
    client: QdrantClient | None,
    collection_name: str,
    *,
    local_path: str = "",
    n_points: int = 0,
    dim: int = 384,
) -> tuple[int, str]:
    if local_path and os.path.isdir(local_path):
        return dir_size(local_path), "storage directory"
    if client is not None:
        try:
            count = client.count(collection_name=collection_name, exact=True).count
            return estimate_raw_vector_bytes(count, dim), "estimate (remote; float32 vectors only)"
        except Exception:
            pass
    return estimate_raw_vector_bytes(n_points, dim), "estimate (float32 vectors only)"


def chroma_storage_bytes(
    collection: Any,
    *,
    persist_dir: str,
    remote: bool,
    dim: int = 384,
) -> tuple[int, str]:
    if not remote and persist_dir and os.path.isdir(persist_dir):
        return dir_size(persist_dir), "persist directory"
    try:
        count = collection.count()
    except Exception:
        count = 0
    return estimate_raw_vector_bytes(count, dim), "estimate (remote; float32 vectors only)"


def print_storage_size(label: str, num_bytes: int, method: str) -> None:
    print(f"{label}: {format_bytes(num_bytes)} [{method}]")
