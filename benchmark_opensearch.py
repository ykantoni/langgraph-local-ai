"""OpenSearch k-NN helpers for benchmark.py and benchmark8.py."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import numpy as np

if TYPE_CHECKING:
    from opensearchpy import OpenSearch, helpers
else:
    try:
        from opensearchpy import OpenSearch, helpers

        OPENSEARCH_AVAILABLE = True
    except ImportError:
        OPENSEARCH_AVAILABLE = False
        OpenSearch = Any  # type: ignore[misc, assignment]
        helpers = Any  # type: ignore[misc, assignment]

_pf_proc: subprocess.Popen | None = None
_bulk_semaphore: threading.Semaphore | None = None


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


def resolve_config(
    *,
    default_index: str,
) -> dict[str, Any]:
    url = os.environ.get("OPENSEARCH_URL", "").strip()
    host = os.environ.get("OPENSEARCH_HOST", "172.19.73.182").strip()
    port = int(os.environ.get("OPENSEARCH_PORT", "9200"))
    use_ssl = _env_bool("OPENSEARCH_USE_SSL")
    user = os.environ.get("OPENSEARCH_USER", "").strip()
    password = os.environ.get("OPENSEARCH_PASSWORD", "").strip()

    if url:
        parsed = urlparse(url)
        host = parsed.hostname or host
        if parsed.port:
            port = parsed.port
        use_ssl = parsed.scheme == "https"
        if parsed.username:
            user = parsed.username
        if parsed.password:
            password = parsed.password

    return {
        "host": host,
        "port": port,
        "use_ssl": use_ssl,
        "user": user,
        "password": password,
        "index": os.environ.get("OPENSEARCH_INDEX", default_index),
        "timeout": int(os.environ.get("OPENSEARCH_TIMEOUT", "120")),
        "refresh_on_write": _env_bool("OPENSEARCH_REFRESH_ON_WRITE"),
    }


def start_port_forward() -> None:
    global _pf_proc
    local_port = int(os.environ.get("OPENSEARCH_LOCAL_PORT", "9200"))
    ns = os.environ.get("OPENSEARCH_PORT_FORWARD_NS", "opensearch-bench")
    svc = os.environ.get("OPENSEARCH_PORT_FORWARD_SVC", "opensearch-bench")
    _pf_proc = subprocess.Popen(
        ["kubectl", "port-forward", "-n", ns, f"svc/{svc}", f"{local_port}:9200"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    print(
        f"OpenSearch: kubectl port-forward svc/{svc} -> 127.0.0.1:{local_port} "
        f"(pid {_pf_proc.pid})"
    )


def stop_port_forward() -> None:
    global _pf_proc
    if _pf_proc is not None:
        _pf_proc.terminate()
        _pf_proc = None


def create_client(cfg: dict[str, Any]) -> OpenSearch:
    if _env_bool("OPENSEARCH_KUBECTL_PORT_FORWARD"):
        start_port_forward()
        cfg = {**cfg, "host": "127.0.0.1", "port": int(os.environ.get("OPENSEARCH_LOCAL_PORT", "9200"))}

    http_auth = None
    if cfg["user"] and cfg["password"]:
        http_auth = (cfg["user"], cfg["password"])

    deadline = time.time() + max(cfg["timeout"], 15)
    last_err: Exception | None = None
    while time.time() < deadline:
        if _pf_proc is not None and _pf_proc.poll() is not None:
            err = (_pf_proc.stderr.read() if _pf_proc.stderr else b"").decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(f"kubectl port-forward exited early: {err or _pf_proc.returncode}")
        try:
            client = OpenSearch(
                hosts=[{"host": cfg["host"], "port": cfg["port"]}],
                http_auth=http_auth,
                use_ssl=cfg["use_ssl"],
                verify_certs=False,
                ssl_show_warn=False,
                timeout=cfg["timeout"],
            )
            if not client.ping():
                raise ConnectionError("OpenSearch ping failed")
            return client
        except Exception as exc:
            last_err = exc
            time.sleep(0.5)

    scheme = "https" if cfg["use_ssl"] else "http"
    raise ConnectionError(
        f"Cannot reach OpenSearch at {scheme}://{cfg['host']}:{cfg['port']}"
    ) from last_err


def index_mapping(dim: int) -> dict[str, Any]:
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": int(os.environ.get("OPENSEARCH_SHARDS", "1")),
                "number_of_replicas": int(os.environ.get("OPENSEARCH_REPLICAS", "0")),
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": os.environ.get("OPENSEARCH_KNN_ENGINE", "lucene"),
                        "parameters": {
                            "m": int(os.environ.get("OPENSEARCH_HNSW_M", "16")),
                            "ef_construction": int(
                                os.environ.get("OPENSEARCH_HNSW_EF_CONSTRUCTION", "128")
                            ),
                        },
                    },
                },
            }
        },
    }


def reset_index(client: OpenSearch, index_name: str, dim: int) -> None:
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name, body=index_mapping(dim))


def bulk_ingest(
    client: OpenSearch,
    index_name: str,
    vectors: np.ndarray,
    start: int,
    end: int,
    batch_size: int,
    *,
    refresh: bool = False,
) -> None:
    for i in range(start, end, batch_size):
        batch_end = min(i + batch_size, end)
        actions = [
            {
                "_index": index_name,
                "_id": str(doc_id),
                "_source": {"id": doc_id, "embedding": vectors[doc_id].tolist()},
            }
            for doc_id in range(i, batch_end)
        ]
        helpers.bulk(
            client,
            actions,
            refresh=refresh,
            request_timeout=int(os.environ.get("OPENSEARCH_TIMEOUT", "120")),
        )


def prepare_parallel_ingest(parallel_limit: int | None = None) -> threading.Semaphore:
    global _bulk_semaphore
    limit = parallel_limit
    if limit is None:
        limit = max(1, int(os.environ.get("OPENSEARCH_INGEST_PARALLEL", "4")))
    _bulk_semaphore = threading.Semaphore(limit)
    return _bulk_semaphore


def parallel_bulk_ingest(
    client: OpenSearch,
    index_name: str,
    vectors: np.ndarray,
    start: int,
    end: int,
    batch_size: int,
    *,
    refresh: bool = False,
) -> None:
    assert _bulk_semaphore is not None
    with _bulk_semaphore:
        bulk_ingest(
            client,
            index_name,
            vectors,
            start,
            end,
            batch_size,
            refresh=refresh,
        )


def refresh_index(client: OpenSearch, index_name: str) -> None:
    client.indices.refresh(index=index_name)


def document_count(client: OpenSearch, index_name: str) -> int:
    return int(client.count(index=index_name)["count"])


def knn_search(
    client: OpenSearch,
    index_name: str,
    query_vector: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    body = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector.tolist(),
                    "k": top_k,
                }
            }
        },
    }
    return client.search(index=index_name, body=body)


def index_store_bytes(client: OpenSearch, index_name: str) -> tuple[int, str]:
    try:
        stats = client.indices.stats(index=index_name)
        primaries = stats.get("_all", {}).get("primaries", {}).get("store", {})
        size = int(primaries.get("size_in_bytes", 0))
        if size > 0:
            return size, "index store size (primaries)"
    except Exception:
        pass
    return 0, "unknown"


def target_label(cfg: dict[str, Any]) -> str:
    scheme = "https" if cfg["use_ssl"] else "http"
    return f"{scheme}://{cfg['host']}:{cfg['port']}/{cfg['index']}"
