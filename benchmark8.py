"""Vector-store benchmarks with 8-thread parallel embedding ingest.

Same four backends as benchmark.py (FAISS, Qdrant, pgvector, Chroma).
Each backend is reset immediately before parallel ingest (clean index/table/collection).
Ingest uses INGEST_THREADS workers (default 8); query/save/load match benchmark.py.

Install deps:  pip install -r requirements.txt
"""

import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import chromadb
import faiss
import numpy as np
from tqdm import tqdm

import benchmark_storage as storage

try:
    import psycopg
    from pgvector.psycopg import register_vector

    _PGVECTOR_AVAILABLE = True
except ImportError:
    _PGVECTOR_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

# -----------------------------
# CONFIG
# $env:PGVECTOR_DSN="postgresql://postgres:postgres@172.19.73.182:32369/postgres"
# $env:QDRANT_URL="http://172.19.73.182:31189"
# $env:CHROMA_HOST="172.19.73.182"
# $env:CHROMA_PORT="32086"
# -----------------------------
N = 100_000
DIM = 384
NQ = 100
TOP_K = 5
INGEST_THREADS = int(os.environ.get("BENCHMARK8_THREADS", "8"))
BATCH_SIZE = 1000

FAISS_INDEX_PATH = "./artifacts/faiss_index_8.bin"
CHROMA_PERSIST_DIR = "./chroma_data_8"
CHROMA_HOST = os.environ.get("CHROMA_HOST", "").strip()
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "bench8")
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_data_8")
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "bench8")
PGVECTOR_TABLE = os.environ.get("PGVECTOR_TABLE", "bench_pgvector_8")
PGVECTOR_DSN = os.environ.get(
    "PGVECTOR_DSN",
    "postgresql://postgres:postgres@127.0.0.1:5432/postgres",
)

_qdrant_ingest_client: "QdrantClient | None" = None
_qdrant_upsert_semaphore: threading.Semaphore | None = None
_chroma_http_client: chromadb.ClientAPI | None = None
_chroma_add_lock = threading.Lock()
_chroma_pf_proc: subprocess.Popen | None = None


def _chunk_ranges(n_items: int, n_workers: int) -> list[tuple[int, int, int]]:
    """Return (worker_id, start, end) slices covering [0, n_items)."""
    chunk = (n_items + n_workers - 1) // n_workers
    ranges: list[tuple[int, int, int]] = []
    for worker_id in range(n_workers):
        start = worker_id * chunk
        end = min(start + chunk, n_items)
        if start < end:
            ranges.append((worker_id, start, end))
    return ranges


def _parallel_ingest(
    n_items: int,
    worker_fn,
    *,
    desc: str,
    n_workers: int = INGEST_THREADS,
) -> None:
    ranges = _chunk_ranges(n_items, n_workers)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(worker_fn, worker_id, start, end)
            for worker_id, start, end in ranges
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            future.result()


def _pgvector_open_connection() -> tuple[object, subprocess.Popen | None, str]:
    use_pf = os.environ.get("PGVECTOR_KUBECTL_PORT_FORWARD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    pf_proc: subprocess.Popen | None = None
    dsn = PGVECTOR_DSN

    if use_pf:
        local_port = int(os.environ.get("PGVECTOR_LOCAL_PORT", "5432"))
        ns = os.environ.get("PGVECTOR_PORT_FORWARD_NS", "pgvector-bench")
        svc = os.environ.get("PGVECTOR_PORT_FORWARD_SVC", "pgvector-bench")
        parsed = urlparse(PGVECTOR_DSN)
        user = parsed.username or "postgres"
        password = parsed.password or "postgres"
        dbname = (parsed.path or "/postgres").lstrip("/") or "postgres"
        dsn = f"postgresql://{user}:{password}@127.0.0.1:{local_port}/{dbname}"
        pf_proc = subprocess.Popen(
            ["kubectl", "port-forward", "-n", ns, f"svc/{svc}", f"{local_port}:5432"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        print(
            f"pgvector: kubectl port-forward svc/{svc} -> 127.0.0.1:{local_port} "
            f"(pid {pf_proc.pid})"
        )

    timeout = int(os.environ.get("PGVECTOR_CONNECT_TIMEOUT", "10"))
    deadline = time.time() + max(timeout, 15)
    last_err: Exception | None = None
    while time.time() < deadline:
        if pf_proc is not None and pf_proc.poll() is not None:
            err = (pf_proc.stderr.read() if pf_proc.stderr else b"").decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(f"kubectl port-forward exited early: {err or pf_proc.returncode}")
        try:
            conn = psycopg.connect(dsn, autocommit=False, connect_timeout=min(5, timeout))
            return conn, pf_proc, dsn
        except Exception as e:
            last_err = e
            time.sleep(0.5)

    if pf_proc is not None:
        pf_proc.terminate()
    assert last_err is not None
    raise last_err


def _chroma_use_remote() -> bool:
    return bool(
        os.environ.get("CHROMA_URL", "").strip()
        or os.environ.get("CHROMA_HOST", "").strip()
        or os.environ.get("CHROMA_KUBECTL_PORT_FORWARD", "").strip().lower()
        in ("1", "true", "yes")
    )


def _chroma_resolve_connection() -> tuple[str, int, bool]:
    """Resolve host, port, ssl from CHROMA_URL, CHROMA_HOST/PORT, or port-forward."""
    use_pf = os.environ.get("CHROMA_KUBECTL_PORT_FORWARD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if use_pf:
        host = "127.0.0.1"
        port = int(os.environ.get("CHROMA_LOCAL_PORT", "8000"))
        return host, port, False

    url = os.environ.get("CHROMA_URL", "").strip()
    if url:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 8000)
        ssl = parsed.scheme == "https"
        return host, port, ssl

    host = os.environ.get("CHROMA_HOST", CHROMA_HOST).strip()
    if host.startswith("http://") or host.startswith("https://"):
        parsed = urlparse(host)
        port = parsed.port or int(os.environ.get("CHROMA_PORT", str(CHROMA_PORT)))
        return parsed.hostname or "localhost", port, parsed.scheme == "https"

    port = int(os.environ.get("CHROMA_PORT", str(CHROMA_PORT)))
    return host, port, False


def _chroma_start_port_forward() -> None:
    global _chroma_pf_proc
    local_port = int(os.environ.get("CHROMA_LOCAL_PORT", "8000"))
    ns = os.environ.get("CHROMA_PORT_FORWARD_NS", "chroma-bench")
    svc = os.environ.get("CHROMA_PORT_FORWARD_SVC", "chroma-bench")
    _chroma_pf_proc = subprocess.Popen(
        ["kubectl", "port-forward", "-n", ns, f"svc/{svc}", f"{local_port}:8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    print(
        f"Chroma: kubectl port-forward svc/{svc} -> 127.0.0.1:{local_port} "
        f"(pid {_chroma_pf_proc.pid})"
    )


def _chroma_http_client_connect() -> chromadb.ClientAPI:
    """HttpClient with heartbeat retries; logs resolved URL on failure."""
    host, port, ssl = _chroma_resolve_connection()
    scheme = "https" if ssl else "http"
    target = f"{scheme}://{host}:{port}"

    client = chromadb.HttpClient(host=host, port=port, ssl=ssl)
    timeout = int(os.environ.get("CHROMA_CONNECT_TIMEOUT", "15"))
    deadline = time.time() + max(timeout, 15)
    last_err: Exception | None = None
    while time.time() < deadline:
        if _chroma_pf_proc is not None and _chroma_pf_proc.poll() is not None:
            err = (_chroma_pf_proc.stderr.read() if _chroma_pf_proc.stderr else b"").decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(f"kubectl port-forward exited early: {err or _chroma_pf_proc.returncode}")
        try:
            client.heartbeat()
            return client
        except Exception as exc:
            last_err = exc
            time.sleep(0.5)

    hint = (
        f"Cannot reach Chroma at {target}. "
        "Use NodePort from kubectl get svc (not 8000 unless port-forwarded). "
        "Example: $env:CHROMA_HOST='<NODE_IP>'; $env:CHROMA_PORT='32086'. "
        "Or: $env:CHROMA_KUBECTL_PORT_FORWARD='1' with port-forward to 127.0.0.1:8000."
    )
    raise ConnectionError(hint) from last_err


def _chroma_client():
    global _chroma_http_client
    if _chroma_use_remote():
        if _chroma_http_client is None:
            if os.environ.get("CHROMA_KUBECTL_PORT_FORWARD", "").strip().lower() in (
                "1",
                "true",
                "yes",
            ):
                _chroma_start_port_forward()
            _chroma_http_client = _chroma_http_client_connect()
        return _chroma_http_client
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def _faiss_reset() -> list:
    """Remove on-disk index and return fresh in-memory shards."""
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    return [faiss.IndexFlatL2(DIM) for _ in range(INGEST_THREADS)]


def _chroma_reset_before_ingest():
    """Wipe local persist dir or remote collection before benchmark ingest."""
    if not CHROMA_HOST and os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
    client = _chroma_client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    client.create_collection(CHROMA_COLLECTION)
    return client


def _pgvector_reset_before_ingest(conn) -> None:
    """Drop benchmark table (and indexes) and recreate empty schema."""
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {PGVECTOR_TABLE} CASCADE")
        cur.execute(
            f"CREATE TABLE {PGVECTOR_TABLE} ("
            "id int PRIMARY KEY, "
            f"embedding vector({DIM})"
            ")"
        )
    conn.commit()


def _qdrant_ingest_parallel_limit() -> int:
    """Max concurrent upserts to Qdrant (HTTP/NodePort is sensitive on Windows)."""
    if QDRANT_URL:
        return max(1, int(os.environ.get("QDRANT_INGEST_PARALLEL", "2")))
    return INGEST_THREADS


def _qdrant_client() -> "QdrantClient":
    if QDRANT_URL:
        return QdrantClient(
            url=QDRANT_URL,
            timeout=int(os.environ.get("QDRANT_TIMEOUT", "120")),
            force_disable_check_same_thread=True,
            pool_size=int(os.environ.get("QDRANT_POOL_SIZE", "4")),
        )
    return QdrantClient(path=QDRANT_PATH)


def _qdrant_prepare_parallel_ingest() -> None:
    global _qdrant_ingest_client, _qdrant_upsert_semaphore
    _qdrant_ingest_client = _qdrant_client()
    limit = _qdrant_ingest_parallel_limit()
    _qdrant_upsert_semaphore = threading.Semaphore(limit)


def _qdrant_reset_before_ingest() -> "QdrantClient":
    """Wipe local storage dir or remote collection before benchmark ingest."""
    if not QDRANT_URL and os.path.exists(QDRANT_PATH):
        shutil.rmtree(QDRANT_PATH)
    client = _qdrant_client()
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.EUCLID),
    )
    return client


# -----------------------------
# GENERATE DATA
# -----------------------------
np.random.seed(42)
xb = np.random.random((N, DIM)).astype("float32")
xq = np.random.random((NQ, DIM)).astype("float32")

print(f"Dataset: {N} vectors, dim={DIM}")
print(f"Parallel ingest: {INGEST_THREADS} threads, batch_size={BATCH_SIZE}")

# =============================
# FAISS BENCHMARK
# =============================
print("\n--- FAISS (8-thread ingest) ---")

print(f"Resetting FAISS ({FAISS_INDEX_PATH})...")
faiss_shards = _faiss_reset()


def _faiss_ingest_worker(worker_id: int, start: int, end: int) -> None:
    faiss_shards[worker_id].add(xb[start:end])


start = time.time()
_parallel_ingest(N, _faiss_ingest_worker, desc="faiss ingest")
index = faiss.IndexFlatL2(DIM)
for shard in faiss_shards:
    index.merge_from(shard)
faiss_index_time = time.time() - start
print(f"Indexing time: {faiss_index_time:.2f}s")

start = time.time()
faiss.write_index(index, FAISS_INDEX_PATH)
faiss_save_time = time.time() - start
faiss_storage_bytes = storage.file_size(FAISS_INDEX_PATH)
print(f"Save time: {faiss_save_time:.2f}s")
print(f"Saved index file: {FAISS_INDEX_PATH}")
storage.print_storage_size("Database size", faiss_storage_bytes, "index file on disk")

start = time.time()
index_loaded = faiss.read_index(FAISS_INDEX_PATH)
faiss_load_time = time.time() - start
print(f"Load time: {faiss_load_time:.2f}s")

start = time.time()
for q in xq:
    index_loaded.search(q.reshape(1, -1), TOP_K)
faiss_query_time = time.time() - start
faiss_latency = faiss_query_time / NQ
print(f"Avg query latency: {faiss_latency * 1000:.2f} ms")
print(f"QPS: {NQ / faiss_query_time:.2f}")

# =============================
# QDRANT BENCHMARK
# =============================
print("\n--- Qdrant (8-thread ingest) ---")

qdrant_index_time = None
qdrant_save_time = None
qdrant_load_time = None
qdrant_query_time = None
qdrant_latency = None
qdrant_storage_bytes = None
qdrant_skipped = False

if not _QDRANT_AVAILABLE:
    print("Skipped: install qdrant-client (pip install -r requirements.txt)")
    qdrant_skipped = True
else:
    try:
        print(f"Resetting Qdrant (collection={QDRANT_COLLECTION})...")
        _qdrant_reset_before_ingest()
        _qdrant_prepare_parallel_ingest()
        assert _qdrant_ingest_client is not None
        assert _qdrant_upsert_semaphore is not None
        if QDRANT_URL:
            print(
                f"Qdrant remote ingest: {INGEST_THREADS} workers, "
                f"{_qdrant_ingest_parallel_limit()} concurrent upserts "
                f"(override: QDRANT_INGEST_PARALLEL)"
            )

        def _qdrant_ingest_worker(_worker_id: int, start: int, end: int) -> None:
            for i in range(start, end, BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, end)
                points = [
                    PointStruct(id=j, vector=xb[j].tolist())
                    for j in range(i, batch_end)
                ]
                with _qdrant_upsert_semaphore:
                    _qdrant_ingest_client.upsert(
                        collection_name=QDRANT_COLLECTION, points=points
                    )

        start = time.time()
        _parallel_ingest(N, _qdrant_ingest_worker, desc="qdrant upsert")
        qdrant_index_time = time.time() - start
        print(f"Indexing time: {qdrant_index_time:.2f}s")

        start = time.time()
        _ = _qdrant_client()
        qdrant_save_time = time.time() - start
        mode = QDRANT_URL or f"path={QDRANT_PATH}"
        print(f"Persist / reopen ({mode}): {qdrant_save_time:.2f}s")

        start = time.time()
        client_loaded = _qdrant_client()
        row_count = client_loaded.count(
            collection_name=QDRANT_COLLECTION, exact=True
        ).count
        qdrant_load_time = time.time() - start
        print(f"Load time: {qdrant_load_time:.2f}s ({row_count:,} points)")

        start = time.time()
        for q in xq:
            client_loaded.query_points(
                collection_name=QDRANT_COLLECTION,
                query=q.tolist(),
                limit=TOP_K,
            )
        qdrant_query_time = time.time() - start
        qdrant_latency = qdrant_query_time / NQ
        print(f"Avg query latency: {qdrant_latency * 1000:.2f} ms")
        print(f"QPS: {NQ / qdrant_query_time:.2f}")

        qdrant_storage_bytes, qdrant_method = storage.qdrant_storage_bytes(
            client_loaded,
            QDRANT_COLLECTION,
            local_path=QDRANT_PATH if not QDRANT_URL else "",
            n_points=row_count,
            dim=DIM,
        )
        storage.print_storage_size("Database size", qdrant_storage_bytes, qdrant_method)

    except Exception as e:
        print(f"Skipped: {e}")
        if QDRANT_URL:
            print(f"  URL: {QDRANT_URL}")
        else:
            print(f"  path: {QDRANT_PATH}")
        qdrant_skipped = True

# =============================
# PGVECTOR BENCHMARK
# =============================
print("\n--- pgvector (PostgreSQL, 8-thread ingest) ---")

pg_index_time = None
pg_save_time = None
pg_load_time = None
pg_query_time = None
pg_latency = None
pg_storage_bytes = None
pg_skipped = False

if not _PGVECTOR_AVAILABLE:
    print("Skipped: install psycopg and pgvector (pip install -r requirements.txt)")
    pg_skipped = True
else:
    pf_proc: subprocess.Popen | None = None
    pg_dsn = PGVECTOR_DSN
    try:
        conn, pf_proc, pg_dsn = _pgvector_open_connection()

        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)

        print(f"Resetting pgvector (table={PGVECTOR_TABLE})...")
        _pgvector_reset_before_ingest(conn)

        pg_connect_timeout = int(os.environ.get("PGVECTOR_CONNECT_TIMEOUT", "10"))

        def _pgvector_ingest_worker(_worker_id: int, start: int, end: int) -> None:
            worker_conn = psycopg.connect(
                pg_dsn,
                autocommit=False,
                connect_timeout=pg_connect_timeout,
            )
            register_vector(worker_conn)
            try:
                with worker_conn.cursor() as cur:
                    for i in range(start, end, BATCH_SIZE):
                        batch_end = min(i + BATCH_SIZE, end)
                        rows = [(j, xb[j]) for j in range(i, batch_end)]
                        cur.executemany(
                            f"INSERT INTO {PGVECTOR_TABLE} (id, embedding) "
                            "VALUES (%s, %s)",
                            rows,
                        )
                worker_conn.commit()
            finally:
                worker_conn.close()

        start = time.time()
        _parallel_ingest(N, _pgvector_ingest_worker, desc="pgvector insert")
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE INDEX {PGVECTOR_TABLE}_hnsw "
                f"ON {PGVECTOR_TABLE} USING hnsw (embedding vector_l2_ops)"
            )
        conn.commit()
        pg_index_time = time.time() - start
        print(f"Indexing time (parallel insert + HNSW): {pg_index_time:.2f}s")

        start = time.time()
        conn.commit()
        pg_save_time = time.time() - start
        print(f"Commit / persist: {pg_save_time:.4f}s")

        conn.close()
        start = time.time()
        conn_loaded = psycopg.connect(
            pg_dsn,
            autocommit=False,
            connect_timeout=pg_connect_timeout,
        )
        with conn_loaded.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn_loaded.commit()
        register_vector(conn_loaded)
        with conn_loaded.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {PGVECTOR_TABLE}")
            row_count = cur.fetchone()[0]
        pg_load_time = time.time() - start
        print(f"Load time (reconnect): {pg_load_time:.2f}s ({row_count:,} rows)")

        start = time.time()
        with conn_loaded.cursor() as cur:
            for q in xq:
                cur.execute(
                    f"SELECT id FROM {PGVECTOR_TABLE} "
                    "ORDER BY embedding <-> %s LIMIT %s",
                    (q, TOP_K),
                )
                cur.fetchall()
        pg_query_time = time.time() - start

        pg_storage_bytes = storage.pgvector_table_bytes(conn_loaded, PGVECTOR_TABLE)
        storage.print_storage_size(
            "Database size", pg_storage_bytes, "pg_total_relation_size (table + indexes)"
        )
        conn_loaded.close()

        pg_latency = pg_query_time / NQ
        print(f"Avg query latency: {pg_latency * 1000:.2f} ms")
        print(f"QPS: {NQ / pg_query_time:.2f}")

    except Exception as e:
        print(f"Skipped: {e}")
        print(f"  DSN: {pg_dsn}")
        if "timeout" in str(e).lower():
            print(
                "  Hint: use node IP + NodePort from kubectl get nodes/svc, or "
                "set PGVECTOR_KUBECTL_PORT_FORWARD=1 for port-forward to 127.0.0.1."
            )
        pg_skipped = True
    finally:
        if pf_proc is not None:
            pf_proc.terminate()

# =============================
# CHROMA BENCHMARK
# =============================
print("\n--- Chroma (8-thread ingest) ---")

chroma_index_time = None
chroma_save_time = None
chroma_load_time = None
chroma_query_time = None
chroma_latency = None
chroma_storage_bytes = None
chroma_skipped = False

try:
    print(f"Resetting Chroma (collection={CHROMA_COLLECTION})...")
    chroma_client = _chroma_reset_before_ingest()
    if _chroma_use_remote():
        host, port, _ = _chroma_resolve_connection()
        print(f"Chroma remote: http://{host}:{port} ({INGEST_THREADS} workers, shared client)")

    collection = chroma_client.get_collection(CHROMA_COLLECTION)

    def _chroma_ingest_worker(_worker_id: int, start: int, end: int) -> None:
        for i in range(start, end, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, end)
            rows = (
                xb[i:batch_end].tolist(),
                [str(j) for j in range(i, batch_end)],
            )
            if _chroma_use_remote():
                with _chroma_add_lock:
                    collection.add(embeddings=rows[0], ids=rows[1])
            else:
                collection.add(embeddings=rows[0], ids=rows[1])

    start = time.time()
    _parallel_ingest(N, _chroma_ingest_worker, desc="chroma add")
    chroma_index_time = time.time() - start
    print(f"Indexing time: {chroma_index_time:.2f}s")

    start = time.time()
    _ = _chroma_client()
    chroma_save_time = time.time() - start
    if _chroma_use_remote():
        h, p, _ = _chroma_resolve_connection()
        mode = f"http://{h}:{p}"
    else:
        mode = f"path={CHROMA_PERSIST_DIR}"
    print(f"Persist / reopen ({mode}): {chroma_save_time:.2f}s")

    start = time.time()
    client_loaded = _chroma_client()
    collection_loaded = client_loaded.get_collection(CHROMA_COLLECTION)
    chroma_load_time = time.time() - start
    print(f"Load time: {chroma_load_time:.2f}s")

    start = time.time()
    for q in xq:
        collection_loaded.query(query_embeddings=[q.tolist()], n_results=TOP_K)
    chroma_query_time = time.time() - start
    chroma_latency = chroma_query_time / NQ
    print(f"Avg query latency: {chroma_latency * 1000:.2f} ms")
    print(f"QPS: {NQ / chroma_query_time:.2f}")

    chroma_storage_bytes, chroma_method = storage.chroma_storage_bytes(
        collection_loaded,
        persist_dir=CHROMA_PERSIST_DIR,
        remote=_chroma_use_remote(),
        dim=DIM,
    )
    storage.print_storage_size("Database size", chroma_storage_bytes, chroma_method)

except Exception as e:
    print(f"Skipped: {e}")
    if _chroma_use_remote():
        try:
            h, p, _ = _chroma_resolve_connection()
            print(f"  target: http://{h}:{p}")
        except Exception:
            pass
    chroma_skipped = True
finally:
    if _chroma_pf_proc is not None:
        _chroma_pf_proc.terminate()

# =============================
# SUMMARY
# =============================
print("\n=== SUMMARY (benchmark8, parallel ingest) ===")
print(f"FAISS ({INGEST_THREADS} threads)")
print(
    f"  → index: {faiss_index_time:.2f}s | save: {faiss_save_time:.2f}s "
    f"| load: {faiss_load_time:.2f}s"
)
print(f"  → query latency: {faiss_latency * 1000:.2f} ms | QPS: {NQ / faiss_query_time:.2f}")
print(f"  → size: {storage.format_bytes(faiss_storage_bytes)}")
if not chroma_skipped and chroma_index_time is not None:
    print(f"\nChroma ({INGEST_THREADS} threads)")
    print(
        f"  → index: {chroma_index_time:.2f}s | persist: {chroma_save_time:.2f}s "
        f"| load: {chroma_load_time:.2f}s"
    )
    print(f"  → query latency: {chroma_latency * 1000:.2f} ms | QPS: {NQ / chroma_query_time:.2f}")
    print(f"  → size: {storage.format_bytes(chroma_storage_bytes)}")
else:
    print("\nChroma: skipped (see section above)")
if not pg_skipped and pg_index_time is not None:
    print(f"\npgvector (HNSW, L2, {INGEST_THREADS} threads)")
    print(
        f"  → index: {pg_index_time:.2f}s | commit: {pg_save_time:.4f}s "
        f"| load: {pg_load_time:.2f}s"
    )
    print(f"  → query latency: {pg_latency * 1000:.2f} ms | QPS: {NQ / pg_query_time:.2f}")
    print(f"  → size: {storage.format_bytes(pg_storage_bytes)}")
else:
    print("\npgvector: skipped (see section above)")
if not qdrant_skipped and qdrant_index_time is not None:
    print(f"\nQdrant (HNSW default, L2, {INGEST_THREADS} threads)")
    print(
        f"  → index: {qdrant_index_time:.2f}s | persist: {qdrant_save_time:.2f}s "
        f"| load: {qdrant_load_time:.2f}s"
    )
    print(
        f"  → query latency: {qdrant_latency * 1000:.2f} ms "
        f"| QPS: {NQ / qdrant_query_time:.2f}"
    )
    print(f"  → size: {storage.format_bytes(qdrant_storage_bytes)}")
else:
    print("\nQdrant: skipped (see section above)")
