"""Vector-store benchmarks (FAISS, Chroma, pgvector, Qdrant).

Install deps:  pip install -r requirements.txt
"""

import os
import subprocess
import time
import shutil
from urllib.parse import urlparse

import numpy as np
import chromadb
import faiss
from tqdm import tqdm

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
# -----------------------------
N = 100_000        # number of vectors
DIM = 384          # embedding size
NQ = 100           # number of queries
TOP_K = 5

# Persistence paths

FAISS_INDEX_PATH = "./artifacts/faiss_index.bin"
CHROMA_PERSIST_DIR = "./chroma_data"
CHROMA_HOST = os.environ.get("CHROMA_HOST", "").strip()
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "bench")
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_data")
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "bench")
PGVECTOR_TABLE = "bench_pgvector"
# postgresql://user:pass@host:port/dbname  (requires CREATE EXTENSION vector)
PGVECTOR_DSN = os.environ.get(
    "PGVECTOR_DSN",
    "postgresql://postgres:postgres@127.0.0.1:5432/postgres",
)


def _pgvector_open_connection() -> tuple[object, subprocess.Popen | None, str]:
    """Connect to Postgres for the pgvector benchmark.

  Use ``PGVECTOR_DSN`` with the node INTERNAL-IP and NodePort from
  ``kubectl get svc pgvector-bench -n pgvector-bench``. Or set
  ``PGVECTOR_KUBECTL_PORT_FORWARD=1`` for ``kubectl port-forward``.
    """
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


def _chroma_client():
    if CHROMA_HOST:
        return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def _chroma_reset_collection(client) -> None:
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    client.create_collection(CHROMA_COLLECTION)


def _qdrant_client() -> "QdrantClient":
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=QDRANT_PATH)


def _qdrant_reset_collection(client: "QdrantClient") -> None:
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.EUCLID),
    )


# -----------------------------
# GENERATE DATA
# -----------------------------
np.random.seed(42)
xb = np.random.random((N, DIM)).astype('float32')
xq = np.random.random((NQ, DIM)).astype('float32')

print(f"Dataset: {N} vectors, dim={DIM}")

# =============================
# FAISS BENCHMARK
# =============================
print("\n--- FAISS ---")

start = time.time()
index = faiss.IndexFlatL2(DIM)  # exact search
index.add(xb)
faiss_index_time = time.time() - start

print(f"Indexing time: {faiss_index_time:.2f}s")

# Save index to disk
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
if os.path.exists(FAISS_INDEX_PATH):
    os.remove(FAISS_INDEX_PATH)

start = time.time()
faiss.write_index(index, FAISS_INDEX_PATH)
faiss_save_time = time.time() - start
faiss_file_size = os.path.getsize(FAISS_INDEX_PATH)
print(f"Save time: {faiss_save_time:.2f}s")
print(f"Saved index file: {FAISS_INDEX_PATH} ({faiss_file_size:,} bytes)")

# Load index from disk
start = time.time()
index_loaded = faiss.read_index(FAISS_INDEX_PATH)
faiss_load_time = time.time() - start
print(f"Load time: {faiss_load_time:.2f}s")

# Query using loaded index
start = time.time()
for q in xq:
    index_loaded.search(q.reshape(1, -1), TOP_K)
faiss_query_time = time.time() - start

faiss_latency = faiss_query_time / NQ

print(f"Avg query latency: {faiss_latency*1000:.2f} ms")
print(f"QPS: {NQ / faiss_query_time:.2f}")

# =============================
# QDRANT BENCHMARK
# =============================
# Local disk: default QDRANT_PATH=./qdrant_data
# Kubernetes: kubectl apply -f deploy/qdrant/qdrant.yaml
#   set QDRANT_URL=http://<NODE_IP>:<NODE_PORT>   # from kubectl get svc -n qdrant-bench
# Or:         set QDRANT_URL=http://127.0.0.1:6333  (kubectl port-forward ... 6333:6333)
print("\n--- Qdrant ---")

qdrant_index_time = None
qdrant_save_time = None
qdrant_load_time = None
qdrant_query_time = None
qdrant_latency = None
qdrant_skipped = False

if not _QDRANT_AVAILABLE:
    print("Skipped: install qdrant-client (pip install -r requirements.txt)")
    qdrant_skipped = True
else:
    try:
        if not QDRANT_URL and os.path.exists(QDRANT_PATH):
            shutil.rmtree(QDRANT_PATH)

        client = _qdrant_client()
        _qdrant_reset_collection(client)

        start = time.time()
        batch_size = 1000
        for i in tqdm(range(0, N, batch_size), desc="qdrant upsert"):
            end = min(i + batch_size, N)
            points = [
                PointStruct(id=j, vector=xb[j].tolist())
                for j in range(i, end)
            ]
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
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
# Requires Postgres with the pgvector extension, e.g.:
#   kubectl apply -f deploy/pgvector/postgres.yaml
#   kubectl get nodes -o wide && kubectl get svc pgvector-bench -n pgvector-bench
# NodePort from host: postgresql://postgres:postgres@<NODE_IP>:<NODE_PORT>/postgres
# Or: PGVECTOR_KUBECTL_PORT_FORWARD=1 with PGVECTOR_DSN (uses kubectl port-forward to 127.0.0.1)
print("\n--- pgvector (PostgreSQL) ---")

pg_index_time = None
pg_save_time = None
pg_load_time = None
pg_query_time = None
pg_latency = None
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

        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {PGVECTOR_TABLE}")
            cur.execute(
                f"CREATE TABLE {PGVECTOR_TABLE} ("
                "id int PRIMARY KEY, "
                f"embedding vector({DIM})"
                ")"
            )
        conn.commit()

        start = time.time()
        batch_size = 1000
        with conn.cursor() as cur:
            for i in tqdm(range(0, N, batch_size), desc="pgvector insert"):
                end = min(i + batch_size, N)
                rows = [(j, xb[j]) for j in range(i, end)]
                cur.executemany(
                    f"INSERT INTO {PGVECTOR_TABLE} (id, embedding) VALUES (%s, %s)",
                    rows,
                )
            cur.execute(
                f"CREATE INDEX {PGVECTOR_TABLE}_hnsw "
                f"ON {PGVECTOR_TABLE} USING hnsw (embedding vector_l2_ops)"
            )
        conn.commit()
        pg_index_time = time.time() - start
        print(f"Indexing time (insert + HNSW): {pg_index_time:.2f}s")

        start = time.time()
        conn.commit()
        pg_save_time = time.time() - start
        print(f"Commit / persist: {pg_save_time:.4f}s")

        conn.close()
        start = time.time()
        conn_loaded = psycopg.connect(
            pg_dsn,
            autocommit=False,
            connect_timeout=int(os.environ.get("PGVECTOR_CONNECT_TIMEOUT", "10")),
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
# Local: default ./chroma_data (PersistentClient)
# Kubernetes: kubectl apply -f deploy/chroma/chroma.yaml
#   set CHROMA_HOST=<NODE_IP>  set CHROMA_PORT=<NODE_PORT>
print("\n--- Chroma ---")

if not CHROMA_HOST and os.path.exists(CHROMA_PERSIST_DIR):
    shutil.rmtree(CHROMA_PERSIST_DIR)

client = _chroma_client()
_chroma_reset_collection(client)
collection = client.get_collection(CHROMA_COLLECTION)

# indexing
start = time.time()
batch_size = 1000
for i in tqdm(range(0, N, batch_size)):
    collection.add(
        embeddings=xb[i:i+batch_size].tolist(),
        ids=[str(j) for j in range(i, i+batch_size)]
    )
chroma_index_time = time.time() - start

print(f"Indexing time: {chroma_index_time:.2f}s")

# PersistentClient auto-persists on every write — measure the overhead of
# an explicit fsync by timing a no-op round-trip to the same directory.
start = time.time()
_ = _chroma_client()
chroma_save_time = time.time() - start
mode = f"{CHROMA_HOST}:{CHROMA_PORT}" if CHROMA_HOST else f"path={CHROMA_PERSIST_DIR}"
print(f"Persist / reopen ({mode}): {chroma_save_time:.2f}s")

start = time.time()
client_loaded = _chroma_client()
collection_loaded = client_loaded.get_collection(CHROMA_COLLECTION)
chroma_load_time = time.time() - start
print(f"Load time: {chroma_load_time:.2f}s")

# query using loaded collection
start = time.time()
for q in xq:
    collection_loaded.query(query_embeddings=[q.tolist()], n_results=TOP_K)
chroma_query_time = time.time() - start

chroma_latency = chroma_query_time / NQ

print(f"Avg query latency: {chroma_latency*1000:.2f} ms")
print(f"QPS: {NQ / chroma_query_time:.2f}")



# =============================
# SUMMARY
# =============================
print("\n=== SUMMARY ===")
print(f"FAISS")
print(f"  → index: {faiss_index_time:.2f}s | save: {faiss_save_time:.2f}s | load: {faiss_load_time:.2f}s")
print(f"  → query latency: {faiss_latency*1000:.2f} ms | QPS: {NQ / faiss_query_time:.2f}")
print(f"\nChroma")
print(f"  → index: {chroma_index_time:.2f}s | persist: {chroma_save_time:.2f}s | load: {chroma_load_time:.2f}s")
print(f"  → query latency: {chroma_latency*1000:.2f} ms | QPS: {NQ / chroma_query_time:.2f}")
if not pg_skipped and pg_index_time is not None:
    print(f"\npgvector (HNSW, L2)")
    print(f"  → index: {pg_index_time:.2f}s | commit: {pg_save_time:.4f}s | load: {pg_load_time:.2f}s")
    print(f"  → query latency: {pg_latency * 1000:.2f} ms | QPS: {NQ / pg_query_time:.2f}")
else:
    print("\npgvector: skipped (see section above)")
if not qdrant_skipped and qdrant_index_time is not None:
    print(f"\nQdrant (HNSW default, L2)")
    print(
        f"  → index: {qdrant_index_time:.2f}s | persist: {qdrant_save_time:.2f}s "
        f"| load: {qdrant_load_time:.2f}s"
    )
    print(
        f"  → query latency: {qdrant_latency * 1000:.2f} ms "
        f"| QPS: {NQ / qdrant_query_time:.2f}"
    )
else:
    print("\nQdrant: skipped (see section above)")
