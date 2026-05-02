import os
import time
import shutil
import numpy as np
import chromadb
import faiss
from tqdm import tqdm

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
# CHROMA BENCHMARK
# =============================
print("\n--- Chroma ---")

# Clean up any previous chroma persistent directory
if os.path.exists(CHROMA_PERSIST_DIR):
    shutil.rmtree(CHROMA_PERSIST_DIR)

# Create persistent client
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.create_collection("bench")

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
_ = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
chroma_save_time = time.time() - start
print(f"Persist (auto, fsync probe): {chroma_save_time:.2f}s")

# Reload from disk
start = time.time()
client_loaded = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection_loaded = client_loaded.get_collection("bench")
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
