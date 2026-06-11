"""sentence-transformers getting started examples.

Prerequisites:
    pip install -r requirements.txt

Run from repo root (avoids shadowing Hugging Face ``transformers`` package):
    python gettingstarted/sentence_transformers_examples.py
    python -m gettingstarted.sentence_transformers_examples

Run one example:
    python gettingstarted/sentence_transformers_examples.py --example encode

Models:
    --model sentence-transformers/all-MiniLM-L6-v2          (default, symmetric)
    --model sentence-transformers/static-retrieval-mrl-en-v1  (asymmetric; used in this repo)

Note: do not name this file ``transformers.py`` — that breaks
``import transformers`` inside sentence-transformers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim

# Same default retrieval model as local_agent/config.py
DEFAULT_SYMMETRIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RETRIEVAL_MODEL = "sentence-transformers/static-retrieval-mrl-en-v1"
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


def _load_model(model_name: str, device: str | None = None) -> SentenceTransformer:
    cache = Path(__file__).resolve().parents[1] / ".hf-cache"
    cache.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"cache_folder": str(cache)}
    if device:
        kwargs["device"] = device
    print(f"Loading {model_name} ...")
    return SentenceTransformer(model_name, **kwargs)


def example_encode(model: SentenceTransformer) -> None:
    """Encode sentences into fixed-size embedding vectors."""
    print("\n=== encode ===")
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    embeddings = model.encode(sentences)
    print(f"sentences: {len(sentences)}")
    print(f"embeddings.shape: {embeddings.shape}")
    print(f"first vector (first 8 dims): {embeddings[0][:8]}")


def example_similarity(model: SentenceTransformer) -> None:
    """Cosine similarity between all pairs of sentence embeddings."""
    print("\n=== similarity ===")
    sentences = [
        "A man is eating pasta.",
        "A man is eating food.",
        "A man is riding a horse.",
        "A woman is eating pasta.",
    ]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    scores = model.similarity(embeddings, embeddings)
    print("Pairwise similarity matrix:")
    print(scores)
    print("Most similar unrelated pair should score lower than paraphrases.")


def example_pairwise(model: SentenceTransformer) -> None:
    """Compare one query sentence to several candidates."""
    print("\n=== pairwise ===")
    query = "How do I reset my password?"
    candidates = [
        "To reset your password, open Settings and choose Security.",
        "The weather forecast calls for rain tomorrow.",
        "Click Forgot password on the login page.",
    ]
    query_emb = model.encode(query, convert_to_tensor=True)
    cand_emb = model.encode(candidates, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, cand_emb)[0]
    ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
    print(f"query: {query!r}")
    for idx, score in ranked:
        print(f"  {score:.4f}  {candidates[idx]!r}")


def example_semantic_search(model: SentenceTransformer) -> None:
    """Semantic search over a small in-memory corpus (symmetric model)."""
    print("\n=== semantic_search ===")
    corpus = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks learn patterns from labeled data.",
        "The capital of France is Paris.",
        "Transformers use self-attention for sequence modeling.",
        "Bread recipes often start with flour, water, and yeast.",
    ]
    queries = [
        "How do neural nets learn?",
        "What is Paris known for?",
    ]
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    for query in queries:
        query_embedding = model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]
        print(f"\nquery: {query!r}")
        for hit in hits:
            doc = corpus[hit["corpus_id"]]
            print(f"  {hit['score']:.4f}  {doc!r}")


def example_asymmetric_search(model: SentenceTransformer) -> None:
    """Asymmetric retrieval: query prompt + encode_query / encode_document.

    Use models trained for retrieval (e.g. static-retrieval-mrl-en-v1), matching
    local_agent/vectorstore.py query vs document encoding.
    """
    print("\n=== asymmetric_search ===")
    corpus = [
        "LangChain is a framework for building LLM applications.",
        "FAISS is a library for efficient similarity search.",
        "PostgreSQL supports the pgvector extension for embeddings.",
        "Chroma provides an embedded or hosted vector database.",
    ]
    queries = [
        "Which database supports vector columns?",
        "framework for language model apps",
    ]
    corpus_embeddings = model.encode_document(corpus, convert_to_tensor=True)
    for query in queries:
        prefixed = QUERY_PROMPT + query
        query_embedding = model.encode_query(prefixed, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=2)[0]
        print(f"\nquery: {query!r}")
        for hit in hits:
            doc = corpus[hit["corpus_id"]]
            print(f"  {hit['score']:.4f}  {doc!r}")


def example_normalize_embeddings(model: SentenceTransformer) -> None:
    """L2-normalized embeddings (common for cosine similarity / ANN indexes)."""
    print("\n=== normalize_embeddings ===")
    sentences = ["vector database benchmark", "embedding ingest throughput"]
    raw = model.encode(sentences, normalize_embeddings=False)
    normed = model.encode(sentences, normalize_embeddings=True)
    import numpy as np

    raw_norms = np.linalg.norm(raw, axis=1)
    normed_norms = np.linalg.norm(normed, axis=1)
    print(f"raw L2 norms:     {raw_norms}")
    print(f"normalized norms: {normed_norms}")


EXAMPLES = {
    "encode": example_encode,
    "similarity": example_similarity,
    "pairwise": example_pairwise,
    "semantic_search": example_semantic_search,
    "asymmetric_search": example_asymmetric_search,
    "normalize": example_normalize_embeddings,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example",
        choices=[*EXAMPLES.keys(), "all"],
        default="all",
        help="Which example to run (default: all)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_SYMMETRIC_MODEL,
        help=f"Hugging Face model id (default: {DEFAULT_SYMMETRIC_MODEL})",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, cuda, or mps (default: library auto-select)",
    )
    args = parser.parse_args()

    if args.example == "asymmetric_search" and args.model == DEFAULT_SYMMETRIC_MODEL:
        args.model = DEFAULT_RETRIEVAL_MODEL

    model = _load_model(args.model, device=args.device)

    if args.example == "all":
        for name, fn in EXAMPLES.items():
            if name == "asymmetric_search":
                m = _load_model(DEFAULT_RETRIEVAL_MODEL, device=args.device)
                fn(m)
            else:
                fn(model)
    else:
        EXAMPLES[args.example](model)


def main2() -> None:
    model = _load_model("all-mpnet-base-v2", device="cuda")
#    model = _load_model(DEFAULT_RETRIEVAL_MODEL, device="cuda")

    e1 = model.encode(
        "How do I reset my password?"
    )

    print(len(e1))
    print(e1[:10])
    e2 = model.encode("I forgot my password")
    print(len(e2))
    print(e2[:10])
    score = cos_sim(e1, e2)

    print(score)

    e3 = model.encode("I love you baby")
    print(len(e3))
    print(e3[:10])
    score = cos_sim(e1, e3)    
    print(score)

    sentences = [
        "How do I reset my password?",
        "I forgot my password",
        "Click Forgot password on the login page.",
        "Check an email for a reset password link.",
        "Pass me a sword please.",
        "I forgot my sword.",
    ]

    # 2. Calculate embeddings by calling model.encode()
    embeddings = model.encode(sentences)
    print(embeddings.shape)
    print(embeddings[:10])
    similarities = model.similarity(embeddings, embeddings)
    print(similarities)

    print("e1:", "How do I reset my password?")
    print("sentences:")
    for sentence in sentences:
        print(f"  {sentence}")

    similarities = model.similarity(e1, embeddings)
    print("e1 vs sentences:")
    for sentence, score in zip(sentences, similarities[0].tolist()):
        print(f"  {score:.4f}  {sentence}")


if __name__ == "__main__":
    main2()
