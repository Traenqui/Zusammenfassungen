#import "../../template_zusammenf.typ": *

= Retrieval-Augmented Generation (RAG)

RAG = combine *retrieval* (external knowledge) with *generation* (LLM) to reduce hallucinations and answer with up-to-date / domain-specific info.

== Core Idea
Instead of relying only on parametric memory (model weights), retrieve relevant documents/chunks and condition the LLM on them.

*High-level flow:*
Query -> Embed -> Retrieve -> (Rerank) -> Augment prompt with context -> Generate answer

== Pipeline (Step-by-step)

1. *Ingest documents*
   - Clean text, split into chunks, store metadata (source, section, timestamp, permissions)

2. *Chunking*
   - Split into passages of length $L$ tokens (often with overlap)
   - Smaller chunks: higher precision, less context per chunk
   - Larger chunks: more context, but higher risk of irrelevant content

3. *Embeddings*
   - Map chunk text to a dense vector $e in RR^d$
   - Store in a vector database (with metadata)

4. *Retrieval*
   - Embed user query to $q in RR^d$
   - Find nearest neighbors in embedding space

5. *Context construction*
   - Select top-$k$ chunks + optionally compress/summarize
   - Inject into prompt (system/user template) with citations/metadata

6. *Answer generation*
   - LLM generates conditioned on retrieved context
   - Optionally: answer + citations + refusal if not supported

== Vector Search (Similarity)
Common similarity functions:
- *Cosine similarity*: $"sim"(q, e) = frac(q dot e, norm(q) norm(e))$
- *Dot product*: $"sim"(q, e) = q dot e$

Nearest-neighbor search often uses approximate indexes (ANN) for speed:
- FAISS, Annoy, HNSW-style indexes

== Retrieval Evaluation

Let:
- $"Rel"@K$ = number of relevant chunks in top-$K$
- $"Rel"_"all"$ = total relevant chunks available in dataset (for that query)

$"Precision@K"$: $P@K = frac("Rel"@K, K) quad "Recall@K"$:$R@K = frac("Rel"@K, "Rel"_"all") $

#hinweis[
Exam intuition: increasing $K$ usually increases recall, but can reduce precision and create longer prompts (more noise).
]

== Reranking
Two-stage retrieval improves quality:

1. *Retrieve many* candidates with fast dense retrieval (e.g., top 50–200)
2. *Rerank* with a stronger (slower) model:
   - Cross-encoder scores pairs (query, chunk)
   - Output is a reordered list; keep top-$n$ for the final prompt

Benefit: better relevance and less irrelevant context in the prompt.

== Common Failure Modes + Fixes

*1) Retrieval misses exact matches*
- Dense embeddings may miss rare terms, codes, names
- Fix: hybrid retrieval (dense + keyword/BM25), query expansion

*2) “Lost in the middle”*
- LLM pays less attention to evidence buried mid-context
- Fix: put strongest evidence early; rerank; reduce $K$; compress context

*3) Long-context dilution / noise*
- Too many chunks -> irrelevant info -> worse answer
- Fix: stricter thresholds, rerank, chunk size tuning, dedup similar chunks

*4) Domain mismatch*
- Embedding model trained on general text may fail on specialized data
- Fix: choose domain embeddings; fine-tune embeddings; add metadata filters

*5) Conflicting sources*
- Retrieved chunks disagree
- Fix: include timestamps/sources; instruct model to compare and state uncertainty

== Practical Design Choices (What to Tune)
- Chunk size $L$ and overlap
- Embedding model choice (general vs domain)
- Top-$K$ retrieval and score thresholds
- Reranker usage (on/off; candidate count)
- Prompt format (citations, strict “only use context” rules)
- Metadata filters (recency, source type, access control)

#hinweis[
A good RAG system is often more about *data quality + chunking + retrieval tuning* than about changing the LLM.
]
