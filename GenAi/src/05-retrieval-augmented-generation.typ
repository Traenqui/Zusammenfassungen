= RAG (Retrieval-Augmented Generation)

LLMs don't know everything. RAG retrieves additional knowledge from external knowledge base. Uses Vector Search with cosine similarity to find relevant documents.

*Vector DB:* Rely on K nearest neighbors like Annoy or FAISS

== Core Pipeline
Query → Embed → Search Vector DB → get relevant context → append to prompt → LLM answers ("grounded generation")

== Chunking Strategies
*One vector per document:*
- Entire document embedded as one vector
- Strong compression → loss of fine-grained information

*Many vectors per document:*
- Split document into chunks
- Chunk units: lines, paragraphs, sections
- Often use overlapping windows to preserve context
- Goal: balance semantic completeness and retrieval precision

== RAG Answer Generation
- User prompt converted into embedding
- Embedding used to search vector database
- Most relevant chunks retrieved
- Retrieved context injected into prompt
- LLM generates answer conditioned on retrieved evidence

== Retrieval Evaluation
$ "Precision@K" = ("# relevant retrieved in top-k") / K $
$ "Recall@K" = ("# relevant retrieved") / ("# relevant in dataset") $

== Retrieval Shortcomings + Fix
- Top-k cutoff / threshold matters
- Exact phrase match → dense retrieval may fail → use *hybrid search* (semantic + keyword)
- Domain shift: retriever trained on web/Wikipedia may perform worse on legal/medical unless trained with domain data
- Long-context issue ("lost in the middle"): LLM may miss relevant info if buried in prompt

== Reranking (Solution)
*Goal tradeoff:* retrieve many (high retrieval recall) but send few to LLM (LLM uses short context better)

*Two-stage:* first retrieve (dense/keyword/hybrid) → rerank → take top-n

*How:* cross-encoder scores each (query, doc) pair jointly → reorder by relevance score