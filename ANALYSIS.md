# Analysis

## Retrieval Approach
The system uses sentence-transformer embeddings to perform semantic search. Documents (title + content) are converted into vectors, and cosine similarity is used to rank them.

A simple reranking step improves results by combining:
- embedding similarity
- keyword overlap
- title overlap

This hybrid approach balances semantic understanding with explainability.

Why this approach?

This approach was chosen because embeddings provide semantic understanding, while keyword and title overlap improve precision and interpretability in a small dataset.
---

## Permission Model
Permissions are enforced before retrieval.

Only documents accessible to the user role are passed to the retriever. Unauthorized documents are never used in:
- ranking
- answer generation
- citations

This guarantees that users never see restricted information.

---

## Evaluation Results
The system achieved:

- Answerable Precision@1: 1.00
- Answerable Precision@3: 0.56
- Answerable Recall@3: 1.00
- Behavior Accuracy: 1.00

Interpretation:
- High recall means the correct document is always retrieved
- Precision@3 is lower because multiple candidate documents are returned
- Precision@1 is high, meaning the correct document is ranked first
- Behavior accuracy confirms correct handling of supported vs insufficient cases

---

## Failure Cases
- Embeddings may retrieve semantically related but incorrect documents
- Keyword overlap may fail when query wording differs significantly
- Small dataset limits variability and robustness

---

## Improvements
- Add a cross-encoder reranker for better precision
- Improve query expansion and synonym handling
- Add document chunking for longer texts
- Use a vector database for scalability
- Introduce LLM for answer formatting (not reasoning)

---