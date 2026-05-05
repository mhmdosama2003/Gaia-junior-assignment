# Gaia Junior Data Scientist Assignment

## Overview
This project is a permission-aware RAG-style prototype that answers questions using a mock internal company knowledge base.

The system retrieves only documents the user is allowed to access and generates grounded answers with citations. If the evidence is weak or inaccessible, it returns `insufficient_evidence` instead of guessing.

---

## Tech Stack
- Python
- JSON (dataset)
- sentence-transformers (embeddings)
- scikit-learn (cosine similarity)

---

## Dataset
The dataset is a mock internal knowledge base stored in:

data/knowledge_base.json

Each document includes:
- id
- title
- department
- content
- allowed_roles
- source_type

The dataset contains:
- role-specific documents (Sales, HR, Engineering, Finance)
- shared documents
- restricted documents
- duplicate and empty documents (used for cleaning)

---

## Data Cleaning
Before retrieval, the system performs basic cleaning:
- Removes empty documents
- Removes duplicate titles
- Normalizes casing for roles and departments

This ensures consistent permission matching and better retrieval quality.

---

## Retrieval
The system uses sentence-transformer embeddings to perform semantic search.

Steps:
1. Filter documents based on user role (permission-aware)
2. Convert documents (title + content) into embeddings
3. Compute cosine similarity with the query
4. Apply reranking using:
   - embedding similarity
   - keyword overlap
   - title overlap
5. Return top-k results

---

## Permissions
Permissions are enforced **before retrieval**.

Only documents where:
user_role ∈ allowed_roles

are used.

Unauthorized documents are never:
- retrieved
- ranked
- used in answers
- cited

---

## Answer Generation
The system uses extractive answer generation.

Steps:
1. Take the top retrieved document
2. Check similarity threshold
3. Check token overlap between query and document
4. Extract the most relevant sentence from the document
5. Return answer with citation and evidence

If evidence is weak or missing:
insufficient_evidence

---

## Evaluation
Evaluation is performed using a small test set.

Metrics:
- Precision@1
- Precision@3
- Recall@3
- Behavior Accuracy

Precision and recall are calculated only on answerable questions.

Results:

Answerable Precision@1: 1.00  
Answerable Precision@3: 0.56  
Answerable Recall@3: 1.00  
Behavior Accuracy: 1.00  

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt