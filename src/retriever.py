from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


STOPWORDS = {
    "what", "who", "how", "many", "the", "is", "are", "do", "does",
    "to", "of", "and", "a", "an", "in", "for", "on", "with", "get",
    "company", "benefit", "employees", "days"
}


def tokens(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return set(
        w for w in words
        if w not in STOPWORDS and len(w) > 2
    )


class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def filter_by_permission(self, user_role):
        user_role = user_role.lower()

        return [
            doc for doc in self.documents
            if user_role in [role.lower() for role in doc["allowed_roles"]]
        ]

    def rerank_score(self, query, doc, embedding_score):
        query_terms = tokens(query)
        doc_terms = tokens(doc["title"] + " " + doc["content"])

        if not query_terms:
            overlap_ratio = 0
        else:
            overlap_ratio = len(query_terms.intersection(doc_terms)) / len(query_terms)

        title_terms = tokens(doc["title"])
        title_overlap = len(query_terms.intersection(title_terms))

        final_score = (
            embedding_score * 0.70
            + overlap_ratio * 0.25
            + title_overlap * 0.05
        )

        return final_score

    def retrieve(self, query, user_role, top_k=3):
        filtered_docs = self.filter_by_permission(user_role)

        if not filtered_docs:
            return []

        texts = [
            doc["title"] + " " + doc["content"]
            for doc in filtered_docs
        ]

        doc_embeddings = self.model.encode(texts)
        query_embedding = self.model.encode([query])

        scores = cosine_similarity(query_embedding, doc_embeddings)[0]

        ranked = []

        for doc, score in zip(filtered_docs, scores):
            final_score = self.rerank_score(query, doc, float(score))
            ranked.append((doc, final_score))

        ranked = sorted(
            ranked,
            key=lambda item: item[1],
            reverse=True
        )

        return ranked[:top_k]