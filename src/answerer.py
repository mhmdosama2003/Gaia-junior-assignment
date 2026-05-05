import re


STOPWORDS = {
    "what", "who", "how", "many", "the", "is", "are", "do", "does",
    "to", "of", "and", "a", "an", "in", "for", "on", "with", "get",
    "company", "benefit", "lunch", "employees", "days"
}

SYNONYMS = {
    "vacation": "leave",
    "holiday": "leave",
    "holidays": "leave",
    "timeoff": "leave",
    "approves": "approve",
    "approved": "approve",
    "deployments": "deployment",
    "deployed": "deployment",
    "submitted": "submit",
    "submits": "submit",
    "started": "start",
    "begin": "start",
    "begins": "start"
}


def tokens(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    normalized = []

    for word in words:
        if word in STOPWORDS or len(word) <= 2:
            continue
        normalized.append(SYNONYMS.get(word, word))

    return set(normalized)


def make_snippet(text, max_chars=220):
    text = " ".join(text.split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def split_sentences(text):
    return [s.strip() for s in text.split(".") if s.strip()]


def best_answer_sentence(query, content):
    sentences = split_sentences(content)

    if not sentences:
        return ""

    query_terms = tokens(query)

    best_sentence = sentences[0]
    best_score = 0

    for sentence in sentences:
        sentence_terms = tokens(sentence)
        score = len(query_terms.intersection(sentence_terms))

        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence + "."


def insufficient_response():
    return {
        "short_answer": "There is not enough accessible evidence to answer this question.",
        "support_label": "insufficient_evidence",
        "citations": [],
        "evidence_snippets": []
    }


def generate_answer(query, retrieved_results, min_score=0.40):
    if not retrieved_results:
        return insufficient_response()

    top_doc, top_score = retrieved_results[0]

    if top_score < min_score:
        return insufficient_response()

    query_terms = tokens(query)
    doc_terms = tokens(top_doc["title"] + " " + top_doc["content"])
    overlap = query_terms.intersection(doc_terms)

    if query_terms and len(overlap) == 0:
        return insufficient_response()

    short_answer = best_answer_sentence(query, top_doc["content"])

    return {
        "short_answer": short_answer,
        "support_label": "supported",
        "citations": [top_doc["id"]],
        "evidence_snippets": [
            {
                "doc_id": top_doc["id"],
                "title": top_doc["title"],
                "snippet": make_snippet(top_doc["content"])
            }
        ]
    }