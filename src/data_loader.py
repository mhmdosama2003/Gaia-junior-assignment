import json


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def clean_data(docs):
    cleaned = []
    seen_titles = set()

    for doc in docs:
        # skip empty content
        if not doc["content"].strip():
            continue

        # normalize
        doc["department"] = doc["department"].lower()
        doc["allowed_roles"] = [r.lower() for r in doc["allowed_roles"]]

        # remove duplicates
        title = doc["title"].lower()
        if title in seen_titles:
            continue
        seen_titles.add(title)

        cleaned.append(doc)

    return cleaned