import json
from data_loader import load_data, clean_data
from retriever import Retriever
from answerer import generate_answer


def load_eval_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def precision_at_k(retrieved_ids, expected_ids, k):
    retrieved_k = retrieved_ids[:k]

    if not retrieved_k:
        return 0.0

    hits = len(set(retrieved_k) & set(expected_ids))
    return hits / len(retrieved_k)


def recall_at_k(retrieved_ids, expected_ids, k):
    if not expected_ids:
        return None

    retrieved_k = retrieved_ids[:k]
    hits = len(set(retrieved_k) & set(expected_ids))
    return hits / len(expected_ids)


def evaluate():
    docs = load_data("data/knowledge_base.json")
    docs = clean_data(docs)

    eval_questions = load_eval_questions("data/eval_questions.json")
    retriever = Retriever(docs)

    k = 3

    total_precision_1 = 0
    total_precision_3 = 0
    total_recall_3 = 0
    answerable_count = 0

    behavior_correct = 0

    print("\nEvaluation Results")
    print("=" * 70)

    for i, case in enumerate(eval_questions, start=1):
        question = case["question"]
        user_role = case["user_role"]
        expected_ids = case["expected_relevant_doc_ids"]
        expected_behavior = case["expected_behavior"]

        results = retriever.retrieve(question, user_role, top_k=k)
        retrieved_ids = [doc["id"] for doc, score in results]

        answer = generate_answer(question, results)
        actual_behavior = answer["support_label"]

        p1 = precision_at_k(retrieved_ids, expected_ids, 1)
        p3 = precision_at_k(retrieved_ids, expected_ids, 3)
        r3 = recall_at_k(retrieved_ids, expected_ids, 3)

        if expected_ids:
            total_precision_1 += p1
            total_precision_3 += p3
            total_recall_3 += r3
            answerable_count += 1

        if actual_behavior == expected_behavior:
            behavior_correct += 1

        print(f"\nCase {i}")
        print("Question:", question)
        print("User:", user_role)
        print("Expected Docs:", expected_ids)
        print("Retrieved Docs:", retrieved_ids)
        print("Expected Behavior:", expected_behavior)
        print("Actual Behavior:", actual_behavior)

        if expected_ids:
            print(f"Precision@1: {p1:.2f}")
            print(f"Precision@3: {p3:.2f}")
            print(f"Recall@3: {r3:.2f}")
        else:
            print("Precision@1: N/A")
            print("Precision@3: N/A")
            print("Recall@3: N/A")

    n = len(eval_questions)

    print("\n" + "=" * 70)
    print("Summary")

    if answerable_count > 0:
        print(f"Answerable Precision@1: {total_precision_1 / answerable_count:.2f}")
        print(f"Answerable Precision@3: {total_precision_3 / answerable_count:.2f}")
        print(f"Answerable Recall@3: {total_recall_3 / answerable_count:.2f}")
    else:
        print("Answerable Precision@1: N/A")
        print("Answerable Precision@3: N/A")
        print("Answerable Recall@3: N/A")

    print(f"Behavior Accuracy: {behavior_correct / n:.2f}")


if __name__ == "__main__":
    evaluate()