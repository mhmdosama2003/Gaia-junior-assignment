from data_loader import load_data, clean_data
from retriever import Retriever
from answerer import generate_answer


docs = load_data("data/knowledge_base.json")
docs = clean_data(docs)

retriever = Retriever(docs)


def run_question(query, user_role):
    print("\n" + "=" * 70)
    print("User:", user_role)
    print("Question:", query)

    results = retriever.retrieve(query, user_role, top_k=3)
    answer = generate_answer(query, results)

    print("\nAnswer:", answer["short_answer"])
    print("Support:", answer["support_label"])
    print("Citations:", answer["citations"])

    print("\nEvidence:")
    for ev in answer["evidence_snippets"]:
        print(f"- {ev['doc_id']} | {ev['title']}")
        print(f"  {ev['snippet']}")


roles = [
    "Sales Manager",
    "HR Manager",
    "Engineering Manager",
    "Finance Manager"
]

while True:
    print("\nAvailable Users:")
    for i, role in enumerate(roles, 1):
        print(f"{i}. {role}")

    choice = input("\nSelect user (1-4) or type exit: ")

    if choice.lower() == "exit":
        break

    if not choice.isdigit() or int(choice) not in range(1, 5):
        print("Invalid choice")
        continue

    user_role = roles[int(choice) - 1]

    query = input("Enter your question: ")

    run_question(query, user_role)