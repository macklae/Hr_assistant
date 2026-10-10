from langsmith import Client

from hr_assistant.assistant import HRAssistant
from hr_assistant.config import LANGSMITH_PROJECT, LANGSMITH_TRACING


def main() -> None:
    print("=" * 60)
    print("PROITBRIDGE HR Assistant  ·  LangGraph + LangSmith")
    print("=" * 60)

    employee_id = input("Enter your employee ID:\n> ").strip() or "unknown"
    assistant = HRAssistant(employee_id=employee_id)

    if LANGSMITH_TRACING:
        print(f"LangSmith tracing : ON  (project: {LANGSMITH_PROJECT})")
        print(f"Session id        : {assistant.session_id}")
    else:
        print("LangSmith tracing : OFF (set LANGSMITH_TRACING=true in .env)")
    print("Ask a question about PROITBRIDGE HR policy. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Ask your HR question:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        if question.lower().startswith("/status"):
            request_id = question[len("/status"):].strip()
            print("\n" + (assistant.status(request_id) if request_id else "Usage: /status <REQUEST-ID>") + "\n")
            continue

        try:
            answer = assistant.ask(question)
            print("\n" + answer + "\n")
            if "Request ID:" in answer:
                break  # request submitted: employee CLI exits, manager acts separately
        except Exception as error:
            print(f"\nSorry, something went wrong: {error}\n")

    # Traces are sent in the background. Wait for the last ones before exiting.
    if LANGSMITH_TRACING:
        Client().flush()
    print("Goodbye.")


if __name__ == "__main__":
    main()