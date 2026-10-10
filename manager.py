from hr_assistant.assistant import ManagerConsole

HELP = "Commands: list | approve <REQ-ID> [comment] | reject <REQ-ID> [comment] | exit"


def main() -> None:
    print("=" * 60)
    print("PROITBRIDGE Manager Console  ·  leave approvals")
    print("=" * 60)
    console = ManagerConsole()
    print(HELP + "\n")

    while True:
        try:
            line = input("manager> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            break

        parts = line.split(maxsplit=2)
        command = parts[0].lower()

        if command == "list":
            pending = console.pending()
            if not pending:
                print("No pending requests.\n")
            for record in pending:
                print(f"{record['request_id']}  employee={record['employee_id']}  "
                      f"{record['details']}  created={record['created_at']}")
            print()
        elif command in {"approve", "reject"} and len(parts) >= 2:
            comment = parts[2] if len(parts) > 2 else ""
            try:
                print(console.decide(parts[1].upper(), command, comment) + "\n")
            except Exception as error:
                print(f"Sorry, something went wrong: {error}\n")
        else:
            print(HELP + "\n")

    print("Goodbye.")


if __name__ == "__main__":
    main()
