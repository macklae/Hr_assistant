import sqlite3
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from . import store
from .config import CHECKPOINT_DB_PATH, LLM_MODEL
from .graph import build_graph


def make_graph():
    """Compile the graph with a SQLite checkpointer shared by employee and manager CLIs."""
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    return build_graph(checkpointer=SqliteSaver(conn))


class HRAssistant:
    """Employee side."""

    def __init__(self, employee_id: str = "unknown") -> None:
        self.employee_id = employee_id
        self.session_id = str(uuid.uuid4())
        self.graph = make_graph()

    def ask(self, question: str) -> str:
        # One thread per question: a paused question can be resumed later by thread_id.
        thread_id = str(uuid.uuid4())
        config = {
            "run_name": "PROITBRIDGE HR Assistant",
            "tags": ["hr-assistant", "human-in-the-loop"],
            "metadata": {
                "session_id": self.session_id,
                "employee_id": self.employee_id,
                "llm_model": LLM_MODEL,
            },
            "configurable": {"thread_id": thread_id, "employee_id": self.employee_id},
        }
        result = self.graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)

        # Paused at the manager-approval step?
        snapshot = self.graph.get_state(config)
        if snapshot.next:
            request_id = snapshot.values.get("request_id", "")
            return (
                f"Your request passed the policy checks and has been sent to your manager "
                f"for approval.\nRequest ID: {request_id}\n"
                f"Check progress with:  /status {request_id}"
            )
        return result["messages"][-1].text

    @staticmethod
    def status(request_id: str) -> str:
        record = store.get_request(request_id.strip().upper())
        if not record:
            return f"No request found with ID {request_id}."
        text = f"Request {record['request_id']}: {record['status']}"
        if record["manager_comment"]:
            text += f" (manager comment: {record['manager_comment']})"
        return text


class ManagerConsole:
    """Manager side: list pending requests and resume the paused graph with a decision."""

    def __init__(self) -> None:
        self.graph = make_graph()

    @staticmethod
    def pending() -> list[dict]:
        return store.list_pending()

    def decide(self, request_id: str, decision: str, comment: str = "") -> str:
        record = store.get_request(request_id)
        if not record:
            return f"No request found with ID {request_id}."
        if record["status"] != "pending":
            return f"Request {request_id} is already {record['status']}."

        config = {"configurable": {"thread_id": record["thread_id"]}}
        result = self.graph.invoke(
            Command(resume={"decision": decision, "comment": comment}), config=config
        )
        return result["messages"][-1].text
