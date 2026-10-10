from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from .config import LLM_MODEL, TEMPERATURE, check_config
from .prompt import (
    HR_AGENT_PROMPT,
    OUT_OF_SCOPE_MESSAGE,
    POLICY_ANSWER_PROMPT,
    REJECTION_CONTEXT_SUFFIX,
    ROUTER_PROMPT,
)
from .rag import build_retriever, format_docs
from .state import HRState, Route
from .tools import HR_TOOLS

check_config()

llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)

retriever = build_retriever()


def latest_question(state: HRState) -> str:
    """The most recent thing the employee typed."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


#Node 1
class RouteDecision(BaseModel):
    """The router's decision. with_structured_output forces the LLM to fill this."""

    route: Route = Field(description="The single route this question belongs to.")
    reason: str = Field(description="One short sentence explaining the choice.")


router_llm = llm.with_structured_output(RouteDecision)


def router(state: HRState) -> dict:
    decision = router_llm.invoke(
        [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=latest_question(state))]
    )
    # Reset the per-question keys so every new question starts clean.
    return {
        "route": decision.route,
        "policy_context": "",
        "rejection_explained": False,
        "request_id": "",
        "approval_status": "",
        "manager_comment": "",
        "pending_request": {},
    }



# NODE 2

def retrieve_policy(state: HRState) -> dict:
    question = latest_question(state)
    update = {}

    if state.get("route") == "hr_action":
        # We got here because a validation tool said valid=False.
        # Search with the rejection reason so we fetch the RULE behind it.
        for message in reversed(state["messages"]):
            if isinstance(message, ToolMessage):
                question = f"{question}\n{message.text}"
                break
        update["rejection_explained"] = True

    documents = retriever.invoke(question)
    update["policy_context"] = (
        format_docs(documents) or "No relevant section was found in the Employee Handbook."
    )
    return update



# NODE 3
def policy_answer(state: HRState) -> dict:
    system = SystemMessage(
        content=POLICY_ANSWER_PROMPT.format(context=state.get("policy_context", ""))
    )
    response = llm.invoke([system, *state["messages"]])
    return {"messages": [response]}


# NODE 4
llm_with_tools = llm.bind_tools(HR_TOOLS)

llm_answer_only = llm.bind_tools(HR_TOOLS, tool_choice="none")


def hr_agent(state: HRState) -> dict:
    if state.get("rejection_explained"):
        prompt = HR_AGENT_PROMPT + REJECTION_CONTEXT_SUFFIX.format(
            context=state.get("policy_context", "")
        )
        response = llm_answer_only.invoke([SystemMessage(content=prompt), *state["messages"]])
    else:
        response = llm_with_tools.invoke(
            [SystemMessage(content=HR_AGENT_PROMPT), *state["messages"]]
        )
    return {"messages": [response]}

# Node 5
tools = ToolNode(HR_TOOLS)


# NODE 6

def out_of_scope(state: HRState) -> dict:
    return {"messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)]}


# ---------------------------------------------------------------------------
# HUMAN-IN-THE-LOOP: manager approval of a validated leave request
# ---------------------------------------------------------------------------
import json
import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from . import store


def _validated_leave_args(state: HRState) -> dict:
    """Arguments of the validate_leave_request call that just passed validation."""
    tool_calls = {}
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_calls = {c["id"]: c for c in message.tool_calls}
            break
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage) and message.name == "validate_leave_request":
            if json.loads(message.content).get("valid") is True:
                call = tool_calls.get(message.tool_call_id)
                if call:
                    return dict(call["args"])
    return {}


# NODE 7 -- side effects only (runs once; never re-run on resume)
def create_request(state: HRState, config: RunnableConfig) -> dict:
    configurable = config.get("configurable", {})
    details = _validated_leave_args(state)
    details["employee_note"] = latest_question(state)

    request_id = f"REQ-{uuid.uuid4().hex[:8].upper()}"
    store.create_request(
        request_id=request_id,
        thread_id=configurable.get("thread_id", ""),
        employee_id=configurable.get("employee_id", "unknown"),
        details=details,
    )
    return {"request_id": request_id, "pending_request": details, "approval_status": "pending"}


# NODE 8 -- pause here until the manager resumes the graph.
# Keep this node free of side effects: on resume LangGraph re-runs it from the top.
def await_manager(state: HRState) -> dict:
    decision = interrupt(
        {
            "request_id": state["request_id"],
            "request": state["pending_request"],
        }
    )
    status = "approved" if decision.get("decision") == "approve" else "rejected"
    return {"approval_status": status, "manager_comment": decision.get("comment", "")}


# NODE 9 -- record the outcome and tell the employee
def finalize_request(state: HRState) -> dict:
    status = state["approval_status"]
    comment = state.get("manager_comment", "")
    store.record_decision(state["request_id"], status, comment)

    request = state["pending_request"]
    summary = f"{request.get('leave_type', '')} leave, {request.get('days', '')} day(s)"
    text = f"Your leave request {state['request_id']} ({summary}) was {status} by your manager."
    if comment:
        text += f" Manager comment: {comment}"
    return {"messages": [AIMessage(content=text)]}
