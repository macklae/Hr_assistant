import json
from langchain_core.messages import AIMessage, ToolMessage
from .state import HRState

def route_question(state: HRState) -> str:
    """Branch 1: send the question down the right path."""
    return state["route"]

def after_hr_tools(state: HRState) -> str:
    """Branch 2: what did the validation tools say?

    - A tool REJECTED the request (valid is False), not yet explained
          -> "explain_rejection" (detour through retrieve_policy)
    - validate_leave_request PASSED (valid is True)
          -> "request_approval" (human-in-the-loop: manager must approve)
    - Otherwise
          -> "hr_agent" (the normal agent <-> tools loop)
    """
    rejected = False
    leave_approved_by_rules = False

    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            break
        if isinstance(message, ToolMessage):
            # ToolNode stores the tool's dict as a JSON string.
            result = json.loads(message.content)
            if result.get("valid") is False:
                rejected = True
            if message.name == "validate_leave_request" and result.get("valid") is True:
                leave_approved_by_rules = True

    if rejected and not state.get("rejection_explained"):
        return "explain_rejection"
    if leave_approved_by_rules and not rejected:
        return "request_approval"
    return "hr_agent"

def after_retrieval(state: HRState) -> str:
    """Branch 3: retrieve_policy is shared by two paths -- where do we go next?"""
    if state.get("route") == "hr_action":
        return "hr_agent"       # came from the explain-rejection detour
    return "policy_answer"      # came straight from the router