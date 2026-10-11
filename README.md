PROITBRIDGE HR Assistant

An agentic HR assistant built with LangChain, LangGraph and LangSmith. It answers employee questions from a company handbook, runs deterministic HR calculations and validations through tools, and routes validated leave requests to a manager for approval using human-in-the-loop.

Features
LLM router: classifies each question as policy_question, hr_action or out_of_scope using Pydantic structured output.
Handbook RAG: PDF loading, recursive chunking, OpenAI embeddings and top-k retrieval. Answers use only retrieved context and say so when the handbook has no answer.
7 deterministic HR tools: leave validation, earned-leave accrual, probation end date, payroll deadlines, reimbursement validation, remote-work allowance and travel daily allowance. Each returns a handbook chapter reference.
Rejection explanation loop: when a tool rejects a request, the graph retrieves the relevant handbook section and explains why and what the employee can do instead.
Human-in-the-loop manager approval: a validated leave request pauses the graph with interrupt() until a manager approves or rejects it from a separate console. State is persisted in SQLite, so pending requests survive restarts.
LangSmith tracing: run names, tags, and session and employee metadata on every run.

## Graph

```
START
  |
  v
router
  |-- policy_question --> retrieve_policy --> policy_answer --> END
  |-- out_of_scope ------> out_of_scope ------------------------> END
  |-- hr_action ---------> hr_agent
                             |-- final answer --> END
                             |-- tool calls ----> tools
                                                    |-- otherwise -------> hr_agent (loop)
                                                    |-- rule rejected ---> retrieve_policy --> hr_agent --> END
                                                    |-- leave valid -----> create_request
                                                                               |
                                                                               v
                                                                         await_manager
                                                                    (paused by interrupt until
                                                                     the manager decides)
                                                                               |
                                                                               v
                                                                        finalize_request --> END
```

Project structure
hr_assistant/
├── app.py                  # Employee CLI
├── manager.py              # Manager CLI (list / approve / reject)
├── langgraph.json          # LangGraph Studio config
├── data/
│   └── PROITBRIDGE_Employee_Handbook_2026_1.pdf
└── src/hr_assistant/
    ├── config.py           # Settings, model and RAG parameters, DB paths
    ├── state.py            # HRState (messages, route, approval fields)
    ├── prompt.py           # Router, answer and agent prompts
    ├── rag.py              # Load, split, embed, retrieve
    ├── tools.py            # 7 deterministic HR tools
    ├── nodes.py            # Graph nodes, including the approval nodes
    ├── edges.py            # Conditional routing logic
    ├── graph.py            # Graph assembly (build_graph)
    ├── store.py            # SQLite table of leave requests
    └── assistant.py        # HRAssistant (employee) and ManagerConsole
Setup

Requires Python 3.11+ and uv.

bash
git clone <https://github.com/macklae/Hr_assistant.git>
cd hr_assistant
uv sync

Create a .env file in the project root:

OPENAI_API_KEY=your-openai-key

# Optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=hr-assistant

.env is listed in .gitignore. Do not commit it.

Usage

1. Employee (terminal 1)

bash
uv run app.py

Enter an employee ID, then ask a question, for example:

What is the casual leave policy? (policy answer from the handbook)
How much EL will I accrue after 6 months? (tool calculation)
Can I take 5 days of EL during probation? (rejected, with explanation)
Can I take 2 days of casual leave with 5 days notice? (valid, sent to manager)

A valid leave request prints a request ID and the employee CLI exits.

2. Manager (terminal 2)

bash
uv run manager.py
manager> list
manager> approve REQ-XXXXXXXX ok to proceed
manager> reject REQ-XXXXXXXX reason here

3. Employee checks the outcome

bash
uv run app.py
> /status REQ-XXXXXXXX

Pending requests persist across restarts. Close everything after step 1, open manager.py later, and list still shows the request.

How human-in-the-loop works
create_request writes the request to SQLite (side effects only, runs once).
await_manager calls interrupt() and the graph pauses. The checkpoint is saved in data/hr_checkpoints.db.
manager.py resumes the same thread with Command(resume={"decision": ..., "comment": ...}).
finalize_request records the decision and produces the employee message.
LangGraph Studio

graph.py exposes a checkpointer-free app for uv run langgraph dev. Studio is useful for visualizing the graph and inspecting traces. Requests started in Studio cannot be resumed from manager.py, because the two use separate checkpoint storage.

Tech stack

Python 3.11, LangChain, LangGraph, LangSmith, OpenAI (gpt-4.1-mini, text-embedding-3-small), Pydantic, pypdf, SQLite, python-dotenv, uv.

Limitations
Approval applies to leave requests only.
The manager console has no authentication and employee IDs are self-declared.
There is no timeout or escalation if a manager does not respond.
The vector store is in memory and is rebuilt on each start.
