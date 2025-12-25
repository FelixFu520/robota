from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_openai import ChatOpenAI
import os
from langchain.tools import tool

@tool
def send_email(email: str) -> str:
    """Send an email to the team."""
    return f"Email sent to {email}"

@tool
def delete_database(database: str) -> str:
    """Delete a database."""
    return f"Database deleted: {database}"

@tool
def search(query: str) -> str:
    """Search the database."""
    return f"Search results for {query}"

model = ChatOpenAI(
    model="doubao-seed-1-6-251015",  # Specify a model available on OpenRouter
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

agent = create_agent(
    model=model,
    tools=[send_email, delete_database, search],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # Require approval for sensitive operations
                "send_email": True,
                "delete_database": True,
                # Auto-approve safe operations
                "search": False,
            }
        ),
    ],
    # Persist the state across interrupts
    checkpointer=InMemorySaver(),
)

# Human-in-the-loop requires a thread ID for persistence
config = {"configurable": {"thread_id": "1"}}

# Agent will pause and wait for approval before executing sensitive tools
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to the team, team@163.com"}]},
    config=config
)
for m in result["messages"]:
    m.pretty_print()
    print('*'*100)

r2 = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config  # Same thread ID to resume the paused conversation
)
for m in r2["messages"]:
    m.pretty_print()
    print('*'*100)