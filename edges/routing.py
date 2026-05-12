"""
Routing logic for the Task Manager Agent.

Inspects the last message's tool calls to decide which memory-update
node to execute next, or whether to end the conversation turn.
"""

from typing import Literal, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState
from langgraph.store.base import BaseStore

# Return type uses the literal value of END ("__end__") so the type-checker
# can verify the conditional_edges mapping.
RouteDest = Literal["update_todos", "update_instructions", "update_profile", "__end__"]


def route_message(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
) -> RouteDest:
    """Route based on the UpdateMemory tool call in the last message.

    Returns:
        The name of the next node to execute, or END if no tool call.
    """
    # The router only runs after the model node, so the last message is
    # always an AIMessage. Cast to satisfy the type-checker.
    message = cast(AIMessage, state["messages"][-1])

    if not message.tool_calls:
        return cast(RouteDest, END)

    tool_call = message.tool_calls[0]
    update_type: str = tool_call["args"]["update_type"]

    if update_type == "user":
        return "update_profile"
    elif update_type == "todo":
        return "update_todos"
    elif update_type == "instructions":
        return "update_instructions"
    else:
        raise ValueError(f"Unknown update_type: {update_type}")
