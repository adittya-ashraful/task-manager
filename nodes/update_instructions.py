"""
Update Instructions node.

Reflects on the conversation to update the agent's own instructions
for how it should manage the user's ToDo list.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore

from long_memory_agent.prompts import CREATE_INSTRUCTIONS


def update_instructions(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
) -> dict:
    """Reflect on the chat history and update the ToDo management instructions."""
    from long_memory_agent.agent import get_model

    model = get_model()

    # Get the user_id from the config
    user_id = config["configurable"]["user_id"]
    namespace = ("instructions", user_id)

    existing_memory = store.get(namespace, "user_instructions")

    # Format the memory in the system prompt
    system_msg = CREATE_INSTRUCTIONS.format(
        current_instructions=(
            existing_memory.value.get("memory", "") if existing_memory else None
        )
    )

    # Ask the LLM to produce updated instructions
    new_memory = model.invoke(
        [SystemMessage(content=system_msg)]
        + state["messages"][:-1]
        + [HumanMessage(content="Please update the instructions based on this conversation.")]
    )

    # Overwrite the existing memory in the store
    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})

    last_message = state["messages"][-1]
    assert isinstance(last_message, AIMessage), "Expected AIMessage with tool_calls"
    return {
        "messages": [
            {
                "role": "tool",
                "content": "updated instructions",
                "tool_call_id": last_message.tool_calls[0]["id"],
            }
        ]
    }
