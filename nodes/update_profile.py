"""
Update Profile node.

Uses Trustcall to extract/update user profile information
from the conversation and persist it in the Store.
"""

import uuid
from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore
from trustcall import create_extractor

from long_memory_agent.prompts import TRUSTCALL_INSTRUCTION
from long_memory_agent.schemas import Profile


def update_profile(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
) -> dict:
    """Reflect on the chat history and update the user's profile in the Store."""
    from long_memory_agent.agent import get_model

    model = get_model()

    # Get the user id from the config
    user_id = config["configurable"]["user_id"]

    # Define the namespace for the memory
    namespace = ("profile", user_id)

    # Retrieve the most recent profile from the store
    existing_items = store.search(namespace)

    # Format the existing memory for the Trustcall extractor
    tool_name = "Profile"
    existing_memories = (
        [
            (existing_item.key, tool_name, existing_item.value)
            for existing_item in existing_items
        ]
        if existing_items
        else None
    )

    # Create the Trustcall extractor for updating the profile
    profile_extractor = create_extractor(
        model,
        tools=[Profile],
        tool_choice=tool_name,
        enable_inserts=True,
    )

    # Merge the chat history and the instructions
    trustcall_instruction = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            [SystemMessage(content=trustcall_instruction)]
            + state["messages"][:-1]
        )
    )

    # Invoke the extractor
    result = profile_extractor.invoke(
        {
            "messages": updated_messages,
            "existing": existing_memories,
        }
    )

    # Save the memories from Trustcall to the store
    for r, meta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            meta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

    # Return a ToolMessage to satisfy the pending tool call
    last_message = state["messages"][-1]
    assert isinstance(last_message, AIMessage), "Expected AIMessage with tool_calls"
    return {
        "messages": [
            {
                "role": "tool",
                "content": "updated profile",
                "tool_call_id": last_message.tool_calls[0]["id"],
            }
        ]
    }
