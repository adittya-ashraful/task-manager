"""Update ToDos node.

Uses Trustcall to extract/update ToDo items from the conversation
and persist them in the Store.  Also uses the Spy listener to
capture the exact changes made for user feedback.
"""

import uuid
from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore
from trustcall import create_extractor

from long_memory_agent.prompts import TRUSTCALL_INSTRUCTION
from long_memory_agent.schemas import ToDo
from long_memory_agent.utils import Spy, extract_tool_info


def update_todos(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
) -> dict:
    """Reflect on the chat history and update the ToDo list in the Store."""
    from long_memory_agent.agent import get_model

    model = get_model()

    # Get the user_id from the config
    user_id = config["configurable"]["user_id"]

    # Define the namespace for the memories
    namespace = ("todo", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "ToDo"
    existing_memories = (
        [
            (existing_item.key, tool_name, existing_item.value)
            for existing_item in existing_items
        ]
        if existing_items
        else None
    )

    # Merge the chat history and the instruction
    trustcall_instruction = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            [SystemMessage(content=trustcall_instruction)]
            + state["messages"][:-1]
        )
    )

    # Initialize the spy for visibility into tool calls made by Trustcall
    spy = Spy()

    # Create the Trustcall extractor for updating the todo list
    todo_extractor = create_extractor(
        model,
        tools=[ToDo],
        tool_choice=tool_name,
        enable_inserts=True,
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = todo_extractor.invoke(
        {
            "messages": updated_messages,
            "existing": existing_memories,
        }
    )

    # Save memories from Trustcall to the store
    for r, meta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            meta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

    # Respond to the tool call made in task_manager, confirming the update
    last_message = state["messages"][-1]
    assert isinstance(last_message, AIMessage), "Expected AIMessage with tool_calls"

    # Extract the changes made by Trustcall and include in the ToolMessage
    todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {
        "messages": [
            {
                "role": "tool",
                "content": todo_update_msg,
                "tool_call_id": last_message.tool_calls[0]["id"],
            }
        ]
    }
