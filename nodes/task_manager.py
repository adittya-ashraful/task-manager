from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore

from long_memory_agent.prompts import MODEL_SYSTEM_MESSAGE
from long_memory_agent.schemas import UpdateMemory


def task_manager_node(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
) -> dict:
    """Load memories from the store and produce a personalised chatbot response.

    This node:
    1. Retrieves the user's profile, todo list, and custom instructions
       from the cross-thread Store.
    2. Formats them into the system prompt.
    3. Calls the LLM (with UpdateMemory bound as a tool).
    4. Returns the LLM response (which may contain a tool call).
    """
    from long_memory_agent.agent import get_model  # deferred to avoid circular imports

    model = get_model()

    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve user profile
    namespace = ("profile", user_id)
    memories = store.search(namespace)
    user_profile = memories[0].value if memories else None

    #Retrieve todo items
    namespace = ("todo", user_id)
    memories = store.search(namespace)
    todo = "\n".join(f"{mem.value}" for mem in memories)

    # Retrieve custom instructions
    namespace = ("instructions", user_id)
    existing_memory = store.get(namespace, "user_instructions")
    instructions = existing_memory.value.get("memory", "") if existing_memory else ""

    # Format system message & invoke LLM 
    system_msg = MODEL_SYSTEM_MESSAGE.format(
        user_profile=user_profile,
        todo=todo,
        instructions=instructions,
    )

    response = model.bind_tools(
        [UpdateMemory], parallel_tool_calls=False
    ).invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )

    return {"messages": [response]}
