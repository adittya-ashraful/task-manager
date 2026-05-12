import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from long_memory_agent.edges.routing import route_message
from long_memory_agent.nodes.task_manager import task_manager_node
from long_memory_agent.nodes.update_instructions import update_instructions
from long_memory_agent.nodes.update_profile import update_profile
from long_memory_agent.nodes.update_todos import update_todos


load_dotenv()

# Model singleton
_model = None


def get_model() -> ChatOpenAI:
    """Return a shared ChatOpenAI model instance (lazy singleton)."""
    global _model
    if _model is None:
        _model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _model


# Building the graph

def build_graph():
    """Construct and compile the Task Manager StateGraph.

    Note: Do NOT pass checkpointer or store here.
    LangGraph API (langgraph dev / LangGraph Cloud) provides its own
    persistence layer automatically.

    Returns:
        A compiled LangGraph graph ready for invocation.
    """
    builder = StateGraph(MessagesState)  # type: ignore[type-var]

    # Add nodes
    builder.add_node("task_manager", task_manager_node)
    builder.add_node("update_todos", update_todos)
    builder.add_node("update_profile", update_profile)
    builder.add_node("update_instructions", update_instructions)

    # Wire edges
    builder.add_edge(START, "task_manager")
    builder.add_conditional_edges("task_manager", route_message)
    builder.add_edge("update_todos", "task_manager")
    builder.add_edge("update_profile", "task_manager")
    builder.add_edge("update_instructions", "task_manager")

    # Compile without checkpointer/store — LangGraph API provides these
    graph = builder.compile()

    return graph



graph = build_graph()

