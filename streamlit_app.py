"""
Streamlit Frontend for the Long Memory Agent.

A polished chat interface that communicates with the LangGraph dev server
via the langgraph-sdk. Features:
  - Thread History sidebar with create / switch / delete threads
  - "Agent Chat" main area with streaming responses
  - Toggle to hide/show tool calls
  - Upload PDF or Image support
  - Memory sidebar panel (profile, todos, instructions)
"""

import base64
import uuid
from datetime import datetime
from typing import Any, cast

import streamlit as st
from langgraph_sdk import get_sync_client

# ─── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agent Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────
LANGGRAPH_URL = "http://localhost:2024"
GRAPH_NAME = "task_manager_agent"
DEFAULT_USER_ID = "streamlit-user-1"


# ─── Custom CSS ───────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ───────────────────────────────────── */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
    }

    /* ── Sidebar ──────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
        border-right: 1px solid rgba(0,0,0,0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #1e293b;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* ── Thread buttons ───────────────────────────── */
    .thread-btn {
        display: block;
        width: 100%;
        padding: 10px 14px;
        margin: 4px 0;
        border-radius: 10px;
        border: 1px solid transparent;
        background: rgba(0,0,0,0.04);
        color: #64748b;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: left;
    }
    .thread-btn:hover {
        background: rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.2);
        color: #475569;
    }
    .thread-btn.active {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
        color: #1e293b;
    }

    /* ── Chat area ────────────────────────────────── */
    .chat-header {
        text-align: center;
        padding: 40px 0 20px 0;
    }
    .chat-header h1 {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin-bottom: 4px;
    }
    .chat-header p {
        color: #6b7280;
        font-size: 14px;
        font-weight: 400;
    }

    /* ── Messages ─────────────────────────────────── */
    .stChatMessage {
        border-radius: 16px !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    }

    /* ── Tool call expander ────────────────────────── */
    .tool-call-box {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 13px;
        color: #4338ca;
    }
    .tool-call-box .tool-name {
        font-weight: 600;
        color: #4f46e5;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    /* ── Input area ────────────────────────────────── */
    .stChatInput > div {
        border-radius: 14px !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        background: rgba(255,255,255,0.8) !important;
    }
    .stChatInput > div:focus-within {
        border-color: rgba(99, 102, 241, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    }

    /* ── Memory panel cards ────────────────────────── */
    .memory-card {
        background: rgba(0,0,0,0.02);
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .memory-card h4 {
        color: #4f46e5;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .memory-card p, .memory-card li {
        color: #475569;
        font-size: 13px;
        line-height: 1.6;
    }

    /* ── Scrollbar ─────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 3px;
    }

    /* ── Buttons ───────────────────────────────────── */
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        font-size: 13px;
        transition: all 0.2s ease;
    }

    /* ── Divider ───────────────────────────────────── */
    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(0,0,0,0.08);
        margin: 16px 0;
    }

    /* ── Upload area ───────────────────────────────── */
    .upload-section {
        margin-top: 8px;
    }

    /* ── Status badge ──────────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .status-connected {
        background: rgba(34, 197, 94, 0.1);
        color: #166534;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    .status-disconnected {
        background: rgba(239, 68, 68, 0.1);
        color: #991b1b;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    /* ── Hide default Streamlit elements ───────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """,
        unsafe_allow_html=True,
    )


inject_css()


# ─── Helper: get LangGraph client ────────────────────────────────────────
@st.cache_resource
def get_client():
    """Create a cached sync LangGraph SDK client."""
    return get_sync_client(url=LANGGRAPH_URL)


def check_server_connection() -> bool:
    """Check if the LangGraph server is reachable."""
    try:
        client = get_client()
        client.assistants.search()
        return True
    except Exception:
        return False


# ─── Session state defaults ──────────────────────────────────────────────
def init_session_state():
    defaults = {
        "threads": {},          # {thread_id: {"name": str, "created_at": str}}
        "current_thread_id": None,
        "messages": [],         # chat messages for current thread
        "hide_tool_calls": False,
        "user_id": DEFAULT_USER_ID,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ─── Thread management helpers ───────────────────────────────────────────
def create_new_thread() -> str:
    """Create a new thread on the LangGraph server and track it."""
    client = get_client()
    thread = client.threads.create()
    thread_id = thread["thread_id"]
    st.session_state.threads[thread_id] = {
        "name": f"Thread {len(st.session_state.threads) + 1}",
        "created_at": datetime.now().strftime("%b %d, %H:%M"),
    }
    st.session_state.current_thread_id = thread_id
    st.session_state.messages = []
    return thread_id


def switch_thread(thread_id: str):
    """Switch to an existing thread and reload its message history."""
    st.session_state.current_thread_id = thread_id
    load_thread_messages(thread_id)


def load_thread_messages(thread_id: str):
    """Load message history for a thread from the LangGraph server."""
    client = get_client()
    try:
        state = cast(dict[str, Any], client.threads.get_state(thread_id))
        raw_messages = state.get("values", {}).get("messages", [])
        st.session_state.messages = []
        for msg in raw_messages:
            role = msg.get("type", msg.get("role", ""))
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", []) if isinstance(msg, dict) else []

            if role == "human":
                st.session_state.messages.append({"role": "user", "content": content})
            elif role == "ai" or role == "assistant":
                entry = {"role": "assistant", "content": content}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                st.session_state.messages.append(entry)
            elif role == "tool":
                st.session_state.messages.append(
                    {"role": "tool", "content": content, "name": msg.get("name", "tool")}
                )
    except Exception:
        st.session_state.messages = []


def delete_thread(thread_id: str):
    """Delete a thread."""
    try:
        client = get_client()
        client.threads.delete(thread_id)
    except Exception:
        pass
    st.session_state.threads.pop(thread_id, None)
    if st.session_state.current_thread_id == thread_id:
        st.session_state.current_thread_id = None
        st.session_state.messages = []


# ─── Memory helpers ──────────────────────────────────────────────────────
def load_memories() -> dict:
    """Load profile, todos, and instructions from the LangGraph Store."""
    client = get_client()
    user_id = st.session_state.user_id
    memories = {"profile": None, "todos": [], "instructions": ""}

    try:
        # Profile
        items = client.store.search_items(("profile", user_id))
        if items and items.get("items"):
            memories["profile"] = items["items"][0].get("value", {})
    except Exception:
        pass

    try:
        # Todos
        items = client.store.search_items(("todo", user_id))
        if items and items.get("items"):
            memories["todos"] = [item.get("value", {}) for item in items["items"]]
    except Exception:
        pass

    try:
        # Instructions
        item = client.store.get_item(("instructions", user_id), "user_instructions")
        if item and item.get("value"):
            memories["instructions"] = item["value"].get("memory", "")
    except Exception:
        pass

    return memories


# ─── Send message to agent ───────────────────────────────────────────────
def send_message(user_input: str, attachments: list | None = None):
    """Send a message to the agent and stream the response."""
    thread_id = st.session_state.current_thread_id
    if not thread_id:
        thread_id = create_new_thread()

    client = get_client()

    # Build the input
    input_message: dict[str, Any] = {"role": "user", "content": user_input}

    # If there are image attachments, encode them
    if attachments:
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_input}]
        for att in attachments:
            if att["type"] == "image":
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{att['mime']};base64,{att['data']}"},
                    }
                )
            elif att["type"] == "text":
                content_parts.append({"type": "text", "text": att["data"]})
        input_message = {"role": "user", "content": content_parts}

    # Add user message to local state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Config for the agent
    config = {"configurable": {"user_id": st.session_state.user_id}}

    try:
        # Use stream for real-time response
        collected_content = ""
        tool_calls_collected = []

        # Use wait mode to get the final result
        agent_input: dict[str, Any] = {"messages": [input_message]}
        result = cast(dict[str, Any], client.runs.wait(
            thread_id=thread_id,
            assistant_id=GRAPH_NAME,
            input=agent_input,
            config=config,
        ))

        # Parse the response messages
        if result and "messages" in result:
            response_messages = result["messages"]
            for msg in response_messages:
                msg_type = msg.get("type", msg.get("role", ""))
                content = msg.get("content", "")
                tc = msg.get("tool_calls", [])

                if msg_type == "ai" or msg_type == "assistant":
                    if tc:
                        tool_calls_collected.extend(tc)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": content, "tool_calls": tc}
                        )
                    elif content:
                        collected_content = content
                elif msg_type == "tool":
                    st.session_state.messages.append(
                        {"role": "tool", "content": content, "name": msg.get("name", "tool")}
                    )

            # Add final assistant message
            if collected_content:
                st.session_state.messages.append(
                    {"role": "assistant", "content": collected_content}
                )

            # Auto-name the thread after first message
            if len(st.session_state.messages) <= 3:
                thread_meta = st.session_state.threads.get(thread_id, {})
                preview = user_input[:40] + ("..." if len(user_input) > 40 else "")
                thread_meta["name"] = preview
                st.session_state.threads[thread_id] = thread_meta

    except Exception as e:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"⚠️ Error communicating with agent: {str(e)}"}
        )


# ─── SIDEBAR ─────────────────────────────────────────────────────────────
with st.sidebar:
    # Connection status
    connected = check_server_connection()
    status_class = "status-connected" if connected else "status-disconnected"
    status_text = "Connected" if connected else "Disconnected"
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
            <span class="status-badge {status_class}">● {status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 📋 Thread History")
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # New Thread button
    if st.button("➕  New Thread", use_container_width=True, key="new_thread_btn"):
        if connected:
            create_new_thread()
            st.rerun()
        else:
            st.error("Cannot create thread – server is not connected.")

    # List existing threads
    if st.session_state.threads:
        for tid, meta in reversed(list(st.session_state.threads.items())):
            is_active = tid == st.session_state.current_thread_id
            col1, col2 = st.columns([5, 1])
            with col1:
                label = f"{'▶ ' if is_active else ''}{meta['name']}"
                if st.button(
                    label,
                    key=f"thread_{tid}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    switch_thread(tid)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{tid}", help="Delete thread"):
                    delete_thread(tid)
                    st.rerun()
    else:
        st.caption("No threads yet. Start a new one!")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Memory panel
    st.markdown("## 🧠 Agent Memory")
    if connected and st.session_state.current_thread_id:
        if st.button("🔄 Refresh Memory", use_container_width=True, key="refresh_mem"):
            st.rerun()

        memories = load_memories()

        # Profile
        with st.expander("👤 User Profile", expanded=False):
            if memories["profile"]:
                profile = memories["profile"]
                for key, val in profile.items():
                    if val:
                        if isinstance(val, list):
                            st.markdown(f"**{key.title()}:** {', '.join(val)}")
                        else:
                            st.markdown(f"**{key.title()}:** {val}")
            else:
                st.caption("No profile data yet.")

        # Todos
        with st.expander("📝 ToDo List", expanded=False):
            if memories["todos"]:
                for i, todo in enumerate(memories["todos"], 1):
                    task = todo.get("task", "Untitled")
                    status = todo.get("status", "not started")
                    status_icons = {
                        "not started": "⬜",
                        "in progress": "🔶",
                        "done": "✅",
                        "archived": "📦",
                    }
                    icon = status_icons.get(status, "⬜")
                    st.markdown(f"{icon} **{task}**")
                    if todo.get("deadline"):
                        st.caption(f"  📅 Deadline: {todo['deadline']}")
                    if todo.get("solutions"):
                        st.caption(f"  💡 {', '.join(todo['solutions'])}")
            else:
                st.caption("No todos yet.")

        # Instructions
        with st.expander("⚙️ Custom Instructions", expanded=False):
            if memories["instructions"]:
                st.markdown(memories["instructions"])
            else:
                st.caption("No custom instructions yet.")
    else:
        st.caption("Connect to the server and select a thread to view memory.")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # User ID config
    with st.expander("🔧 Settings", expanded=False):
        new_user_id = st.text_input(
            "User ID",
            value=st.session_state.user_id,
            key="user_id_input",
        )
        if new_user_id != st.session_state.user_id:
            st.session_state.user_id = new_user_id


# ─── MAIN CHAT AREA ──────────────────────────────────────────────────────
# Header
if not st.session_state.messages:
    st.markdown(
        """
        <div class="chat-header">
            <h1>🤖 Agent Chat</h1>
            <p>Your personal task manager with long-term memory</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Display messages
for msg in st.session_state.messages:
    role = msg["role"]

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])

    elif role == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            # Show tool calls if present and not hidden
            if msg.get("tool_calls") and not st.session_state.hide_tool_calls:
                for tc in msg["tool_calls"]:
                    tool_name = tc.get("name", "Unknown Tool")
                    tool_args = tc.get("args", {})
                    st.markdown(
                        f"""<div class="tool-call-box">
                            <div class="tool-name">🔧 {tool_name}</div>
                            <div>{tool_args}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            if msg.get("content"):
                st.markdown(msg["content"])

    elif role == "tool" and not st.session_state.hide_tool_calls:
        with st.chat_message("assistant", avatar="🔧"):
            tool_name = msg.get("name", "tool")
            st.markdown(
                f"""<div class="tool-call-box">
                    <div class="tool-name">📥 Tool Response: {tool_name}</div>
                    <div>{msg['content']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

# Chat input
if prompt := st.chat_input("Type your message...", key="chat_input"):
    if not connected:
        st.error("⚠️ Cannot send message – LangGraph server is not connected. Run `langgraph dev` first.")
    else:
        send_message(prompt, None)
        st.rerun()

# Footer
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center; padding:40px 0 20px 0; color:#4b5563; font-size:13px;">
            Start a new thread from the sidebar, then type a message to begin.
            <br>The agent will automatically remember your profile, tasks, and preferences.
        </div>
        """,
        unsafe_allow_html=True,
    )
