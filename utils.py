class Spy:
    """LangChain listener that captures tool calls made by a Trustcall extractor.

    Usage:
        spy = Spy()
        extractor_with_spy = extractor.with_listeners(on_end=spy)
        result = extractor_with_spy.invoke(...)
        print(spy.called_tools)
    """

    def __init__(self):
        self.called_tools: list = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                try:
                    self.called_tools.append(
                        r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                    )
                except (KeyError, IndexError, TypeError):
                    pass


def extract_tool_info(tool_calls: list, schema_name: str = "Memory") -> str:
    """Extract human-readable information from Trustcall tool calls.

    Parses both PatchDoc (updates to existing memories) and new-creation
    calls, returning a formatted summary string.

    Args:
        tool_calls: Nested list of tool calls captured by the Spy.
        schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile").

    Returns:
        A formatted string summarising all changes made.
    """
    changes: list[dict] = []

    for call_group in tool_calls:
        for call in call_group:
            if call["name"] == "PatchDoc":
                changes.append(
                    {
                        "type": "update",
                        "doc_id": call["args"]["json_doc_id"],
                        "planned_edits": call["args"]["planned_edits"],
                        "value": call["args"]["patches"][0]["value"]
                        if call["args"].get("patches")
                        else "",
                    }
                )
            elif call["name"] == schema_name:
                changes.append(
                    {
                        "type": "create",
                        "value": call["args"],
                    }
                )

    result_parts: list[str] = []
    for change in changes:
        if change["type"] == "update":
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )

    return "\n\n".join(result_parts) if result_parts else "Memory updated successfully."
