"""Minimal long-term memory tools (LangGraph `runtime.store`)."""

from __future__ import annotations

from langchain.tools import ToolRuntime, tool


@tool
def save_session_note(text: str, runtime: ToolRuntime) -> str:
    """Save a short fact to long-term memory. Call this when the user asks you to remember something."""
    if runtime.store is None:
        return "Long-term store is not configured."
    runtime.store.put(("mdcrow", "ltm"), "session", {"note": text})
    return "Saved to long-term memory."


@tool
def read_session_note(runtime: ToolRuntime) -> str:
    """Read the fact previously saved in long-term memory. Use when the user asks what you remember."""
    if runtime.store is None:
        return "Long-term store is not configured."
    item = runtime.store.get(("mdcrow", "ltm"), "session")
    return str(item.value) if item else "No note saved yet."
