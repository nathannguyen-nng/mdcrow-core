from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _message_heading(msg: BaseMessage) -> str:
    t = getattr(msg, "type", None) or ""
    return {
        "human": "User",
        "ai": "Assistant",
        "system": "System",
        "tool": "Tool",
    }.get(t, t or msg.__class__.__name__)


class AgentRunOutput:
    """Readable in Jupyter via _repr_markdown_; use .state for the raw dict."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self.state.get("messages", []))

    @property
    def final_answer(self) -> str:
        for msg in reversed(self.messages):
            if isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai":
                text = _stringify_message_content(msg.content).strip()
                if text:
                    return text
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    lines = ["*(assistant used tools; see transcript below)*"]
                    for tc in tool_calls:
                        name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                        lines.append(f"- **{name}**")
                    return "\n".join(lines)
        return ""

    def _transcript_markdown(self) -> str:
        chunks: list[str] = []
        for msg in self.messages:
            role = _message_heading(msg)
            body = _stringify_message_content(msg.content)
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tc_lines = "\n".join(
                    (
                        f"- `{tc.get('name', '?')}` — `{tc.get('args', {})}`"
                        if isinstance(tc, dict)
                        else f"- `{getattr(tc, 'name', '?')}`"
                    )
                    for tc in msg.tool_calls
                )
                body = (body + "\n\n**Tool calls**\n" + tc_lines).strip()
            chunks.append(f"### {role}\n\n{body or '*(empty)*'}\n")
        return "\n---\n".join(chunks)

    def __str__(self) -> str:
        parts: list[str] = []
        for msg in self.messages:
            role = _message_heading(msg)
            body = _stringify_message_content(msg.content)
            parts.append(f"[{role}]\n{body}\n")
        return "\n".join(parts).strip()

    def _repr_markdown_(self) -> str:
        return self._transcript_markdown()

    def __repr__(self) -> str:
        return f"AgentRunOutput({len(self.messages)} messages)"
