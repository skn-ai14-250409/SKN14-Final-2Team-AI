# scentpick/mas/nodes/supervisor_node.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from ..config import llm
from ..state import AgentState
from ..prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT
from typing import Any, Dict, List
import json

ALLOWED = {
    "LLM_parser",
    "FAQ_agent",
    "human_fallback",
    "price_agent",
    "ML_agent",
    "memory_echo",
    "rec_echo",
}

def _build_rec_context(state: AgentState) -> str:
    hist = state.get("rec_history") or []
    if not hist or not (hist[-1] or {}).get("items"):
        return "(none)"
    items: List[Dict[str, Any]] = hist[-1]["items"]
    lines = []
    for i, it in enumerate(items, 1):
        name = f"{(it.get('brand','') or '').strip()} {(it.get('name','') or '').strip()}".strip()
        if name:
            lines.append(f"{i}. {name}")
    return "\n".join(lines) if lines else "(none)"

def supervisor_node(state: AgentState) -> AgentState:
    user_query = "(empty)"
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break

    rec_context = _build_rec_context(state)
    last_agent = state.get("last_agent")

    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_SYSTEM_PROMPT),
        ("user", "USER_QUERY:\n{query}\n\nREC_CONTEXT:\n{rec_context}\n\nLAST_AGENT:\n{last_agent}")
    ])
    chain = prompt | llm  # 권장: temperature 0~0.2

    ai = chain.invoke({"query": user_query, "rec_context": rec_context, "last_agent": last_agent})
    raw = getattr(ai, "content", "")

    chosen = "human_fallback"
    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
        nxt = parsed.get("next")
        if isinstance(nxt, str) and nxt in ALLOWED:
            chosen = nxt
        else:
            parsed = {"error": "invalid_next", "raw": raw}
    except Exception:
        parsed = {"error": "invalid_json", "raw": raw}

    # ✅ messages에는 아무것도 추가하지 않음
    return {"next": chosen, "router_json": parsed}
