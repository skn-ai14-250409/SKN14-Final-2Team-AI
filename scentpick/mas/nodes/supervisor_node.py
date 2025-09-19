# scentpick/mas/nodes/supervisor_node.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from ..config import llm
from ..state import AgentState
from ..prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT
from typing import Any, Dict, List
import json
import logging

logger = logging.getLogger(__name__)

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

    # 🔧 핵심: system 프롬프트를 템플릿 문자열에 직접 넣지 말고,
    #          {system} 하나로 받아 주입합니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system}"),  # ← system 본문은 값으로 주입 (중괄호 문제 해결)
        ("user", "USER_QUERY:\n{query}\n\nREC_CONTEXT:\n{rec_context}\n\nLAST_AGENT:\n{last_agent}")
    ])

    chain = prompt | llm  # 온도는 llm 설정에서 0~0.2 권장

    try:
        ai = chain.invoke({
            "system": SUPERVISOR_SYSTEM_PROMPT,  # 🔧 여기로 안전하게 주입
            "query": user_query,
            "rec_context": rec_context,
            "last_agent": last_agent,
        })
    except Exception as e:
        # 템플릿 변수 관련 에러가 여기서 사라집니다.
        msg = f"[supervisor_node] Prompt invoke error: {e}"
        logger.error(msg)
        return {"next": "human_fallback", "router_json": {"error": "prompt_invoke", "detail": str(e)}}

    raw = getattr(ai, "content", "")

    chosen = "human_fallback"
    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
        nxt = parsed.get("next")
        if isinstance(nxt, str) and nxt in ALLOWED:
            chosen = nxt
        else:
            logger.warning(f"[supervisor_node] invalid 'next': {nxt} raw={raw[:200]}")
            parsed = {"error": "invalid_next", "raw": raw}
    except Exception as e:
        logger.warning(f"[supervisor_node] invalid JSON: {e} raw={raw[:200]}")
        parsed = {"error": "invalid_json", "raw": raw}

    return {"next": chosen, "router_json": parsed}
