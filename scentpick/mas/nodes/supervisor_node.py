# scentpick/mas/nodes/supervisor_node.py — 복붙용 최종본
# CHANGED: system 프롬프트를 템플릿 변수로 안전 주입("{system}")
# NEW    : enforce_message_budget 훅으로 messages 윈도우링+요약 선처리
# KEEP   : JSON 파싱 방어, 미허용 next 가드(기본 human_fallback)

from typing import Dict, Any, List, Optional
import json
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from ..state import AgentState
from ..prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT
from ..config import llm
from ..tools.state_utils import enforce_message_budget

logger = logging.getLogger(__name__)

# 허용 노드 목록
ALLOWED = {
    "LLM_parser",
    "FAQ_agent",
    "human_fallback", 
    "price_agent",
    "ML_agent",
    "memory_echo",
    "rec_echo",
    "review_agent",  # <- review_agent_node가 아니라 review_agent
    "multimodal_agent",
}

def _build_rec_context(state: AgentState, max_items: int = 5) -> str:
    """최근 추천 목록을 번호와 함께 한 줄씩 반환. 없으면 '(none)'."""
    items = state.get("perfume_list") or []
    if not items:
        # rec_history에서 역순으로 스캔
        for e in reversed(state.get("rec_history") or []):
            cand = e.get("items") or []
            if cand:
                items = cand
                break
    if not items:
        return "(none)"
    lines: List[str] = []
    for i, it in enumerate(items[:max_items], 1):
        brand = (it.get("brand") or "").strip()
        name = (it.get("name") or "").strip()
        if not (brand or name):
            continue
        lines.append(f"{i}. {brand} {name}".strip())
    return "\n".join(lines) if lines else "(none)"

def supervisor_node(state: AgentState) -> AgentState:
    # NEW: 메시지 윈도우링 + 요약 선처리 (컨텍스트 경량화)
    try:
        msgs: List[BaseMessage] = state.get("messages") or []
        state["messages"] = enforce_message_budget(
            msgs=msgs,
            llm=llm,
            target_ctx_tokens=1800,  # 모델/요금제에 맞춰 조정
            keep_turns=6,            # 최근 N턴 유지
        )
    except Exception as e:
        logger.warning(f"[supervisor_node] enforce_message_budget skipped: {e}")

    # 최신 사용자 질의
    user_query = "(empty)"
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            user_query = m.content or "(empty)"
            break

    rec_context = _build_rec_context(state)
    last_agent = state.get("last_agent")

    # CHANGED: system 프롬프트를 템플릿 변수 {system}로 안전 주입
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system}"),
        ("user", "USER_QUERY:\n{query}\n\nREC_CONTEXT:\n{rec_context}\n\nLAST_AGENT:\n{last_agent}")
    ])
    chain = prompt | llm  # (권장) llm 온도는 0~0.2

    image_url = state.get("image_url")

    # 강제 가드: 이미지 있으면 무조건 multimodal_agent
    if image_url:
        return {"next": "multimodal_agent", "router_json": {"forced": True}}

    try:
        ai = chain.invoke({
            "system": SUPERVISOR_SYSTEM_PROMPT,
            "query": user_query,
            "rec_context": rec_context,
            "last_agent": last_agent,
            "image_url": state.get("image_url"),
        })
        raw = getattr(ai, "content", "") if ai is not None else ""
    except Exception as e:
        msg = f"[supervisor_node] Prompt invoke error: {e}"
        logger.error(msg)
        return {
            "next": "human_fallback",
            "router_json": {"error": "prompt_invoke", "detail": str(e)},
        }

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

    # 최종 반환: 다음 노드와 라우터 원본 JSON
    return {"next": chosen, "router_json": parsed}
