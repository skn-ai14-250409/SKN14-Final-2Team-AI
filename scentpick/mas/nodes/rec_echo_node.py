# scentpick/mas/nodes/rec_echo_node.py
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from ..state import AgentState
from ..config import llm
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
from ..prompts.rec_echo_prompt import REC_ECHO_SUMMARY_SYSTEM_PROMPT    
def _to_int_ml(v):
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).lower().replace("ml", "").strip()
        return int(float(s))
    except Exception:
        return None

def _get_last_ai_before_current_turn(state: AgentState) -> Optional[str]:
    """현재 Human 직전의 AI 메시지(=직전 답변) 찾기"""
    seen_current_human = False
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage) and not seen_current_human:
            seen_current_human = True
            continue
        if seen_current_human and isinstance(m, AIMessage):
            return m.content
    # 폴백: 가장 최근 AI
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            return m.content
    return None

def _fmt_item_line(it: Dict[str, Any], idx: int, highlight_idx: Optional[int]) -> str:
    brand = (it.get("brand") or "").strip()
    name  = (it.get("name")  or "").strip()
    size  = it.get("size")
    url   = it.get("detail_url") or it.get("url") or it.get("detailUrl")

    star = "⭐ " if (highlight_idx is not None and idx == highlight_idx) else ""
    base = f"{star}{idx}. {brand} {name}".strip()
    if size:
        base += f" {size}ml"
    if url:
        base += f" — {url}"
    return base

def _format_plain(items: List[Dict[str, Any]], highlight_idx: Optional[int] = None) -> str:
    lines = [_fmt_item_line(it, i, highlight_idx) for i, it in enumerate(items, 1)]
    return "\n".join(lines)

def _build_action_hints(items: List[Dict[str, Any]], highlight_idx: Optional[int]) -> str:
    n = len(items)
    if n == 0:
        return ""

    # 예시 인덱스 선택(친근한 팔로업 문구용)
    h = highlight_idx if (isinstance(highlight_idx, int) and 1 <= highlight_idx <= n) else 1
    other = 2 if n >= 2 and h != 2 else (1 if h != 1 else (3 if n >= 3 else 1))
    last  = n if n >= 2 and n != h else (n-1 if n >= 2 else 1)

    examples = [
        f"\"{other}번 가격은?\"",
        f"\"{last}번 노트 알려줘\"",
        f"\"{h}번이랑 {last}번 비교\""
    ]
    return "예) " + ", ".join(examples)

def _summarize_with_llm(items: List[Dict[str, Any]], last_ai: Optional[str], highlight_idx: Optional[int]) -> Optional[str]:
    """직전 AI 답변(last_ai)와 items JSON에 '있는 내용만' 기반으로 1줄 요약.
    새 정보 추측 금지. "정보 없음" 같은 부정적 라벨 출력 금지."""
    if not last_ai:
        return None

    # ✅ 프롬프트 분리본 사용
    sys = SystemMessage(content=REC_ECHO_SUMMARY_SYSTEM_PROMPT)

    user = HumanMessage(content=(
        "items(JSON):\n" + json.dumps(items[:5], ensure_ascii=False) + "\n\n" +
        "highlight_idx: " + (str(highlight_idx) if highlight_idx is not None else "null") + "\n\n" +
        "last_ai:\n" + (last_ai or "")
    ))

    out = llm.invoke([sys, user])
    txt = getattr(out, "content", "") or ""
    txt = txt.strip()
    if not txt:
        return None

    # 혹시 모를 부정 라벨 제거(더 부드럽게 표기)
    txt = txt.replace(" — 정보 없음", "").replace(" — N/A", "").replace(" — n/a", "")
    return txt
def rec_echo_node(state: AgentState) -> AgentState:
    # 0) followup 하이라이트 인덱스(선택)
    router = state.get("router_json") or {}
    ref = router.get("followup_reference") or {}
    highlight_idx = ref.get("index")
    if isinstance(highlight_idx, str) and highlight_idx.isdigit():
        highlight_idx = int(highlight_idx)
    if not isinstance(highlight_idx, int):
        highlight_idx = None

    # 1) rec_history에서 최근 추천 묶음
    history = state.get("rec_history") or []
    items: List[Dict[str, Any]] = []
    if history and (history[-1] or {}).get("items"):
        items = history[-1]["items"]

    # 2) 폴백: search_results에서 복원 (recs|candidates|matches)
    fallback_entry = None
    if not items:
        sr = state.get("search_results") or {}
        items = sr.get("recs") or sr.get("candidates") or []
        if not items:
            matches = sr.get("matches", []) or []
            recov = []
            for m in matches[:5]:
                meta = (m or {}).get("metadata", {}) or {}
                brand = meta.get("brand") or meta.get("Brand") or ""
                name  = meta.get("name")  or meta.get("Name")  or ""
                url   = meta.get("detail_url") or meta.get("url") or meta.get("detailUrl") or None
                size  = _to_int_ml(meta.get("size") or meta.get("size_ml") or meta.get("Size"))
                if name:
                    recov.append({"brand": brand, "name": name, "size": size, "detail_url": url})
            items = recov

        # 복원 성공 시 rec_history에 보정 저장할 엔트리 준비
        if items:
            fallback_entry = {"ts": datetime.now(timezone.utc).isoformat(), "source": "fallback", "items": items}

    # 3) no items → 안내
    if not items:
        ans = (
            "아직 요약할 추천 이력이 없어요.\n"
            "먼저 원하는 향수 조건을 알려주시면 추천해드리고, 그 다음에 \"방금 추천 다시 보여줘\"라고 물어보세요!"
        )
        return {"messages": [AIMessage(content=ans)], "final_answer": ans, "last_agent": "rec_echo"}

    # 4) (선택) 직전 어시스턴트 답변으로 각 항목 1줄 요약 생성
    last_ai = _get_last_ai_before_current_turn(state)
    pretty = _summarize_with_llm(items, last_ai, highlight_idx)
    if not pretty:
        # LLM 요약 실패 시 플레인 포맷
        pretty = _format_plain(items, highlight_idx)

    # 5) 헤더/추천 포커스/예시 팔로업 문구
    header = f"🔁 방금 추천드린 향수 요약 (총 {len(items)}개)\n"
    focus = ""
    if highlight_idx is not None and 1 <= highlight_idx <= len(items):
        focus_item = items[highlight_idx - 1]
        focus_name = f"{focus_item.get('brand','')} {focus_item.get('name','')}".strip()
        focus = f"\n⭐ 추천 포커스: {highlight_idx}번 {focus_name}\n"

    hints = _build_action_hints(items, highlight_idx)
    hint_block = f"\n💡 {hints}" if hints else ""

    final = header + "\n" + pretty + focus + hint_block

    ret = {
        "messages": [AIMessage(content=final)],
        "final_answer": final,
        "last_agent": "rec_echo",
    }
    # 폴백 복원으로 만든 items가 있고, 아직 history가 비어 있었다면 rec_history 보정도 함께
    if fallback_entry and not history:
        ret["rec_history"] = [fallback_entry]
    return ret
