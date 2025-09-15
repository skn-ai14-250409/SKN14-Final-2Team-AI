# scentpick/mas/nodes/price_agent_node.py
from langchain_core.messages import HumanMessage, AIMessage
from ..state import AgentState
from ..tools.tools_price import price_tool
from typing import Any, Dict, List, Optional
import re

ORDINAL_KO = {
    "첫": 1, "첫째": 1, "첫번째": 1,
    "두": 2, "둘째": 2, "두번째": 2,
    "세": 3, "셋째": 3, "세번째": 3,
    "네": 4, "넷째": 4, "네번째": 4,
    "다섯": 5, "다섯째": 5, "다섯번째": 5,
}

def _get_latest_user_query(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content
    return "(empty)"

def _parse_ordinal_from_query(q: str) -> Optional[int]:
    if not q:
        return None
    # 숫자형 "2번", "3 번", "2번째"
    m = re.search(r"(\d+)\s*번(째)?", q)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 20:
                return n
        except Exception:
            pass
    # 한글 서수어
    for key, idx in ORDINAL_KO.items():
        if key in q:
            return idx
    return None

def _select_from_rec_history(state: AgentState, idx: Optional[int], name_hint: Optional[str]) -> Optional[Dict[str, Any]]:
    hist = state.get("rec_history") or []
    if not hist or not (hist[-1] or {}).get("items"):
        return None
    items: List[Dict[str, Any]] = hist[-1]["items"]

    # name 힌트 우선
    if name_hint:
        nh = name_hint.lower()
        for it in items:
            cand_name = f"{(it.get('brand','') or '').strip()} {(it.get('name','') or '').strip()}".strip().lower()
            if nh in cand_name:
                return it

    # 그 다음 ordinal index
    if isinstance(idx, int) and 1 <= idx <= len(items):
        return items[idx - 1]

    # 못 찾으면 None
    return None

def _to_int_ml(v) -> Optional[int]:
    try:
        if v is None: return None
        if isinstance(v, (int, float)): return int(v)
        s = str(v).lower().replace("ml", "").strip()
        return int(float(s))
    except Exception:
        return None

def _build_price_queries(item: Dict[str, Any]) -> List[str]:
    brand = (item.get("brand") or "").strip()
    name  = (item.get("name")  or "").strip()
    size  = _to_int_ml(item.get("size"))
    size_s = f"{size}ml" if size else ""

    base = f"{brand} {name}".strip()
    qs = []
    if base and size_s:
        qs += [f"{base} {size_s}", f"{base} {size_s} 가격", f"{base} {size_s} price"]
    if base:
        qs += [base, f"{base} 가격", f"{base} price", f"{base} 50ml 가격"]  # 마지막은 백업
    # 중복 제거, 공백 정리
    out, seen = [], set()
    for q in qs:
        qn = " ".join(q.split())
        if qn and qn not in seen:
            out.append(qn); seen.add(qn)
    return out[:6]

def price_agent_node(state: AgentState) -> AgentState:
    """Price agent - 추천 문맥 follow-up + 일반 가격 조회"""
    user_query = _get_latest_user_query(state)

    # 1) supervisor 라우팅 메타에서 followup 후보 정보 가져오기
    router = state.get("router_json") or {}
    ref = router.get("followup_reference") or {}
    idx = ref.get("index")
    name_hint = ref.get("name")

    # 2) 사용자가 "2번 가격?"처럼 말했는데 router가 index를 못 잡았을 경우 대비해 쿼리에서 한 번 더 파싱
    if idx is None:
        ord_from_text = _parse_ordinal_from_query(user_query)
        if ord_from_text is not None:
            idx = ord_from_text

    # 3) rec_history에서 대상 후보 선택
    target_item = _select_from_rec_history(state, idx=idx, name_hint=name_hint)

    try:
        if target_item:
            # 후보가 있으면 후보 기반으로 가격 검색
            queries = _build_price_queries(target_item)
            sections = []
            for q in queries:
                try:
                    res = price_tool.invoke({"user_query": q})
                    if res:
                        # price_tool 출력이 dict/str 어떤 형태든 문자열로
                        sections.append(f"🔎 **{q}**\n{str(res)}")
                        break  # 첫 성공에서 멈춤 (원하면 주석 처리로 다 모을 수도 있음)
                except Exception as e:
                    # 개별 쿼리 실패는 계속 진행
                    print(f"[price_agent] price_tool error for '{q}': {e}")

            if sections:
                head = f"💰 **가격 정보** — 대상: {target_item.get('brand','')} {target_item.get('name','')}" + \
                       (f" {target_item.get('size')}ml" if target_item.get('size') else "")
                final = head + "\n\n" + "\n\n".join(sections)
            else:
                final = ("💰 **가격 정보**\n\n"
                         "추천 후보명으로 검색했지만 확실한 결과를 찾지 못했어요.\n"
                         "가능하면 **정확한 제품명 + 농도(EDT/EDP 등) + 용량(예: 50ml)**로 알려주시면 더 정확해집니다.")

            return {
                "messages": [AIMessage(content=final)],
                "final_answer": final,
                "last_agent": "price_agent",
            }

        # 4) 후보 문맥이 없으면 사용자 쿼리 그대로 가격 검색
        raw = price_tool.invoke({"user_query": user_query})
        final = f"💰 **가격 정보**\n\n{str(raw)}" if raw else "💰 **가격 정보**\n\n결과를 찾지 못했어요."
        return {
            "messages": [AIMessage(content=final)],
            "final_answer": final,
            "last_agent": "price_agent",
        }

    except Exception as e:
        err = f"❌ 가격 조회 중 오류가 발생했습니다: {e}"
        return {
            "messages": [AIMessage(content=err)],
            "final_answer": err,
            "last_agent": "price_agent",
        }
