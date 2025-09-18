# scentpick/mas/nodes/ML_agent_node.py
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from ..state import AgentState
import json
from ..prompts.ML_agent_prompt import ML_agent_system_prompt
from ..tools.tools_recommend import recommend_perfume_vdb   # Pinecone VDB 기반 추천 도구
from ..tools.tools_parsers import run_llm_parser
from ..config import llm
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

def _to_int_ml(v) -> Optional[int]:
    try:
        if v is None: return None
        if isinstance(v, (int, float)): return int(v)
        s = str(v).lower().replace("ml", "").strip()
        return int(float(s))
    except Exception:
        return None

def _normalize_item(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """다양한 키 스키마를 표준 스키마로 정규화"""
    if not isinstance(raw, dict):
        return None
    brand = (raw.get("brand") or raw.get("Brand") or raw.get("maker") or "").strip()
    name  = (raw.get("name") or raw.get("Name") or raw.get("perfume") or raw.get("title") or "").strip()
    if not name:
        return None
    size  = _to_int_ml(raw.get("size") or raw.get("size_ml") or raw.get("ml") or raw.get("Size"))
    url   = raw.get("detail_url") or raw.get("url") or raw.get("link") or raw.get("detailUrl")
    
    pid = raw.get("no") or raw.get("perfume_id") \
          or (raw.get("perfume_data") or {}).get("no")
    try:
        pid_int = int(float(pid)) if pid is not None else None
    except Exception:
        pid_int = None

    rank = raw.get("rank")
    score = raw.get("score") or 0.0
    perfume_data = raw.get("perfume_data") or {}
    text = perfume_data.get("text") if "text" in perfume_data else raw.get("text") or ""
    
    return {
        "id": pid_int,
        "brand": brand,
        "name": name,
        "size": size,
        "detail_url": url,
        "rank": rank,
        "score": score,
        "text": text,
        }

def _extract_candidates_from_ml_result(ml_result: Any, top_n: int = 3) -> List[Dict[str, Any]]:
    """
    ml_result의 가능한 구조를 폭넓게 지원:
    - {"items":[...]}, {"top_items":[...]}, {"recommendations":[...]}, {"recs":[...]}, {"candidates":[...]}
    """
    if not isinstance(ml_result, dict):
        return []
    # 후보 리스트 찾기
    for key in ("items", "top_items", "recommendations", "recs", "candidates"):
        arr = ml_result.get(key)
        if isinstance(arr, list) and arr:
            normed = []
            for it in arr:
                n = _normalize_item(it)
                if n:
                    normed.append(n)
            if normed:
                return normed[:top_n]
    return []

def _extract_brand_from_query(query: str) -> Optional[str]:
    known_brands = ["샤넬", "디올", "구찌", "톰포드", "조말론", "입생로랑"]
    for b in known_brands:
        if b.lower() in query.lower():
            return b
    return None

def ML_agent_node(state: AgentState) -> AgentState:
    """ML agent - Pinecone VDB를 통해 상위 N개 추천 후, LLM이 설명문 생성 (멀티턴/rec_history 누적)"""
    # 0) 최신 사용자 메시지
    user_query = "(empty)"
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break

    try:
        print(f"🔍 ML_parser 실행: {user_query}")

        parsed_json = run_llm_parser(user_query)
        brand = parsed_json.get("brand")
        n_recs = int(parsed_json.get("recommendation_count") or 3)  # 기본값 3

        params = {
            "user_text": user_query,
            "topk_labels": 3,
            "top_n_perfumes": n_recs,
            "use_thresholds": True,
            "alpha_labels": 0.8,
            "index_name": "perfume-vectordb2",
        }

        if brand:
            params["metadata_filter"] = {"brand": brand}

        # 1) VDB 기반 추천
        ml_result = recommend_perfume_vdb.invoke(params)

        # 2) 후보 표준화 (rec_echo가 바로 읽을 수 있게)
        candidates = _extract_candidates_from_ml_result(ml_result, top_n=n_recs)

        # 3) LLM 설명 생성 (시스템+휴먼 메시지)
        ml_json_str = json.dumps(ml_result, ensure_ascii=False)
        human_prompt = (
            f"사용자 질문:\n{user_query}\n\n"
            f"ML 추천 JSON:\n```json\n{ml_json_str}\n```"
        )
        llm_out = llm.invoke([
            SystemMessage(content=ML_agent_system_prompt),
            HumanMessage(content=human_prompt)
        ])
        explanation = getattr(llm_out, "content", "").strip()

        # 4) 사용자에게 보여줄 최종 텍스트 (번호 매겨진 목록 + 설명)
        if candidates:
            lines = []
            for i, r in enumerate(candidates, 1):
                line = f"{i}. {r.get('brand','')} {r.get('name','')}"
                if r.get("size"):
                    line += f" {r['size']}ml"
                lines.append(line)
            header = "💬 추천 결과:\n\n" + "\n".join(lines)
            final_answer = header + ("\n\n" + explanation if explanation else "")
        else:
            final_answer = "추천 결과를 찾지 못했어요." + ("\n\n" + explanation if explanation else "")

        # 5) rec_history에 누적 (모드 A: 다음 턴 '방금 추천?' 회상용)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": "ML_agent",
            "items": candidates,  # 표준 스키마
        }

        # perfume_list: id, brand, name만 추출
        perfume_list = []
        for r in candidates:
            if r.get("id") and r.get("name"):
                perfume_list.append({
                    "id": r.get("id"),
                    "brand": r.get("brand"),
                    "name": r.get("name"),
                    "rank": r.get("rank"),
                    "score": r.get("score"),
                    "text": r.get("text"),
                })

        # 6) 델타 메시지만 반환
        return {
            "messages": [AIMessage(content=final_answer)],
            "final_answer": final_answer,
            "rec_history": [entry],        # 리스트 누적
            "last_agent": "ML_agent",      # supervisor 프롬프트용
            "perfume_list": perfume_list,
        }

    except Exception as e:
        err = f"❌ ML 추천 생성 중 오류가 발생했습니다: {e}"
        return {
            "messages": [AIMessage(content=err)],
            "final_answer": err,
            "last_agent": "ML_agent",
            "perfume_list": [],
        }
