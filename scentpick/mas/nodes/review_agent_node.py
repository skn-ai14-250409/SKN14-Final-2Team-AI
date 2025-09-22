# scentpick/mas/nodes/review_agent_node.py
from __future__ import annotations

import re
import json
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage, AIMessage

from ..state import AgentState
from ..config import embeddings
from ..tools.tools_parsers import run_llm_parser
from ..tools.tools_metafilters import apply_meta_filters
from ..tools.tools_rag import query_pinecone, generate_response
from ..tools.tools_price import price_tool
from ..tools.vector_db_utils import build_item_queries_from_vectordb
from ..prompts.review_agent_prompt import review_agent_system_prompt  # (가격 섹션 헤더에만 사용 가능)
from ..tools.price_parse import extract_budget_krw  # 가격 파서

INDEX_NAME = "review-vectordb"   # 향설명 인덱스 고정

# ──────────────────────────────────────────────────────────────────────────────
# (라우팅) "향설명 + 가격" 쿼리 감지 ⇒ 외부 라우터에서 is_review_agent_query()로 강제 라우팅
# ──────────────────────────────────────────────────────────────────────────────
# 가격/향 키워드
_REVIEW_PRICE_RE = re.compile(r"(가격|금액|원|만원|예산|이하|이상)")
_REVIEW_SCENT_RE = re.compile(r"(향|향수|노트|향기|시트러스|우디|플로럴|히노끼|히노키|머스크|앰버)", re.IGNORECASE)

# 브랜드/제품명 키워드
_BRAND_OR_PRODUCT_RE = re.compile(
    r"(샤넬|Chanel|디올|Dior|구찌|Gucci|톰포드|Tom\s*Ford|조말론|Jo\s*Malone|입생로랑|YSL|"
    r"랑방|Lanvin|겔랑|Guerlain|버버리|Burberry|CK|씨케이|CK\s*one|CK원|"
    r"바이레도|Byredo|르라보|Le\s*Labo|딥티크|딥디크|Diptyque|아쿠아디파르마|Acqua\s*di\s*Parma|"
    r"펜할리곤스|Penhaligon|프라다|Prada|캘빈클라인|Calvin\s*Klein|"
    r"EDP|EDT|Extrait|Eau\s*de\s*(Parfum|Toilette)|오드(퍼퓸|뚜왈렛)|퍼퓸|토일렛)",
    re.IGNORECASE,
)


def is_review_agent_query(text: str) -> bool:
    if not text:
        return False
    t = str(text)

    has_price = bool(_REVIEW_PRICE_RE.search(t))
    has_scent = bool(_REVIEW_SCENT_RE.search(t))
    has_brand_or_product = bool(_BRAND_OR_PRODUCT_RE.search(t))

    return has_price and has_scent and not has_brand_or_product

# ──────────────────────────────────────────────────────────────────────────────
# helpers (LLM_parser_node와 동일 스타일)
# ──────────────────────────────────────────────────────────────────────────────
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

def _extract_candidates(search_results: dict, preferred_size=None, top_n=5):
    matches = (search_results or {}).get("matches", []) or []
    items = []
    for m in matches[: top_n * 2]:
        meta = (m or {}).get("metadata", {}) or {}
        brand = meta.get("brand") or meta.get("Brand") or ""
        name  = meta.get("name")  or meta.get("Name")  or ""
        url   = meta.get("detail_url") or meta.get("url") or meta.get("detailUrl") or None
        size  = _to_int_ml(meta.get("size") or meta.get("size_ml") or meta.get("Size"))
        if name:
            items.append({"brand": brand, "name": name, "size": size, "detail_url": url})

    if preferred_size is not None:
        ps = _to_int_ml(preferred_size)
        filtered = [it for it in items if it.get("size") == ps]
        if filtered:
            items = filtered

    return items[:top_n]

# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def review_agent_node(state: AgentState) -> AgentState:
    """
    흐름(LLM_parser_node와 동일 철학):
    1) run_llm_parser → parsed_json
    2) apply_meta_filters(parsed_json)  # 시그니처 동일
    3) embeddings.embed_query(user_query) → query_pinecone(vector, filtered_json, top_k)
    4) 후보 추려서 generate_response(user_query, search_results, limit=n) 로 본문 생성
    5) 가격 의도면 price_tool로 가격 섹션 추가
    """
    try:
        # STEP 0) 최신 사용자 메시지
        user_query = "(empty)"
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                user_query = m.content or "(empty)"
                break
        print(f"[STEP 0] ✅ review_agent 선택됨: query='{user_query}'", flush=True)

        # STEP 0-1) 가격 파서(규칙 기반) — LLM 오인식 보호 가드에만 사용
        budget_info = extract_budget_krw(user_query) or {}
        budget_display = None
        if budget_info.get("budget") is not None:
            budget_display = budget_info["budget"]
        elif budget_info.get("budget_max") is not None:
            budget_display = budget_info["budget_max"]
        if budget_display:
            print(f"[STEP 0] 🔍 budget <= {budget_display:,} {budget_info.get('currency','KRW')}", flush=True)
        else:
            print("[STEP 0] 🔍 budget 파악 안됨", flush=True)

        # STEP 1) LLM 파싱
        print("[STEP 1] run_llm_parser 호출 시작", flush=True)
        parsed_json = run_llm_parser(user_query)
        print(f"[STEP 1] 파싱 결과: {json.dumps(parsed_json, ensure_ascii=False)}", flush=True)
        if "error" in parsed_json:
            err = f"[review_agent] 쿼리 파싱 오류: {parsed_json['error']}"
            return {"messages": [AIMessage(content=err)],
                    "parsed_slots": {},
                    "search_results": {"matches": []},
                    "final_answer": err,
                    "last_agent": "review_agent"}

        # 가격이 파악된 경우에 한해, 숫자만 들어간 size 오인식 제거
        if budget_display:
            for key in ("size", "size_ml", "sizes"):
                val = parsed_json.get(key)
                if isinstance(val, str) and val.isdigit():
                    parsed_json[key] = None
        print(f"[STEP 1] 파싱 결과(보정): {json.dumps(parsed_json, ensure_ascii=False)}", flush=True)

        # STEP 2) 메타 필터 (LLM_parser_node와 동일: 인자 1개)
        print("[STEP 2] 메타 필터 적용", flush=True)
        filtered_json = apply_meta_filters(parsed_json)

        # STEP 3) 벡터 쿼리 + Pinecone 검색 (동일)
        print(f"[STEP 3] Pinecone 검색 시작 (index='{INDEX_NAME}', vector query)", flush=True)
        query_vector = embeddings.embed_query(user_query)
        n_recs = int(parsed_json.get("recommendation_count") or 3)
        search_results = query_pinecone(query_vector, filtered_json, top_k=max(10, n_recs * 5))
        if hasattr(search_results, "to_dict"):
            search_results = search_results.to_dict()
        print(f"[STEP 3] 검색 결과 수: {len((search_results or {}).get('matches', []) or [])}", flush=True)

        # STEP 4) 후보 추림 (표준 스키마)
        preferred_size = parsed_json.get("sizes") or parsed_json.get("size") or parsed_json.get("size_ml")
        candidates = _extract_candidates(search_results, preferred_size=preferred_size, top_n=n_recs)
        print(f"[STEP 4] 후보 {len(candidates)}개: {candidates}", flush=True)

        # STEP 5) 본문 생성 — 다른 에이전트와 동일 시그니처
        print("[STEP 5] 베이스 응답 생성", flush=True)
        final_response = generate_response(user_query, search_results, limit=n_recs)

        # STEP 6) 가격 섹션 — 가격 의도 또는 예산 텍스트가 있을 때만
        price_keywords_ko = ['가격', '얼마', '가격대', '구매', '판매', '할인', '어디서 사', '어디서사', '배송비', '최저가', '이하', '이상', '예산']
        price_keywords_en = ['price', 'cost', 'cheapest', 'buy', 'purchase', 'discount']
        lower = user_query.lower()
        has_price_intent = bool(budget_display) or any(k in user_query for k in price_keywords_ko) or any(k in lower for k in price_keywords_en)

        if has_price_intent and candidates:
            bundles = build_item_queries_from_vectordb(
                search_results=search_results,
                facets=parsed_json,
                top_n_items=min(n_recs, len(candidates))
            )
            price_sections = []
            for bundle in bundles:
                label = bundle["item_label"]
                for q in bundle["queries"]:
                    try:
                        res = price_tool.invoke({"user_query": q})
                        if res:
                            if budget_display:
                                price_sections.append(f"**{label}**  *(예산 ≤ {budget_display:,}원)*\n{res}")
                            else:
                                price_sections.append(f"**{label}**\n{res}")
                            break
                    except Exception as e:
                        print(f"[STEP 6] ❌ 가격 검색 오류({q}): {e}", flush=True)

            if price_sections:
                final_response = f"""{final_response}

---

💰 **가격 정보**{f" *(≤ {budget_display:,}원)*" if budget_display else ""}

{chr(10).join(price_sections)}
"""

        # STEP 7) 반환 (다른 에이전트와 동일 필드)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": "review_agent",
            "items": candidates,
        }
        return {
            "messages": [AIMessage(content=final_response)],
            "parsed_slots": parsed_json,
            "search_results": search_results,
            "final_answer": final_response,
            "rec_history": [entry],
            "last_agent": "review_agent",
        }

    except Exception as e:
        err = f"[review_agent] 실행 중 오류: {e}"
        print(err, flush=True)
        return {
            "messages": [AIMessage(content=err)],
            "parsed_slots": {},
            "search_results": {"matches": []},
            "final_answer": err,
            "last_agent": "review_agent",
        }