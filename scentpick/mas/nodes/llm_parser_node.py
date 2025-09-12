from langchain_core.messages import HumanMessage, AIMessage
from ..state import AgentState
from ..tools.tools_parsers import run_llm_parser
from ..tools.tools_metafilters import apply_meta_filters
from ..tools.tools_rag import query_pinecone, generate_response
import json
from ..config import llm, embeddings
from ..tools.tools_price import price_tool
from ..tools.vector_db_utils import build_item_queries_from_vectordb

def LLM_parser_node(state: AgentState) -> AgentState:
    """실제 RAG 파이프라인을 실행하는 LLM_parser 노드 + 가격 검색(벡터DB 한정) 통합"""
    user_query = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break
    if not user_query:
        user_query = "(empty)"

    try:
        print(f"🔍 LLM_parser 실행: {user_query}")
        
        # 1단계: LLM으로 쿼리 파싱
        parsed_json = run_llm_parser(user_query)
        if "error" in parsed_json:
            error_msg = f"[LLM_parser] 쿼리 파싱 오류: {parsed_json['error']}"
            msgs = state["messages"] + [AIMessage(content=error_msg)]
            return {"messages": msgs, "next": None, "router_json": state.get("router_json")}
        
        # 2단계: 메타필터 적용
        filtered_json = apply_meta_filters(parsed_json)
        
        # 3단계: 쿼리 벡터화
        query_vector = embeddings.embed_query(user_query)
        
        # 4단계: Pinecone 검색
        search_results = query_pinecone(query_vector, filtered_json, top_k=5)
        if hasattr(search_results, "to_dict"):
            search_results = search_results.to_dict()
        
        # 5단계: 최종 응답 생성
        final_response = generate_response(user_query, search_results)
        
        # 6단계: 가격 의도 감지
        price_keywords_ko = ['가격', '얼마', '가격대', '구매', '판매', '할인', '어디서 사', '어디서사', '배송비', '최저가']
        price_keywords_en = ['price', 'cost', 'cheapest', 'buy', 'purchase', 'discount']
        lower = user_query.lower()
        has_price_intent = any(k in user_query for k in price_keywords_ko) or any(k in lower for k in price_keywords_en)
        
        if has_price_intent:
            # 🔒 vectorDB에서 검색된 아이템만으로 가격 쿼리 생성
            item_query_bundles = build_item_queries_from_vectordb(
                search_results=search_results,
                facets=parsed_json,
                top_n_items=5
            )
            print("💰 가격 검색(벡터DB 한정) 대상:")
            for b in item_query_bundles:
                print(f" - {b['item_label']} :: {b['queries'][:3]}")

            price_sections = []
            for bundle in item_query_bundles:
                label = bundle["item_label"]
                queries = bundle["queries"]
                found_block = None

                for q in queries:
                    try:
                        res = price_tool.invoke({"user_query": q})
                        if res:  # 필요시 res 포맷에 맞춘 유효성 검사 추가
                            found_block = f"🔎 **{label}**\n(검색어: `{q}`)\n{res}"
                            break
                    except Exception as price_error:
                        print(f"❌ 가격 검색 오류({q}): {price_error}")
                        continue

                if found_block:
                    price_sections.append(found_block)

            if price_sections:
                final_response_with_price = f"""{final_response}

---

💰 **가격 정보 (vectorDB 추천만)**
{'\n\n'.join(price_sections)}"""
            else:
                final_response_with_price = f"""{final_response}

---

💰 **가격 정보 (vectorDB 추천만)**
🔍 벡터DB에서 추천된 제품명으로 검색했지만, 일치 결과를 찾지 못했어요.
원하시는 **제품명 + 농도 + 용량(예: 50ml)** 조합으로 다시 알려주세요."""
        else:
            final_response_with_price = final_response
        
        # 결과 요약
        summary = f"""[LLM_parser] RAG 파이프라인 완료 ✅

📊 파싱 결과: {json.dumps(parsed_json, ensure_ascii=False)}
🔍 필터링 결과: {json.dumps(filtered_json, ensure_ascii=False)}
🎯 검색된 향수 개수: {len(search_results.get('matches', []))}

💬 추천 결과:
{final_response_with_price}"""

        msgs = state["messages"] + [AIMessage(content=summary)]
        return {
            **state,
            "messages": msgs,
            "next": None,
            "router_json": state.get("router_json"),
            "parsed_slots": parsed_json,           # 파싱된 슬롯
            "search_results": search_results,      # 벡터DB 검색 결과
            "final_answer": final_response_with_price  # 최종 응답
        }
        
    except Exception as e:
        error_msg = f"[LLM_parser] RAG 파이프라인 실행 중 오류: {str(e)}"
        print(f"❌ LLM_parser 전체 오류: {e}")
        msgs = state["messages"] + [AIMessage(content=error_msg)]
        return {
            **state,
            "messages": msgs,
            "next": None,
            "router_json": state.get("router_json"),
            "parsed_slots": {},
            "search_results": {"matches": []},
            "final_answer": error_msg
        }
    
    

