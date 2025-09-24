# scentpick/mas/nodes/review_agent_node.py
import json
import logging
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
import openai
import os

from ..state import AgentState
from ..config import llm
from ..tools.price_parse import extract_budget_krw
from ..tools.tools_price import price_tool

logger = logging.getLogger(__name__)

# API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# 파인콘 초기화
pc = Pinecone(api_key=pinecone_api_key)
REVIEW_INDEX_NAME = "review-vectordb"
PERFUME_INDEX_NAME = "perfume-vectordb"
review_index = pc.Index(REVIEW_INDEX_NAME)
perfume_index = pc.Index(PERFUME_INDEX_NAME)

# 🔽 마지노 유사도 임계값 (기존 0.7 → 0.1)
MIN_SIMILARITY_THRESHOLD = 0.1

def get_openai_embedding(text: str) -> List[float]:
    """OpenAI 임베딩 모델로 텍스트 벡터화"""
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"[get_openai_embedding] Error: {e}")
        raise

def parse_user_query(user_query: str) -> Dict[str, Any]:
    """사용자 쿼리를 향 설명과 가격대로 분리"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "사용자의 향수 질문을 JSON으로 변환하세요. "
         "향 설명은 scent_description, 가격 관련 내용은 price_query에 넣으세요. "
         "형태: {{\"scent_description\": \"...\", \"price_query\": \"...\"}}"),
        ("user", "{query}")
    ])
    try:
        chain = prompt | llm
        response = chain.invoke({"query": user_query})
        parsed = json.loads(getattr(response, "content", "{}"))
        return {
            "scent_description": parsed.get("scent_description", user_query),
            "price_query": parsed.get("price_query", "")
        }
    except Exception as e:
        logger.error(f"[parse_user_query] Error: {e}")
        return {
            "scent_description": user_query,
            "price_query": ""
        }

def search_review_vectordb(scent_description: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """review-vectordb에서 향 설명 기반 RAG 검색"""
    try:
        logger.info(f"[search_review_vectordb] Searching for: {scent_description}")
        query_embedding = get_openai_embedding(scent_description)
        results = review_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        if not results.get('matches'):
            logger.warning("[search_review_vectordb] No matches found")
            return []
        rag_results = []
        for i, match in enumerate(results['matches']):
            metadata = match.get('metadata', {})
            logger.info(f"[search_review_vectordb] Match {i}: score={match.get('score')}, metadata={metadata}")
            rag_results.append({
                'content': metadata.get('content', ''),
                'brand': metadata.get('brand', ''),
                'name': metadata.get('name', ''),
                'score': match.get('score', 0.0),
                'metadata': metadata
            })
        return rag_results
    except Exception as e:
        logger.error(f"[search_review_vectordb] Error: {e}")
        return []

def analyze_rag_results(scent_description: str, rag_results: List[Dict]) -> Dict[str, Any]:
    """RAG 결과를 분석해서 향의 특성 추출"""
    if not rag_results:
        return {"analyzed_scent": scent_description, "confidence": 0.0}
    
    rag_text = ""
    for i, result in enumerate(rag_results, 1):
        rag_text += f"{i}. {result.get('brand','')} {result.get('name','')}: {result.get('content','')}\n"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """다음은 향수 리뷰 데이터베이스에서 검색된 결과입니다.
사용자가 원하는 향의 특성을 분석해서 JSON으로 반환해주세요.

형태:
{{
    "analyzed_scent": "분석된 향의 특성 (구체적이고 상세하게)",
    "confidence": 0.0~1.0
}}

반드시 JSON 형태로만 응답하세요."""), 
        ("user", "사용자 요청: {scent_description}\n\n검색 결과:\n{rag_text}")
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({
            "scent_description": scent_description,
            "rag_text": rag_text
        })
        parsed = json.loads(getattr(response, "content", "{}"))
        return {
            "analyzed_scent": parsed.get("analyzed_scent", scent_description),
            "confidence": float(parsed.get("confidence", 0.5))
        }
    except Exception as e:
        logger.error(f"[analyze_rag_results] Error: {e}")
        return {"analyzed_scent": scent_description, "confidence": 0.0}

def search_perfume_vectordb(analyzed_scent: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """perfume-vectordb에서 분석된 향 특성으로 유사도 검색"""
    try:
        query_embedding = get_openai_embedding(analyzed_scent)
        results = perfume_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        perfumes = []
        for match in results['matches']:
            metadata = match.get('metadata', {})
            score = match.get('score', 0.0)
            logger.info(f"[search_perfume_vectordb] Score: {score}")
            if score < MIN_SIMILARITY_THRESHOLD:
                logger.info(f"Skipped {metadata.get('name','')} - score {score:.2f} < {MIN_SIMILARITY_THRESHOLD}")
                continue
            perfumes.append({
                'brand': metadata.get('brand', ''),
                'name': metadata.get('name', ''),
                'score': score,
                'size_ml': metadata.get('size_ml'),
                'metadata': metadata
            })
        return perfumes
    except Exception as e:
        logger.error(f"[search_perfume_vectordb] Error: {e}")
        return []

def check_prices_and_filter(perfume_list: List[Dict], budget_info: Dict) -> List[Dict]:
    """향수 리스트의 가격을 조회하고 예산 내 필터링"""
    if not perfume_list:
        return []
    
    logger.info(f"[check_prices_and_filter] Processing {len(perfume_list)} perfumes")
    logger.info(f"[check_prices_and_filter] Budget info: {budget_info}")
    
    budget_matched = []
    
    for perfume in perfume_list:
        brand = perfume.get('brand', '').strip()
        name = perfume.get('name', '').strip()
        size_ml = perfume.get('size_ml')
        
        if not (brand and name):
            logger.warning(f"[check_prices_and_filter] Skipping perfume with missing brand/name: {perfume}")
            continue
            
        logger.info(f"[check_prices_and_filter] Checking price for: {brand} {name}")
        
        try:
            # price_tool 호출 - JSON 모드로 결과 받기
            price_result = price_tool(
                user_query=f"{brand} {name}",
                brand=brand,
                name=name,
                size_ml=size_ml,
                topk_fetch=10,  # 더 많은 결과에서 찾기
                topk_return=3,  # 최저가 3개까지 확인
                return_json=True
            )
            
            logger.info(f"[check_prices_and_filter] Price result: {price_result}")
            
            # 가격 정보가 있는지 확인
            if isinstance(price_result, dict) and price_result.get('items'):
                # 가장 저렴한 가격 사용
                cheapest_item = price_result['items'][0]
                price = cheapest_item['price']
                title = cheapest_item['title']
                
                perfume['price'] = price
                perfume['price_title'] = title
                
                logger.info(f"[check_prices_and_filter] Found price {price:,}원 for {brand} {name}")
                
                # 예산 필터링
                is_within_budget = True  # 예산 정보가 없으면 모두 포함
                
                if budget_info:
                    is_within_budget = False
                    
                    if budget_info.get('budget'):
                        budget = budget_info['budget']
                        op = budget_info.get('budget_op', 'eq')
                        
                        if op == 'lte' and price <= budget:
                            is_within_budget = True
                            logger.info(f"[check_prices_and_filter] ✓ Within budget (≤{budget:,}): {price:,}")
                        elif op == 'gte' and price >= budget:
                            is_within_budget = True
                            logger.info(f"[check_prices_and_filter] ✓ Within budget (≥{budget:,}): {price:,}")
                        elif op == 'eq' and abs(price - budget) <= budget * 0.2:  # 20% 오차 허용
                            is_within_budget = True
                            logger.info(f"[check_prices_and_filter] ✓ Within budget (~{budget:,}): {price:,}")
                        else:
                            logger.info(f"[check_prices_and_filter] ✗ Not within budget {op} {budget:,}: {price:,}")
                            
                    elif budget_info.get('budget_min') and budget_info.get('budget_max'):
                        min_budget = budget_info['budget_min']
                        max_budget = budget_info['budget_max']
                        if min_budget <= price <= max_budget:
                            is_within_budget = True
                            logger.info(f"[check_prices_and_filter] ✓ Within range {min_budget:,}-{max_budget:,}: {price:,}")
                        else:
                            logger.info(f"[check_prices_and_filter] ✗ Not within range {min_budget:,}-{max_budget:,}: {price:,}")
                
                if is_within_budget:
                    budget_matched.append(perfume)
                    
            else:
                logger.warning(f"[check_prices_and_filter] No price found for {brand} {name}: {price_result}")
                # 가격을 찾지 못한 경우 예산 정보가 없으면 포함, 있으면 제외
                if not budget_info:
                    perfume['price'] = None
                    perfume['price_title'] = "가격 정보 없음"
                    budget_matched.append(perfume)
                    
        except Exception as e:
            logger.error(f"[check_prices_and_filter] Price check failed for {brand} {name}: {e}")
            # 오류 발생시 예산 정보가 없으면 포함
            if not budget_info:
                perfume['price'] = None
                perfume['price_title'] = "가격 조회 실패"
                budget_matched.append(perfume)
            continue
    
    logger.info(f"[check_prices_and_filter] Final matched: {len(budget_matched)}/{len(perfume_list)}")
    return budget_matched

def generate_perfume_response(budget_matched: List[Dict], budget_info: Dict, scent_description: str) -> str:
    """예산에 맞는 향수들의 응답 생성"""
    response_text = f"🌸 '{scent_description}' 취향에 맞는 향수를 찾았어요!\n\n"
    
    # 예산 정보 표시
    if budget_info:
        if budget_info.get('budget'):
            budget = budget_info['budget']
            op = budget_info.get('budget_op', 'eq')
            if op == 'lte':
                response_text += f"💰 예산 {budget:,}원 이하 범위 내 추천:\n\n"
            elif op == 'gte':
                response_text += f"💰 예산 {budget:,}원 이상 범위 내 추천:\n\n"
            else:
                response_text += f"💰 예산 약 {budget:,}원 범위 내 추천:\n\n"
        elif budget_info.get('budget_min') and budget_info.get('budget_max'):
            response_text += f"💰 예산 {budget_info['budget_min']:,}원~{budget_info['budget_max']:,}원 범위 내 추천:\n\n"
    
    # 향수 정보 표시 (가격순 정렬)
    sorted_perfumes = sorted(budget_matched, key=lambda x: x.get('price', float('inf')))
    
    for i, perfume in enumerate(sorted_perfumes, 1):
        brand = perfume.get('brand', '')
        name = perfume.get('name', '')
        score = perfume.get('score', 0)
        price = perfume.get('price')
        price_title = perfume.get('price_title', '')
        
        response_text += f"{i}. **{brand} {name}**\n"
        
        if price:
            response_text += f"   💰 최저가: {price:,}원\n"
            if price_title and len(price_title) > 0:
                # 제목이 너무 길면 자르기
                display_title = price_title[:50] + "..." if len(price_title) > 50 else price_title
                response_text += f"   🛒 상품: {display_title}\n"
        else:
            response_text += f"   💰 가격: {price_title}\n"
            
        response_text += f"   🎯 유사도: {score:.2f}\n\n"
    
    # 주의사항 추가
    response_text += "📝 **참고사항**\n"
    response_text += "• 가격은 변동될 수 있으니 구매 전 확인하세요\n"
    response_text += "• 향수는 개인차가 있으니 샘플 테스트를 권장합니다\n"
    
    return response_text

def generate_final_llm_response(user_query: str, scent_description: str, price_query: str) -> str:
    """조건에 맞는 향수가 없을 때 LLM이 직접 추천"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """향수 전문가로서 사용자의 요청에 맞는 향수를 추천해주세요.
구체적인 브랜드명과 제품명, 가격대, 향의 특징을 포함해서 3개 정도 추천해주세요.
친근하고 전문적인 톤으로 답변해주세요."""), 
        ("user", "전체 요청: {user_query}\n향 취향: {scent_description}\n가격대: {price_query}")
    ])
    try:
        chain = prompt | llm
        response = chain.invoke({
            "user_query": user_query,
            "scent_description": scent_description,
            "price_query": price_query
        })
        return getattr(response, "content", "")
    except Exception as e:
        logger.error(f"[generate_final_llm_response] Error: {e}")
        return "죄송합니다. 추천을 생성하는 중에 오류가 발생했습니다."

def is_review_agent_query(query: str) -> bool:
    return True

def review_agent_node(state: AgentState) -> AgentState:
    try:
        messages = state.get("messages", [])
        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content or ""
                break
        if not user_query:
            return {
                "messages": [AIMessage(content="질문을 다시 입력해주세요.")],
                "last_agent": "review_agent"
            }
        
        # 1단계: 사용자 쿼리 파싱
        parsed_query = parse_user_query(user_query)
        scent_description = parsed_query["scent_description"]
        price_query = parsed_query["price_query"]
        budget_info = extract_budget_krw(price_query) if price_query else {}
        
        # 2단계: 리뷰 데이터베이스 RAG 검색
        rag_results = search_review_vectordb(scent_description)
        if not rag_results:
            llm_response = generate_final_llm_response(user_query, scent_description, price_query)
            return {
                "messages": [AIMessage(content=llm_response)],
                "last_agent": "review_agent"
            }
        
        # 3단계: RAG 결과 분석
        analysis_result = analyze_rag_results(scent_description, rag_results)
        analyzed_scent = analysis_result["analyzed_scent"]
        
        # 4단계: 향수 데이터베이스에서 유사 향수 검색
        perfume_candidates = search_perfume_vectordb(analyzed_scent)
        if not perfume_candidates:
            llm_response = generate_final_llm_response(user_query, scent_description, price_query)
            return {
                "messages": [AIMessage(content=llm_response)],
                "last_agent": "review_agent"
            }
        
        # 5단계: 향수 리스트 정리
        perfume_list = [{
            'brand': p.get('brand', ''),
            'name': p.get('name', ''),
            'score': p.get('score', 0.0),
            'size_ml': p.get('size_ml')
        } for p in perfume_candidates]
        
        # 6단계: 가격 조회 및 예산 필터링
        budget_matched = check_prices_and_filter(perfume_list, budget_info)
        
        # 7단계: 최종 응답 생성
        if budget_matched:
            response_text = generate_perfume_response(budget_matched, budget_info, scent_description)
            return {
                "messages": [AIMessage(content=response_text)],
                "perfume_list": budget_matched,
                "last_agent": "review_agent"
            }
        else:
            # 조건에 맞는 향수가 없을 때
            if budget_info:
                if budget_info.get('budget'):
                    budget = budget_info['budget']
                    op = budget_info.get('budget_op', 'eq')
                    budget_text = f"{budget:,}원 {'이하' if op == 'lte' else '이상' if op == 'gte' else '대'}"
                else:
                    budget_text = f"{budget_info.get('budget_min', 0):,}원~{budget_info.get('budget_max', 0):,}원"
                
                fallback_msg = f"😔 '{scent_description}' 취향과 {budget_text} 예산에 맞는 향수를 찾지 못했어요.\n\n"
                fallback_msg += "💡 다음을 시도해보세요:\n"
                fallback_msg += "• 예산 범위를 조금 늘려보세요\n"
                fallback_msg += "• 향의 특성을 다르게 표현해보세요\n"
                fallback_msg += "• 구체적인 브랜드나 제품명을 알려주세요\n\n"
                
                # LLM 추천도 포함
                llm_response = generate_final_llm_response(user_query, scent_description, price_query)
                fallback_msg += f"🤖 **AI 추천**:\n{llm_response}"
                
                return {
                    "messages": [AIMessage(content=fallback_msg)],
                    "perfume_list": perfume_list,
                    "last_agent": "review_agent"
                }
            else:
                # 예산 정보 없이도 매칭 실패
                llm_response = generate_final_llm_response(user_query, scent_description, price_query)
                return {
                    "messages": [AIMessage(content=llm_response)],
                    "perfume_list": perfume_list,
                    "last_agent": "review_agent"
                }
                
    except Exception as e:
        logger.error(f"[review_agent_node] Error: {e}")
        return {
            "messages": [AIMessage(content="죄송합니다. 추천 과정에서 오류가 발생했습니다. 다시 시도해주세요.")],
            "last_agent": "review_agent",
            "last_error": str(e)
        }