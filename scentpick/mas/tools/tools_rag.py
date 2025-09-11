# tools/rag.py

# --- stdlib ---
import json

# --- langchain ---
from langchain_core.prompts import ChatPromptTemplate

# --- local ---
from ..config import llm, index
from .tools_metafilters import build_pinecone_filter  # 필터 함수 별도 모듈로 분리했다고 가정


def query_pinecone(vector, filtered_json: dict, top_k: int = 5):
    """Pinecone 벡터 검색 + 메타데이터 필터 적용"""
    pinecone_filter = build_pinecone_filter(filtered_json)

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter if pinecone_filter else None
    )
    return result


def format_search_results(pinecone_results):
    """Pinecone 검색 결과를 텍스트로 포맷팅"""
    if not pinecone_results or not pinecone_results.get("matches"):
        return "검색된 향수가 없습니다."

    formatted_results = []
    for i, match in enumerate(pinecone_results["matches"], 1):
        metadata = match.get("metadata", {})
        score = match.get("score", 0)

        result_text = f"""
{i}. 향수명: {metadata.get('perfume_name', '정보없음')}
   - 브랜드: {metadata.get('brand', '정보없음')}
   - 성별: {metadata.get('gender', '정보없음')}
   - 용량: {metadata.get('sizes', '정보없음')}ml
   - 계절: {metadata.get('season_score', '정보없음')}
   - 사용시간: {metadata.get('day_night_score', '정보없음')}
   - 농도: {metadata.get('concentration', '정보없음')}
   - 유사도 점수: {score:.3f}
"""
        formatted_results.append(result_text.strip())

    return "\n\n".join(formatted_results)


# 🔹 RAG 응답 프롬프트
response_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 향수 전문가야. 사용자의 질문에 대해 검색된 향수 정보를 바탕으로 친절하고 전문적인 추천을 해줘.

추천할 때 다음을 포함해줘:
1. 왜 이 향수를 추천하는지
2. 향의 특징과 느낌
3. 어떤 상황에 적합한지
4. 가격대나 용량 관련 조언 (있다면)

자연스럽고 친근한 톤으로 답변해줘."""),
    ("user", """사용자 질문: {original_query}

검색된 향수 정보:
{search_results}

위 정보를 바탕으로 향수를 추천해줘.""")
])


def generate_response(original_query: str, search_results):
    """검색 결과를 바탕으로 최종 응답 생성"""
    try:
        formatted_results = format_search_results(search_results)

        chain = response_prompt | llm
        response = chain.invoke({
            "original_query": original_query,
            "search_results": formatted_results
        })

        return response.content
    except Exception as e:
        return f"응답 생성 중 오류가 발생했습니다: {str(e)}"


def extract_price_search_keywords(search_results, original_query: str, parsed_json: dict) -> str:
    """
    검색 결과에서 실제 향수 제품명을 추출하여 가격 검색 키워드로 사용
    """
    # 1. 검색 결과에서 향수명 추출 (최상위 1개)
    if search_results and search_results.get("matches"):
        top_match = search_results["matches"][0]  # 가장 유사도 높은 향수
        metadata = top_match.get("metadata", {})

        perfume_name = metadata.get("perfume_name", "")
        brand_name = metadata.get("brand", "")

        if perfume_name and brand_name:
            search_keyword = f"{brand_name} {perfume_name}"
            sizes = parsed_json.get("sizes")
            if sizes:
                search_keyword += f" {sizes}ml"
            return search_keyword

        elif perfume_name:
            sizes = parsed_json.get("sizes")
            if sizes:
                return f"{perfume_name} {sizes}ml"
            return perfume_name

        elif brand_name:
            sizes = parsed_json.get("sizes")
            if sizes:
                return f"{brand_name} 향수 {sizes}ml"
            return f"{brand_name} 향수"

    # 2. 검색 결과가 없거나 메타데이터가 부족한 경우 파싱 결과 사용
    brand = parsed_json.get("brand")
    sizes = parsed_json.get("sizes")

    if brand:
        if sizes:
            return f"{brand} 향수 {sizes}ml"
        return f"{brand} 향수"

    # 3. 모든 정보가 없으면 기본값
    return "향수"
