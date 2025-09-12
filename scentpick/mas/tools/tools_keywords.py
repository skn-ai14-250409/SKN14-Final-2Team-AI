from ..config import llm
from ..prompts.price_prompt import keyword_extraction_prompt

def extract_search_keyword_with_llm(user_query: str) -> str:
    """LLM을 사용해서 검색 키워드 추출"""
    try:
        chain = keyword_extraction_prompt | llm
        response = chain.invoke({"query": user_query})
        keyword = response.content.strip()
        
        # 빈 응답이거나 너무 긴 경우 기본값 반환
        if not keyword or len(keyword) > 20:
            return "향수"
        
        return keyword
    except Exception as e:
        print(f"키워드 추출 오류: {e}")
        return "향수"  # 오류 시 기본값
