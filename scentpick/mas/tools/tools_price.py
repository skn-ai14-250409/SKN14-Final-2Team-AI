
from langchain_core.tools import tool
import requests
import re
from ..tools.tools_keywords import extract_search_keyword_with_llm 
from ..config import naver_client_id, naver_client_secret
@tool
def price_tool(user_query: str) -> str:
    """A tool that uses the Naver Shopping API to look up perfume prices (results are returned as formatted strings)"""
    
    # LLM으로 검색 키워드 추출
    search_keyword = extract_search_keyword_with_llm(user_query)
    
    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": naver_client_id,
        "X-Naver-Client-Secret": naver_client_secret
    }
    params = {"query": search_keyword, "display": 5, "sort": "sim"}
    
    try:
        response = requests.get(url, headers=headers, params=params)
    except Exception as e:
        return f"❌ 요청 오류: {e}"
    
    if response.status_code != 200:
        return f"❌ API 오류: {response.status_code}"
    
    data = response.json()
    if not data or "items" not in data or len(data["items"]) == 0:
        return f"😔 '{search_keyword}'에 대한 검색 결과가 없습니다.\n💡 다른 브랜드명이나 향수명으로 다시 검색해보세요."
    
    # HTML 태그 제거 함수
    def remove_html_tags(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text)
    
    # 상위 3개만 정리
    products = data["items"][:1]
    output = f"🔍 '{search_keyword}' 검색 결과:\n\n"
    
    prices = []  # 가격 정보 수집용
    
    for i, item in enumerate(products, 1):
        title = remove_html_tags(item.get("title", ""))
        lprice = item.get("lprice", "0")
        mall = item.get("mallName", "정보 없음")
        link = item.get("link", "정보 없음")
        
        output += f"📦 {i}. {title}\n"
        if lprice != "0":
            formatted_price = f"{int(lprice):,}원"
            output += f"   💰 가격: {formatted_price}\n"
            prices.append(int(lprice))
        output += f"   🏪 판매처: {mall}\n"
        output += f"   🔗 링크: {link}\n\n"
    
    # 가격 범위 정보 추가 (최저가/최고가 대신 가격대 정보 제공)
    if prices:
        min_price = min(prices)
        max_price = max(prices)
        if len(prices) > 1:
            output += f"💡 **가격대 정보**\n"
            output += f"   📊 검색된 가격 범위: {min_price:,}원 ~ {max_price:,}원\n"
            output += f"   ⚠️ 정확한 최저가/최고가 정보는 각 쇼핑몰에서 직접 확인해주세요.\n"
        else:
            output += f"💡 **참고사항**\n"
            output += f"   ⚠️ 더 많은 가격 비교를 원하시면 여러 쇼핑몰을 직접 확인해보세요.\n"
    
    return output

