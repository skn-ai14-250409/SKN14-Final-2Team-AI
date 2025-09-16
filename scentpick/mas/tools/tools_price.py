from langchain_core.tools import tool
import requests
import re
from ..tools.tools_keywords import extract_search_keyword_with_llm
from ..config import naver_client_id, naver_client_secret

@tool
def price_tool(user_query: str) -> str:
    """네이버 쇼핑 API로 향수 가격을 조회하고 상위 3개(제목+가격만)로 요약합니다."""

    # 1) LLM으로 검색 키워드 추출
    search_keyword = extract_search_keyword_with_llm(user_query)

    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": naver_client_id,
        "X-Naver-Client-Secret": naver_client_secret,
    }
    # API에서는 넉넉히 가져오고, 실제 출력은 3개로 제한
    params = {"query": search_keyword, "display": 10, "sort": "sim"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
    except Exception as e:
        return f"❌ 요청 오류: {e}"

    if response.status_code != 200:
        return f"❌ API 오류: {response.status_code}"

    data = response.json()
    if not data or "items" not in data or len(data["items"]) == 0:
        return f"😔 '{search_keyword}'에 대한 검색 결과가 없습니다.\n💡 다른 브랜드명이나 향수명으로 다시 검색해보세요."

    # HTML 태그 제거
    def remove_html_tags(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text or "")

    # 상위 3개만 사용
    products = data["items"][:3]
    output = f"🔍 '{search_keyword}' 검색 결과(최대 3개):\n\n"

    prices = []
    for i, item in enumerate(products, 1):
        title = remove_html_tags(item.get("title", "")).strip()

        # 가격 파싱 (없거나 비정상이면 건너뜀)
        raw_lprice = item.get("lprice", "")
        price_val = None
        try:
            price_val = int(raw_lprice) if str(raw_lprice).isdigit() else None
        except Exception:
            price_val = None

        output += f"📦 {i}. {title}\n"
        if price_val is not None and price_val > 0:
            prices.append(price_val)
            output += f"   💰 가격: {price_val:,}원\n"
        else:
            output += f"   💰 가격: 확인 불가\n"
        output += "\n"  # 판매처/링크 출력 제거

    # 가격대 정보(2개 이상 있을 때만)
    if len(prices) >= 2:
        output += "💡 **가격대 정보**\n"
        output += f"   📊 검색된 가격 범위: {min(prices):,}원 ~ {max(prices):,}원\n"
        output += "   ⚠️ 정확한 가격은 각 쇼핑몰에서 확인해주세요.\n"
    elif len(prices) == 1:
        output += "💡 **참고사항**\n"
        output += "   ⚠️ 더 많은 가격 비교를 원하시면 여러 쇼핑몰을 직접 확인해보세요.\n"

    return output
