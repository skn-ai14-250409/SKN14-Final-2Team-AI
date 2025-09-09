# --- stdlib ---
import json
import re

# --- third-party ---
import requests
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

# --- langchain ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
# from config import llm, index, naver_client_id, naver_client_secret
from .config import llm, index, naver_client_id, naver_client_secret

###yyh
from pathlib import Path
import joblib, json

# tools.py 파일이 있는 폴더를 기준으로 경로 설정
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models.pkl"
DEFAULT_PERFUME_JSON = BASE_DIR / "perfumes.json"
###yyh

parse_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 향수 쿼리 파서야.
사용자의 질문에서 다음 정보를 JSON 형식으로 추출해줘:
- brand: 브랜드명 (예: 샤넬, 디올, 입생로랑 등)
- concentration: (퍼퓸, 코롱 등)
- day_night_score: 사용시간 (주간, 야간, 데일리 등)
- gender: 성별 (남성, 여성, 유니섹스)
- season_score: 계절 (봄, 여름, 가을, 겨울)
- sizes: 용량 (30ml, 50ml, 100ml 등) 단위는 무시하고 숫자만

없는 값은 null로 두고, 반드시 유효한 JSON 형식으로만 응답해줘.

예시:
{{"brand": "샤넬", "gender": null, "sizes": "50", "season_score": null, "concentration": null, "day_night_score": null}}"""),
    ("user", "{query}")
])

def run_llm_parser(query: str):
    """사용자 쿼리를 JSON으로 파싱"""
    try:
        chain = parse_prompt | llm
        ai_response = chain.invoke({"query": query})
        response_text = ai_response.content.strip()

        # JSON 부분만 추출
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].strip()

        parsed = json.loads(response_text)
        return parsed
    except Exception as e:
        return {"error": f"파싱 오류: {str(e)}"}

# 메타필터 함수들
def filter_brand(brand_value):
    valid_brands = [
        '겔랑', '구찌', '끌로에', '나르시소 로드리게즈', '니샤네', '도르세', '디올', '딥티크', '랑콤',
        '로라 메르시에', '로에베', '록시땅', '르 라보', '메모', '메종 마르지엘라', '메종 프란시스 커정',
        '멜린앤게츠', '미우미우', '바이레도', '반클리프 아펠', '버버리', '베르사체', '불가리', '비디케이',
        '산타 마리아 노벨라', '샤넬', '세르주 루텐', '시슬리 코스메틱', '아쿠아 디 파르마', '에따 리브르 도량쥬',
        '에르메스', '에스티 로더', '엑스 니힐로', '이니시오 퍼퓸', '이솝', '입생로랑', '제르조프', '조 말론',
        '조르지오 아르마니', '줄리엣 헤즈 어 건', '지방시', '질 스튜어트', '크리드', '킬리안', '톰 포드',
        '티파니앤코', '퍼퓸 드 말리', '펜할리곤스', '프라다', '프레데릭 말'
    ]
    if brand_value is None:
        return None
    return brand_value if brand_value in valid_brands else None

def filter_concentration(concentration_value):
    valid_concentrations = ['솔리드 퍼퓸', '엑스트레 드 퍼퓸', '오 드 뚜왈렛', '오 드 코롱', '오 드 퍼퓸', '퍼퓸']
    if concentration_value is None:
        return None
    return concentration_value if concentration_value in valid_concentrations else None

def filter_day_night_score(day_night_value):
    valid_day_night = ["day", "night"]
    if day_night_value is None:
        return None
    if isinstance(day_night_value, str) and ',' in day_night_value:
        values = [v.strip() for v in day_night_value.split(',')]
        filtered_values = [v for v in values if v in valid_day_night]
        return ','.join(filtered_values) if filtered_values else None
    return day_night_value if day_night_value in valid_day_night else None

def filter_gender(gender_value):
    valid_genders = ['Female', 'Male', 'Unisex', 'unisex ']
    if gender_value is None:
        return None
    return gender_value if gender_value in valid_genders else None

def filter_season_score(season_value):
    valid_seasons = ['winter', 'spring', 'summer', 'fall']
    if season_value is None:
        return None
    return season_value if season_value in valid_seasons else None

def filter_sizes(sizes_value):
    valid_sizes = ['30', '50', '75', '100', '150']
    if sizes_value is None:
        return None
    if isinstance(sizes_value, str):
        numbers = re.findall(r'\d+', sizes_value)
        for num in numbers:
            if num in valid_sizes:
                return num
    return str(sizes_value) if str(sizes_value) in valid_sizes else None

def apply_meta_filters(parsed_json: dict) -> dict:
    """파싱된 JSON에 메타price링 적용"""
    if not parsed_json or "error" in parsed_json:
        return parsed_json
    
    return {
        'brand': filter_brand(parsed_json.get('brand')),
        'concentration': filter_concentration(parsed_json.get('concentration')),
        'day_night_score': filter_day_night_score(parsed_json.get('day_night_score')),
        'gender': filter_gender(parsed_json.get('gender')),
        'season_score': filter_season_score(parsed_json.get('season_score')),
        'sizes': filter_sizes(parsed_json.get('sizes'))
    }

def build_pinecone_filter(filtered_json: dict) -> dict:
    """메타price링 결과를 Pinecone filter dict로 변환"""
    pinecone_filter = {}
    if filtered_json.get("brand"):
        pinecone_filter["brand"] = {"$eq": filtered_json["brand"]}
    if filtered_json.get("sizes"):
        pinecone_filter["sizes"] = {"$eq": filtered_json["sizes"]}
    if filtered_json.get("season_score"):
        pinecone_filter["season_score"] = {"$eq": filtered_json["season_score"]}
    if filtered_json.get("gender"):
        pinecone_filter["gender"] = {"$eq": filtered_json["gender"]}
    if filtered_json.get("concentration"):
        pinecone_filter["concentration"] = {"$eq": filtered_json["concentration"]}
    if filtered_json.get("day_night_score"):
        pinecone_filter["day_night_score"] = {"$eq": filtered_json["day_night_score"]}
    return pinecone_filter

# 키워드 정제용 프롬프트 템플릿 추가
keyword_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 네이버 쇼핑 검색용 키워드 추출 전문가야.

사용자의 향수 가격 질문에서 네이버 쇼핑 API 검색에 최적화된 키워드만 추출해줘.

**추출 규칙:**
1. 브랜드명과 제품명만 추출 (한국어 우선)
2. 가격 관련 단어들은 모두 제거 (가격, 얼마, 최저가, 할인, 어디서, 사는, 구매 등)
3. 질문 형태 단어들도 제거 (?, 야, 게, 줘, 알려, 뭐, 어떤 등)
4. 최대 2-3개 단어로 간결하게
5. 향수 브랜드명과 제품명이 명확하면 "브랜드 제품명" 형태로
6. 브랜드명만 있으면 "브랜드명" 만
7. 애매하면 "향수" 키워드 사용

**예시:**
- "디올 소바쥬 가격 얼마야?" → "디올 소바쥬"  
- "샤넬 넘버5 50ml 어디서 사?" → "샤넬 넘버5"
- "톰포드 향수 최저가 알려줘" → "톰포드 향수"
- "향수 가격 알려줘" → "향수"

반드시 검색 키워드만 응답하고 다른 설명은 하지마."""),
    ("user", "{query}")
])

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


@tool
def recommend_perfume_simple(
    user_text: str,
    topk_labels: int = 4,
    top_n_perfumes: int = 5,
    use_thresholds: bool = True,
    # model_pkl_path: str = "./models.pkl",
    # perfume_json_path: str = "perfumes.json",
    model_pkl_path: str = str(DEFAULT_MODEL_PATH),      # 수정
    perfume_json_path: str = str(DEFAULT_PERFUME_JSON), # 수정

    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    max_len: int = 256,
):
    """
    Minimal one-shot perfume recommender (no caching).
    Loads model & data every call, predicts labels, retrieves with BM25, and returns a JSON-serializable dict.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load ML bundle
    data = joblib.load(model_pkl_path)
    clf = data["classifier"]
    mlb = data["mlb"]
    thresholds = data.get("thresholds", {}) or {}

    # 2) Encoder
    tok = AutoTokenizer.from_pretrained(model_name)
    enc_model = AutoModel.from_pretrained(model_name).to(device)
    enc_model.eval()

    # 3) Load perfumes
    with open(perfume_json_path, "r", encoding="utf-8") as f:
        perfumes = json.load(f)
        if not isinstance(perfumes, list):
            raise ValueError("perfumes.json must contain a list of perfume objects")

    # 4) BM25 index (simple: use 'fragrances' or fallback to description/name/brand)
    def doc_of(p):
        fr = p.get("fragrances")
        if isinstance(fr, list):
            text = " ".join(map(str, fr))
        elif isinstance(fr, str):
            text = fr
        else:
            text = " ".join(
                str(x)
                for x in [
                    p.get("description", ""),
                    p.get("main_accords", ""),
                    p.get("name_perfume") or p.get("name", ""),
                    p.get("brand", ""),
                ]
                if x
            )
        return (text or "unknown").lower()

    tokenized_corpus = [doc_of(p).split() for p in perfumes]
    bm25 = BM25Okapi(tokenized_corpus)

    # 5) Encode text -> vector
    batch = tok(
        [user_text],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = enc_model(**batch)
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy()  # ultra-simple mean

    # 6) Predict labels (supports predict_proba / decision_function / predict)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(emb)[0]
    elif hasattr(clf, "decision_function"):
        logits = np.asarray(clf.decision_function(emb)[0], dtype=float)
        proba = 1.0 / (1.0 + np.exp(-logits))
    else:
        proba = np.asarray(clf.predict(emb)[0], dtype=float)

    classes = list(mlb.classes_)
    if use_thresholds and thresholds:
        picked_idx = [i for i, p in enumerate(proba) if p >= float(thresholds.get(classes[i], 0.5))]
        if not picked_idx:
            picked_idx = np.argsort(-proba)[:topk_labels].tolist()
    else:
        picked_idx = np.argsort(-proba)[:topk_labels].tolist()

    labels = [classes[i] for i in picked_idx]

    # 7) Retrieve with BM25
    scores = bm25.get_scores(" ".join(labels).split())
    top_idx = np.argsort(scores)[-top_n_perfumes:][::-1]

    def _safe(d, *keys, default="N/A"):
        for k in keys:
            if k in d and d[k] not in (None, ""):
                return d[k]
        return default

    recs = []
    for rnk, idx in enumerate(top_idx, 1):
        p = perfumes[int(idx)]
        fr = p.get("fragrances")
        if isinstance(fr, list):
            fr_text = ", ".join(map(str, fr))
        else:
            fr_text = fr if isinstance(fr, str) else _safe(p, "main_accords", default="N/A")
        recs.append({
            "rank": int(rnk),
            "index": int(idx),
            "score": float(scores[int(idx)]),
            "brand": _safe(p, "brand"),
            "name": _safe(p, "name_perfume", "name"),
            "fragrances": fr_text,
            "perfume_data": p,  # JSON-native
        })

    return {
        "user_input": user_text,
        "predicted_labels": labels,
        "recommendations": recs,
        "meta": {
            "model_name": model_name,
            "device": device,
            "max_len": int(max_len),
            "db_size": int(len(perfumes)),
        },
    }

def query_pinecone(vector, filtered_json: dict, top_k: int = 5):
    """Pinecone 벡터 검색 + 메타데이터 price 적용"""
    pinecone_filter = build_pinecone_filter(filtered_json)
    
    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter if pinecone_filter else None
    )
    return result

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

def format_search_results(pinecone_results):
    """Pinecone 검색 결과를 텍스트로 포맷팅"""
    if not pinecone_results or not pinecone_results.get('matches'):
        return "검색된 향수가 없습니다."
    
    formatted_results = []
    for i, match in enumerate(pinecone_results['matches'], 1):
        metadata = match.get('metadata', {})
        score = match.get('score', 0)
        
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
    if search_results and search_results.get('matches'):
        top_match = search_results['matches'][0]  # 가장 유사도 높은 향수
        metadata = top_match.get('metadata', {})
        
        perfume_name = metadata.get('perfume_name', '')
        brand_name = metadata.get('brand', '')
        
        if perfume_name and brand_name:
            # "브랜드 + 향수명" 조합으로 구체적인 검색
            search_keyword = f"{brand_name} {perfume_name}"
            
            # 용량 정보 추가 (있다면)
            sizes = parsed_json.get('sizes')
            if sizes:
                search_keyword += f" {sizes}ml"
            
            return search_keyword
        
        elif perfume_name:
            # 향수명만 있는 경우
            sizes = parsed_json.get('sizes')
            if sizes:
                return f"{perfume_name} {sizes}ml"
            return perfume_name
        
        elif brand_name:
            # 브랜드명만 있는 경우
            sizes = parsed_json.get('sizes')
            if sizes:
                return f"{brand_name} 향수 {sizes}ml"
            return f"{brand_name} 향수"
    
    # 2. 검색 결과가 없거나 메타데이터가 부족한 경우 파싱 결과 사용
    brand = parsed_json.get('brand')
    sizes = parsed_json.get('sizes')
    
    if brand:
        if sizes:
            return f"{brand} 향수 {sizes}ml"
        return f"{brand} 향수"
    
    # 3. 모든 정보가 없으면 기본값
    return "향수"
