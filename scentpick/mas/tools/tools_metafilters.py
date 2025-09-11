# 메타필터 함수들
import re

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
