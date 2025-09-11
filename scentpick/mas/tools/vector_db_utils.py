import re
from scentpick.mas.tools.brand_utils import BRAND_ALIASES, CONC_SYNONYMS

#  벡터DB 결과에서 메타데이터 뽑고, 검색 키워드 묶음 만드는 부분

def _normalize_size(size_val):
    """'50' -> '50ml', '50 ml' -> '50ml'"""
    if not size_val:
        return None
    s = str(size_val).strip().lower().replace(" ", "")
    if s.endswith("ml"):
        return s
    if re.fullmatch(r"\d{1,4}", s):
        return s + "ml"
    return s

def _expand_brand(brand):
    if not brand:
        return []
    return BRAND_ALIASES.get(brand, [brand])

def _expand_concentration(conc):
    if not conc:
        return []
    c = str(conc)
    for k, syns in CONC_SYNONYMS.items():
        if k.replace(" ", "") in c.replace(" ", "") or c in syns:
            return syns
    return [c]

def _extract_matches(search_results: dict):
    """Pinecone matches -> list of metadata dict"""
    matches = (search_results or {}).get("matches") or []
    return [m.get("metadata") or {} for m in matches]

def _make_display_name(meta, size=None):
    brand = (meta.get("brand") or "").strip()
    name  = (meta.get("name") or "").strip()
    conc  = (meta.get("concentration") or "").strip()
    toks = [brand, name, conc, size]
    return " ".join([t for t in toks if t])

def build_item_queries_from_vectordb(
    search_results: dict,
    facets: dict | None = None,
    top_n_items: int = 5,
) -> list[dict]:
    """
    반환: [{item_label, queries}] 리스트
    - item_label: 사용자에게 보여줄 라벨(brand name conc size)
    - queries: 이 아이템만을 겨냥한 네이버 검색 후보들(문자열 리스트)
    (※ 브랜드+제품명 필수. 다른 제품으로 샐 여지를 최소화)
    """
    facets = facets or {}
    target_size = _normalize_size(facets.get("sizes"))
    metas = _extract_matches(search_results)[:top_n_items]

    results = []
    seen_items = set()  # (brand|name)로 중복 제거

    for meta in metas:
        brand = meta.get("brand")
        name  = meta.get("name")
        conc  = meta.get("concentration")
        sizes = meta.get("sizes")

        if not brand or not name:
            continue

        key = f"{brand}|{name}"
        if key in seen_items:
            continue
        seen_items.add(key)

        size_for_query = target_size
        if not size_for_query:
            if isinstance(sizes, (list, tuple)) and sizes:
                if "50" in sizes or "50ml" in sizes:
                    size_for_query = "50ml"
                else:
                    size_for_query = _normalize_size(sizes[0])

        brand_variants = _expand_brand(brand)
        conc_variants  = _expand_concentration(conc) if conc else []

        def join(*toks): return " ".join([t for t in toks if t and str(t).strip()])
        qs = []

        # A. 브랜드 + 제품명 + 농도 + 사이즈
        if brand_variants and name and conc_variants and size_for_query:
            for b in brand_variants:
                for c in conc_variants:
                    qs.append(join(b, name, c, size_for_query))

        # B. 브랜드 + 제품명 + 사이즈
        if brand_variants and name and size_for_query:
            for b in brand_variants:
                qs.append(join(b, name, size_for_query))

        # C. 브랜드 + 제품명 (백업)
        if brand_variants and name:
            for b in brand_variants:
                qs.append(join(b, name))

        # 중복 제거
        seen, deduped = set(), []
        for q in qs:
            if q not in seen:
                seen.add(q)
                deduped.append(q)

        results.append({
            "item_label": _make_display_name(meta, size_for_query),
            "queries": deduped[:6],
        })

    return results

