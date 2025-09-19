# scentpick/mas/nodes/price_agent_node.py
from langchain_core.messages import HumanMessage, AIMessage
from typing import Any, Dict, List, Optional, Tuple
import re
import unicodedata
import logging

from ..state import AgentState
from ..tools.tools_price import price_tool
from ..tools.brand_utils import BRAND_ALIASES, BRAND_LIST, CONC_SYNONYMS

log = logging.getLogger(__name__)

# ---------- 공통 유틸 ----------
def _latest_items(state: AgentState) -> List[Dict[str, Any]]:
    items = state.get("perfume_list") or []
    if items:
        return items
    hist = state.get("rec_history") or []
    for e in reversed(hist):
        it = e.get("items") or []
        if it:
            return it
    return []

def _to_int_ml(v) -> Optional[int]:
    try:
        if v is None: return None
        if isinstance(v, (int, float)): return int(v)
        s = str(v).lower().replace("ml", "").strip()
        return int(float(s))
    except Exception:
        return None

# 한글 서수/숫자 파싱
_ORD_KO = {"첫":1,"첫째":1,"첫번째":1,"두":2,"둘째":2,"두번째":2,"세":3,"셋째":3,"세번째":3}
def _ordinal_from_text(q: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*번(째)?", q or "")
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 20: return n
        except: pass
    for k, i in _ORD_KO.items():
        if k in (q or ""): return i
    return None

def _router_ref(state: AgentState):
    r = state.get("router_json") or {}
    return (r.get("followup_reference") or {}).get("index"), (r.get("followup_reference") or {}).get("name")

def _extract_budget_from_text(text: str) -> Optional[int]:
    """간단 추출: '10만원 이하', '100,000원 이하' 등에서 상한값."""
    if not text: return None
    m = re.search(r"(\d+)\s*만\s*원?\s*(이하|under)?", text)
    if m:
        return int(m.group(1)) * 10000
    m = re.search(r"(\d[\d,]*)\s*원\s*(이하|under)?", text)
    if m:
        try: return int(m.group(1).replace(",", ""))
        except: return None
    return None

def _fmt_krw(v: Optional[int]) -> str:
    if v is None: return ""
    return f"{int(v):,}원"

# ---------- brand_utils 기반 단일 쿼리 파서 ----------
_STOP_TOKENS = {
    "가격","얼마","알려줘","문의","최저가","구매","사줘","링크","추천","향수",
    "가격대","리뷰","후기","세일","할인","쿠폰","공식","공홈","정품","정가","auth","공식몰",
}

def _flatten_conc_synonyms() -> set:
    s = set()
    for _, arr in (CONC_SYNONYMS or {}).items():
        for a in arr:
            s.add(a.lower())
    return s

_CONC_STOP = _flatten_conc_synonyms()

def _deaccent(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))

def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def _alias_index() -> List[Tuple[str, str, re.Pattern]]:
    """
    BRAND_ALIASES를 (canonical, alias, compiled_pattern) 리스트로 변환.
    긴 alias부터 매칭. 공백/따옴표/점/&/and 등 변형 허용.
    """
    idx: List[Tuple[str, str, re.Pattern]] = []
    for canonical, aliases in (BRAND_ALIASES or {}).items():
        for alias in aliases:
            a = _deaccent(alias)
            a = a.replace(".", r"\.?").replace("&", r"(?:&|and)")
            a = re.sub(r"\s+", r"\\s*", a)
            a = a.replace("'", "[’'`]")
            patt = re.compile(a, flags=re.IGNORECASE)
            idx.append((canonical, alias, patt))
    idx.sort(key=lambda x: -len(x[1]))
    return idx

_ALIAS_IDX = _alias_index()

def _normalize_name_tokens(text: str) -> str:
    """
    제품명 정리:
    - 가격/요청 불용어, 농도 토큰 제거
    - Chanel No.5 표기 보정: no5/no.5/n°5/넘버5 → 'No 5'
    - 50ml 같은 용량 토큰은 유지
    """
    s = _deaccent(text)
    s = _norm_space(s)
    s = re.sub(r"\b(n[oº°]|n°)\.?\s*0*5\b", "No 5", s, flags=re.IGNORECASE)
    s = re.sub(r"\b넘버\s*0*5\b", "No 5", s, flags=re.IGNORECASE)

    toks = s.split()
    out = []
    stop_lower = {t.lower() for t in _STOP_TOKENS}
    for w in toks:
        lw = w.lower()
        if lw in stop_lower:
            continue
        if lw in _CONC_STOP:
            continue
        out.append(w)
    return " ".join(out[:5]).strip()

def _extract_brand_name_from_query(q: str) -> Optional[Tuple[str, str]]:
    """brand_utils의 BRAND_ALIASES/BRAND_LIST로 (브랜드, 제품명) 추출."""
    if not q:
        return None
    text = _norm_space(q)
    text_de = _deaccent(text)

    # 1) alias 우선 매칭
    for canonical, alias_raw, patt in _ALIAS_IDX:
        m = patt.search(text_de)
        if not m:
            continue
        brand_std = canonical
        start, end = m.span()

        def strip_noise(s: str) -> str:
            s = re.sub(r"(가격|얼마|알려줘|문의|최저가|구매|사줘|링크|추천|향수)", " ", s, flags=re.IGNORECASE)
            return _norm_space(s)

        tail = strip_noise(text_de[end:].strip())
        head = strip_noise(text_de[:start].strip())
        cand = tail or head
        name_std = _normalize_name_tokens(cand)
        return (brand_std, name_std)

    # 2) 실패 시 BRAND_LIST 보조 매칭
    for canonical in (BRAND_LIST or []):
        patt = re.compile(re.sub(r"\s+", r"\\s*", _deaccent(canonical)), flags=re.IGNORECASE)
        m = patt.search(text_de)
        if not m:
            continue
        start, end = m.span()
        brand_std = canonical
        tail = _norm_space(text_de[end:])
        head = _norm_space(text_de[:start])
        tail = re.sub(r"(가격|얼마|알려줘|문의|최저가|구매|사줘|링크|추천|향수)", " ", tail, flags=re.IGNORECASE)
        head = re.sub(r"(가격|얼마|알려줘|문의|최저가|구매|사줘|링크|추천|향수)", " ", head, flags=re.IGNORECASE)
        cand = _norm_space(tail or head)
        name_std = _normalize_name_tokens(cand)
        return (brand_std, name_std)

    return None

def _extract_size_hint(q: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*ml", q or "", flags=re.IGNORECASE)
    if not m: return None
    try: return int(m.group(1))
    except: return None

def _no5_regex_if_needed(name: str) -> Optional[str]:
    if not name:
        return None
    n = name.lower()
    if "no 5" in n or "no5" in n or "넘버 5" in n or "넘버5" in n or "n°5" in n or "nº5" in n:
        return r"(?:\bno\.?\s*5\b|\bn[oº°]\.?\s*5\b|\bn°\s*5\b|\b넘버\s*5\b)"
    return None

def _brands_in_items(items: List[Dict[str, Any]]) -> set:
    out = set()
    for it in items:
        b = (it.get("brand") or "").strip().lower()
        if b: out.add(b)
    return out

# ---------- 본체 ----------
def price_agent_node(state: AgentState) -> AgentState:
    """
    - 추천 목록이 없으면: LLM 키워드 우선 검색 → (필요 시) 이름 정규식 보정 재시도
    - 추천 목록이 있더라도, 유저가 명시적 브랜드/제품을 제시했고 그 브랜드가 목록과 다르면: 단일 쿼리 경로로 강제
    - 추천 목록이 있고 동일 브랜드 컨텍스트면: 후보별 1회 정밀 조회(topk_return=1, return_json=True)
    - 예산 이내 결과 0건이면 동적 예산 안내
    - 모든 응답 하단에 가격 비교 주의 문구 추가
    """
    user_q = ""
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            user_q = m.content or ""
            break

    # 예산: parsed_slots → 텍스트 추출
    slots = state.get("parsed_slots") or {}
    budget = slots.get("budget")
    if budget is None and slots.get("budget_min") and slots.get("budget_max"):
        budget = slots["budget_max"]
    if budget is None:
        budget = _extract_budget_from_text(user_q)

    # 최신 추천 후보
    items = _latest_items(state)

    # 명시적 브랜드/제품 추출 (추천목록 무시 여부 판단용)
    explicit_bn = _extract_brand_name_from_query(user_q)
    size_hint = _extract_size_hint(user_q)
    name_regex = _no5_regex_if_needed(explicit_bn[1]) if explicit_bn else None

    # ----- 오버라이드 가드 -----
    # 추천목록이 있어도, 유저가 제시한 브랜드가 목록 내 브랜드들과 다르면 단일 쿼리(A)로 강제 전환
    if items and explicit_bn:
        explicit_brand = explicit_bn[0].strip().lower()
        item_brands = _brands_in_items(items)
        if explicit_brand and explicit_brand not in item_brands:
            items = []  # 강제로 단일 쿼리 경로로 진입

    # ---------------------------
    # A) 추천 목록 없는 경우 → LLM 키워드 우선 + 정규식 보정
    # ---------------------------
    if not items:
        # 1) LLM 키워드 기반 1차 검색
        try:
            broad = price_tool.invoke({
                "user_query": user_q,
                "brand": None, "name": None,
                "size_ml": size_hint,
                "budget_krw": int(budget) if budget else None,
                "topk_return": 1,
                "return_json": True,
            }) or {}
        except Exception as e:
            err = f"❌ 가격 조회 중 오류가 발생했습니다: {e}"
            return {"messages":[AIMessage(content=err)], "final_answer": err, "last_agent":"price_agent"}

        b_items = broad.get("items") or []
        b_under = broad.get("under_budget") or []

        # 2) 이름 정규식 보정 필요하면 재호출(필터링)
        if name_regex:
            try:
                strict = price_tool.invoke({
                    "user_query": user_q,
                    "brand": None, "name": None,
                    "size_ml": size_hint,
                    "budget_krw": int(budget) if budget else None,
                    "topk_return": 3,          # 필터 효과 확보
                    "return_json": True,
                    "name_regex": name_regex,
                }) or {}
            except Exception:
                strict = {}
            s_items = strict.get("items") or []
            s_under = strict.get("under_budget") or []
            if s_items:
                b_items, b_under = s_items, s_under

        # 3) 예산 메시지/출력
        if budget is not None and not b_under and b_items:
            msg = f"현재 {_fmt_krw(budget)} 이하로 검색된 가격대가 없습니다.\n자세한 가격 비교는 여러 쇼핑몰에서 직접 확인하세요."
            return {"messages":[AIMessage(content=msg)], "final_answer": msg, "last_agent":"price_agent"}

        if b_items:
            p = b_items[0]
            header = "💰 가격 정보"
            if budget is not None:
                header += f" (≤ {_fmt_krw(budget)})"
            # ✅ 제목에 검색 결과 제목 사용
            final = f"{header}\n\n**{p['title']}**" + (f" {size_hint}ml" if size_hint else "")
            final += f"\n- 검색결과: ₩ {p['price']:,}원"
            final += "\n\n⚠️ 가격은 변동될 수 있으니 각 쇼핑몰에서 직접 확인하세요."
            return {"messages":[AIMessage(content=final)], "final_answer": final, "last_agent":"price_agent"}

        # 4) 그래도 없으면 안내
        nores = "검색 결과를 찾지 못했어요. 가능하면 **정확한 제품명 + 농도(EDT/EDP) + 용량(예: 50ml)**로 다시 알려주세요."
        return {"messages":[AIMessage(content=nores)], "final_answer": nores, "last_agent":"price_agent"}

    # ---------------------------
    # B) 추천 목록 있는 경우 → 멀티턴 흐름(후보별 1회)
    # ---------------------------
    idx_ref, _ = _router_ref(state)
    if idx_ref is None:
        idx_ref = _ordinal_from_text(user_q)

    if isinstance(idx_ref, int) and 1 <= idx_ref <= len(items):
        targets = [items[idx_ref-1]]
    else:
        targets = items[:3]  # 상위 3개만

    sections: List[str] = []
    under_budget_hits = 0

    for t in targets:
        brand = (t.get("brand") or "").strip()
        name  = (t.get("name")  or "").strip()
        size  = _to_int_ml(t.get("size") or t.get("sizes"))

        if not (brand and name):
            sections.append(f"- {brand} {name}: 조회 불가(브랜드/제품명 누락)")
            continue

        try:
            res = price_tool.invoke({
                "user_query": f"{brand} {name}",
                "brand": brand,
                "name": name,
                "size_ml": size,
                "budget_krw": int(budget) if budget else None,
                "topk_return": 1,
                "return_json": True,
            }) or {}
        except Exception as e:
            sections.append(f"**{brand} {name}**" + (f" {size}ml" if size else "") + f"\n- 오류: {e}")
            continue

        items_view = res.get("items") or []
        under = res.get("under_budget") or []

        if budget is not None and under:
            under_budget_hits += len(under)

        if items_view:
            p = items_view[0]
            # ✅ 결과 제목을 그대로 사용
            line = f"**{p['title']}**"
            line += f"\n- 검색결과: ₩ {p['price']:,}원"
            if budget is not None:
                line += "  (예산 이내)" if p["price"] <= int(budget) else "  (예산 초과)"
            sections.append(line)
        else:
            sections.append(f"**{brand} {name}**" + (f" {size}ml" if size else "") + "\n- 결과 없음")

    if budget is not None and under_budget_hits == 0 and sections:
        msg = f"현재 {_fmt_krw(budget)} 이하로 검색된 가격대가 없습니다.\n자세한 가격 비교는 여러 쇼핑몰에서 직접 확인하세요."
        return {"messages":[AIMessage(content=msg)], "final_answer": msg, "last_agent":"price_agent"}

    header = "💰 가격 정보"
    if budget is not None:
        header += f" (≤ {_fmt_krw(budget)})"
    final = header + "\n\n" + "\n\n".join(sections)
    final += "\n\n⚠️ 가격은 변동될 수 있으니 각 쇼핑몰에서 직접 확인하세요."

    return {
        "messages":[AIMessage(content=final)],
        "final_answer": final,
        "last_agent": "price_agent",
    }
