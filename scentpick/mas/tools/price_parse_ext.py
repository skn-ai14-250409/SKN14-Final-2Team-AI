import re
from typing import Dict, Optional

# 기존 파서를 안전하게 가져옴 (있으면 사용, 없으면 무시)
try:
    from .price_parse import extract_budget_krw as extract_budget_legacy
except Exception:
    extract_budget_legacy = None

_NUM = r"(\d+(?:\.\d+)?)"
_SP  = r"\s*"

def _to_krw_million(val_str: str) -> int:
    # "10" -> 100000, "7.5" -> 75000
    return int(float(val_str) * 10000)

def _normalize(d: Dict) -> Dict:
    """ supervisor facets 스키마에 맞춤:
        keys: budget, budget_min, budget_max, budget_op ∈ {lte,gte,eq,approx}, currency='KRW'
    """
    if not d:
        return {}
    out = {}
    # pass-through
    for k in ("budget", "budget_min", "budget_max", "budget_op", "currency"):
        if k in d and d[k] is not None:
            out[k] = d[k]
    # 기본값 처리
    if "currency" not in out:
        out["currency"] = "KRW"
    # budget_op 보정
    op = out.get("budget_op")
    if op and op not in {"lte","gte","eq","approx"}:
        # legacy가 다른 키워드로 줄 수 있으니 eq로 안전 보정
        out["budget_op"] = "eq"
    return out

def extract_budget_krw_ext(text: str) -> Dict:
    """
    확장된 예산 파서 (비파괴):
    - 1) 먼저 legacy extract_budget_krw(text) 호출 (있다면)
    - 2) 비거나 부족할 때 regex로 보강 (만원대/범위/이하/이상/미만/초과/원단위)
    - 항상 supervisor 스키마로 표준화하여 반환
    """
    # 0) legacy 먼저 시도
    if extract_budget_legacy:
        try:
            legacy = extract_budget_legacy(text)
            norm = _normalize(legacy)
            # 충분히 채워졌으면 바로 반환
            if norm.get("budget") or norm.get("budget_min") or norm.get("budget_max"):
                return norm
        except Exception:
            pass

    if not text:
        return {}
    t = text.replace(",", "").strip().lower()

    # 1) 범위: n만~m만 / n만-m만 / n만 원 ~ m만 원
    m = re.search(rf"{_NUM}{_SP}만{_SP}(?:원)?{_SP}[~\-]{_SP}{_NUM}{_SP}만{_SP}(?:원)?", t)
    if m:
        a = _to_krw_million(m.group(1))
        b = _to_krw_million(m.group(2))
        if a > b:
            a, b = b, a
        return {"budget_min": a, "budget_max": b, "budget_op": "approx", "currency": "KRW"}

    # 2) 숫자 ~ 숫자 (단위 생략) + '만' 뒤에만 있는 경우: 예) 7~9만
    m = re.search(rf"{_NUM}{_SP}[~\-]{_SP}{_NUM}{_SP}만", t)
    if m:
        a = _to_krw_million(m.group(1))
        b = _to_krw_million(m.group(2))
        if a > b:
            a, b = b, a
        return {"budget_min": a, "budget_max": b, "budget_op": "approx", "currency": "KRW"}

    # 3) '만원대' / '만 원 대'
    m = re.search(rf"{_NUM}{_SP}만{_SP}(?:원)?{_SP}대", t)
    if m:
        base = _to_krw_million(m.group(1))
        return {"budget_min": base, "budget_max": base + 9999, "budget_op": "approx", "currency": "KRW"}

    # 4) 단일 금액 + 비교어 (이하/이상/미만/초과/under/over)
    m = re.search(rf"{_NUM}{_SP}만{_SP}(?:원)?{_SP}(이하|이상|미만|초과|under|over)", t)
    if m:
        val = _to_krw_million(m.group(1))
        op  = m.group(2)
        if op in ("이하","under"):
            return {"budget": val, "budget_max": val, "budget_op": "lte", "currency": "KRW"}
        if op in ("이상","over"):
            return {"budget": val, "budget_min": val, "budget_op": "gte", "currency": "KRW"}
        if op == "미만":
            return {"budget": val, "budget_max": max(val - 1, 0), "budget_op": "lte", "currency": "KRW"}
        if op == "초과":
            return {"budget": val, "budget_min": val + 1, "budget_op": "gte", "currency": "KRW"}

    m = re.search(rf"(\d[\d]*){_SP}원{_SP}(이하|이상|미만|초과|under|over)", t)
    if m:
        val = int(m.group(1))
        op  = m.group(2)
        if op in ("이하","under"):
            return {"budget": val, "budget_max": val, "budget_op": "lte", "currency": "KRW"}
        if op in ("이상","over"):
            return {"budget": val, "budget_min": val, "budget_op": "gte", "currency": "KRW"}
        if op == "미만":
            return {"budget": val, "budget_max": max(val - 1, 0), "budget_op": "lte", "currency": "KRW"}
        if op == "초과":
            return {"budget": val, "budget_min": val + 1, "budget_op": "gte", "currency": "KRW"}

    # 5) 단일 금액만 존재 (만/원) → eq
    m = re.search(rf"{_NUM}{_SP}만{_SP}(?:원)?", t)
    if m:
        val = _to_krw_million(m.group(1))
        return {"budget": val, "budget_op": "eq", "currency": "KRW"}

    m = re.search(rf"(\d[\d]*){_SP}원", t)
    if m:
        val = int(m.group(1))
        return {"budget": val, "budget_op": "eq", "currency": "KRW"}

    return {}
