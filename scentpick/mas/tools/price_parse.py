# scentpick/mas/utils/price_parse.py  (없으면 추가)
import re
from typing import Dict, Optional

def extract_budget_krw(text: str) -> Dict:
    """
    Returns {} 또는 { budget/budget_min/budget_max, budget_op, currency='KRW' }
    - '10만원 이하', '7만~9만', '만원대' 등을 처리
    """
    if not text:
        return {}
    t = text.strip()

    # 범위: 7만~9만
    m = re.search(r"(\d+(?:\.\d+)?)\s*만\s*[~\-]\s*(\d+(?:\.\d+)?)\s*만", t)
    if m:
        a = int(float(m.group(1))*10000)
        b = int(float(m.group(2))*10000)
        if a < b:
            return {"budget_min": a, "budget_max": b, "currency": "KRW"}

    # 단일 금액 + 이하/이상
    m = re.search(r"(\d+(?:\.\d+)?)\s*만\s*(이하|이상|미만|초과)?", t)
    if m:
        val = int(float(m.group(1))*10000)
        op = m.group(2)
        op_map = {"이하":"lte","이상":"gte","미반":"lt","초과":"gt"}
        return {"budget": val, "budget_op": op_map.get(op, "eq"), "currency": "KRW"}

    # 숫자원 패턴: 100000원 이하
    m = re.search(r"(\d[\d,]*)\s*원\s*(이하|이상|미만|초과)?", t)
    if m:
        val = int(m.group(1).replace(",", ""))
        op = m.group(2)
        op_map = {"이하":"lte","이상":"gte","미반":"lt","초과":"gt"}
        return {"budget": val, "budget_op": op_map.get(op, "eq"), "currency": "KRW"}

    # 만원대
    m = re.search(r"(\d+)\s*만원대", t)
    if m:
        base = int(m.group(1))*10000
        return {"budget_min": base, "budget_max": base+9999, "currency":"KRW"}

    return {}
