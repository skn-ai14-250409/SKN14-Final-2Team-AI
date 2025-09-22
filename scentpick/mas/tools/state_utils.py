# scentpick/mas/utils/memory.py
from typing import List, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import math

try:
    import tiktoken
except ImportError:
    tiktoken = None

# 대략적 토큰 카운터 (tiktoken 없으면 대략 4 chars ~= 1 token)
def _count_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken:
        # 모델명은 대충 'gpt-4o' 계열 가정. 필요시 바꿔도 OK.
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return max(1, math.ceil(len(text) / 4))

def _messages_token_len(msgs: List[BaseMessage]) -> int:
    tot = 0
    for m in msgs:
        tot += _count_tokens(getattr(m, "content", "") or "")
    return tot

def _split_old_new(msgs: List[BaseMessage], keep_turns: int = 6) -> Tuple[List[BaseMessage], List[BaseMessage]]:
    """
    최근 keep_turns 턴(휴먼/어시스턴트 쌍 기준)을 '새 메시지'로 남기고,
    그 이전은 '오래된 메시지'로 분리.
    """
    if not msgs:
        return [], []
    # 시스템 메시지는 맨 위로 유지
    sys = [m for m in msgs if isinstance(m, SystemMessage)]
    core = [m for m in msgs if not isinstance(m, SystemMessage)]

    # 휴먼/AI 페어로 뒤에서부터 카운팅
    turns = []
    cur = []
    for m in core:
        cur.append(m)
        if isinstance(m, AIMessage):
            turns.append(cur)
            cur = []
    if cur:
        turns.append(cur)

    new_turns = turns[-keep_turns:] if keep_turns > 0 else turns
    old_turns = turns[:-keep_turns] if keep_turns > 0 else []

    new_msgs = sys + [m for t in new_turns for m in t]
    old_msgs = [m for t in old_turns for m in t]
    return old_msgs, new_msgs

def build_summary_prompt(old_msgs: List[BaseMessage]) -> str:
    lines = []
    for m in old_msgs:
        role = "USER" if isinstance(m, HumanMessage) else ("ASSISTANT" if isinstance(m, AIMessage) else "SYSTEM")
        content = (m.content or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    body = "\n".join(lines[:2000])  # 안전절단
    return (
        "다음은 지금까지의 과거 대화 요약입니다. 핵심 의도/취향/제약(예산, 브랜드, 계절 등)만 간결하게 남기고, 불필요한 디테일은 버리세요.\n\n"
        f"{body}\n\n요약:"
    )

def enforce_message_budget(
    msgs: List[BaseMessage],
    llm,
    max_model_tokens: int = 8000,
    target_ctx_tokens: int = 1800,
    keep_turns: int = 6,
) -> List[BaseMessage]:
    """
    1) 최근 keep_turns 턴만 남기고
    2) 이전 대화는 1개의 SystemMessage 요약으로 축약
    3) 전체 토큰이 target_ctx_tokens를 넘으면 keep_turns를 더 줄여서 재시도
    """
    if not msgs:
        return msgs

    # 1차 분리
    old_msgs, new_msgs = _split_old_new(msgs, keep_turns=keep_turns)

    # 오래된 게 있으면 요약을 1개 시스템 메시지로 붙임
    if old_msgs:
        prompt = build_summary_prompt(old_msgs)
        summary = llm.invoke([SystemMessage(content="요약 작성자"), HumanMessage(content=prompt)])
        summary_txt = getattr(summary, "content", "").strip() or "(요약 없음)"
        new_msgs = [SystemMessage(content=f"[이전 대화 요약]\n{summary_txt}")] + new_msgs

    # 토큰 예산 체크 → 초과면 keep_turns 줄여서 재귀적으로 축소
    if _messages_token_len(new_msgs) > target_ctx_tokens and keep_turns > 2:
        return enforce_message_budget(msgs, llm, max_model_tokens, target_ctx_tokens, keep_turns=keep_turns-2)

    return new_msgs
