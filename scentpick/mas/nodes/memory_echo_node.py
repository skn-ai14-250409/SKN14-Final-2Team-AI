# scentpick/mas/nodes/memory_echo_node.py
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from ..state import AgentState
from ..tools.utils import get_prev_user_utterance
from ..config import llm

def _get_last_ai_before_current_turn(state: AgentState):
    """현재 Human 메시지 직전의 AI 답변을 찾아 반환 (없으면 None)"""
    seen_current_human = False
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage) and not seen_current_human:
            seen_current_human = True
            continue
        if seen_current_human and isinstance(m, AIMessage):
            return m.content
    # 폴백: 그냥 가장 최근 AI
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            return m.content
    return None

def memory_echo_node(state: AgentState) -> AgentState:
    prev = get_prev_user_utterance(state.get("messages", []))
    if not prev:
        ans = "직전에 하신 질문을 찾지 못했어요. 대화를 조금 더 이어가면 기억해둘게요."
        return {"messages": [AIMessage(content=ans)], "final_answer": ans}

    last_ai = _get_last_ai_before_current_turn(state)

    # ➜ 요약 규칙:
    # - 사용자 '질문 자체'를 한 줄로 요약(의도 위주)
    # - (있다면) 직전 어시스턴트 답변의 핵심만 2~3불릿으로 정리
    # - 절대 새로운 사실/추측 추가 금지. last_ai가 없으면 요약만.
    # - 한국어로 간결하게.
    sys = SystemMessage(content=(
        "너는 대화 히스토리를 요약하는 비서다.\n"
        "- 먼저 사용자의 직전 질문을 한 줄로 요약하라(의도 중심, 예: 'EDT 뜻을 물어봄').\n"
        "- 그 다음, 직전 어시스턴트 답변이 주어졌다면 그 내용에서만 핵심 2~3개를 불릿으로 정리하라.\n"
        "- 새로운 사실을 절대 추가하지 말라. 답변 요약은 제공된 last_ai 내용에서만 뽑아라.\n"
        "- 한국어로 간결하게 출력하되, 불필요한 서론/결어는 넣지 말라."
    ))
    user = HumanMessage(content=(
        f"[직전 사용자 질문]\n{prev}\n\n"
        f"[직전 어시스턴트 답변]\n{last_ai if last_ai else '(없음)'}\n\n"
        "위 정보를 바탕으로 다음 형식으로 출력:\n"
        "방금 전 당신의 질문(요약): <한 줄>\n"
        "(선택) 답변 핵심:\n"
        "- <핵심1>\n- <핵심2>\n- <핵심3>\n"
        "※ last_ai가 없으면 '답변 핵심' 섹션은 생략."
    ))
    out = llm.invoke([sys, user])
    summary = (getattr(out, "content", "") or "").strip()

    # 원문도 살짝 인용해 주면 사용자가 헷갈리지 않음
    final = f"{summary}\n\n> 원문 질문: {prev}"

    return {
        "messages": [AIMessage(content=final)],   # ✅ 이번 턴 델타만
        "final_answer": final,
        "last_agent": "memory_echo",
    }
