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
        ans = (
            "방금 직전의 질문을 아직 찾지 못했어요.\n"
            "조금만 더 대화를 이어가시면 최근 질문을 요약해서 바로 알려드릴게요!"
        )
        return {"messages": [AIMessage(content=ans)], "final_answer": ans, "last_agent": "memory_echo"}

    last_ai = _get_last_ai_before_current_turn(state)

    # ➜ 요약 규칙(친절 톤 + '완전 추출식'):
    # - 1) 사용자의 '직전 질문'을 한 줄로 아주 간단히 요약(의도 중심).
    # - 2) 직전 어시스턴트 답변(last_ai)이 있으면, 그 '문장에 실제로 등장한 표현'만 골라
    #      2~3개 불릿으로 축약해 제시(동의어/추론/상식 추가 금지, 수치/지속시간/가격 등 단정은
    #      last_ai에 '정확히' 있지 않으면 쓰지 말 것).
    # - 3) last_ai가 없으면 '직전 답변 핵심' 섹션 자체를 생략.
    # - 4) 한국어, 공손하지만 부담스럽지 않게. 불필요한 서론/결어 금지.
    sys = SystemMessage(content=(
        "너는 대화 히스토리를 '친절하지만 간결하게' 요약하는 비서다.\n"
        "반드시 '완전 추출식'으로 요약하라. 즉, 직전 어시스턴트 답변(last_ai)에 실제로 존재하는 "
        "문장/어구만 그대로 축약해서 사용하라. 동의어 치환, 상식적 보충, 추론, 일반화 금지.\n"
        "수치, 지속시간, 가격, 사용 시기 등 단정적 사실은 last_ai에 명시된 경우에만 사용하라.\n"
        "출력 형식(정확히 이 순서/문구 사용):\n"
        "📝 방금 하신 질문 요약: <한 줄>\n"
        "(선택) ✅ 직전 답변 핵심:\n"
        "- <핵심1>\n- <핵심2>\n- <핵심3>\n"
        "※ last_ai가 없거나 뽑을 핵심이 없으면 '(선택) ✅ 직전 답변 핵심:' 섹션은 통째로 생략한다."
    ))
    user = HumanMessage(content=(
        f"[직전 사용자 질문]\n{prev}\n\n"
        f"[직전 어시스턴트 답변]\n{last_ai if last_ai else '(없음)'}\n\n"
        "위 규칙대로만 출력하라."
    ))
    out = llm.invoke([sys, user])
    summary = (getattr(out, "content", "") or "").strip()

    # 최종 출력(원문 인용은 살짝, 사용자 친화적 이모지 유지)
    final = f"{summary}\n\n🗣️ 원문 질문: {prev}"

    return {
        "messages": [AIMessage(content=final)],
        "final_answer": final,
        "last_agent": "memory_echo",
    }
