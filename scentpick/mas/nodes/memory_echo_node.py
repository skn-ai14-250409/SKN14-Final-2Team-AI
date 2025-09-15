# scentpick/mas/nodes/memory_echo_node.py
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from ..state import AgentState
from ..tools.utils import get_prev_user_utterance
from ..config import llm
from ..prompts.memory_echo_prompt import MEMORY_ECHO_SYSTEM_PROMPT



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

    # ✅ 시스템 프롬프트는 분리 파일에서 가져옴
    sys = SystemMessage(content=MEMORY_ECHO_SYSTEM_PROMPT)

    # ✅ 휴먼 메시지는 인라인로 구성 (템플릿 분리 안 함)
    user = HumanMessage(content=(
        f"[직전 사용자 질문]\n{prev}\n\n"
        f"[직전 어시ส턴트 답변]\n{last_ai if last_ai else '(없음)'}\n\n"
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
