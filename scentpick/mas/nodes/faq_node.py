from ..config import llm
from langchain_core.messages import HumanMessage, AIMessage
from ..state import AgentState
from ..prompts.faq_prompt import faq_prompt

def FAQ_agent_node(state: AgentState) -> AgentState:
    """FAQ agent - LLM 기본 지식으로 향수 관련 질문 답변"""
    # 안전하게 최신 사용자 메시지 추출
    user_query = "(empty)"
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break

    try:
        chain = faq_prompt | llm
        ai = chain.invoke({"question": user_query})
        body = getattr(ai, "content", str(ai))

        final_answer = f"💬 답변 결과:\n{body}"

        # ✅ add_messages 사용 시, 이번 턴에 “추가될” 메시지만 반환
        return {
            "messages": [AIMessage(content=final_answer)],
            "final_answer": final_answer,
            # router_json은 바뀐 게 없으면 굳이 다시 넣지 않아도 됩니다.
        }
    except Exception as e:
        error_msg = f"❌ 향수 지식 답변 생성 중 오류가 발생했습니다: {e}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "final_answer": error_msg,
        }
