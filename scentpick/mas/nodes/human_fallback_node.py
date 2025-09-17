from langchain_core.messages import HumanMessage, AIMessage
from ..state import AgentState

def human_fallback_node(state: AgentState) -> AgentState:
    """향수 관련 복잡한 질문에 대한 기본 응답"""
    user_query = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break
    if not user_query:
        user_query = "(empty)"
    
    fallback_response = (
        f"💬 ScentPick 챗봇은 향수에 관한 질문에 답변하는 전용 챗봇입니다.\n"
        f"❓ '{user_query}' 해당 질문은 향수와 직접 관련이 없어 답변드리기 어려워요.\n"
        f"👉 향수에 대해 구체적으로 다시 질문해 주세요! (예: '여름에 어울리는 시트러스 향 추천해줘')\n"
        f"💡 또는 특정 브랜드/노트/상황에 맞는 향수를 물어보시면 멋진 추천을 드릴 수 있습니다."
    )
    
    msgs = state["messages"] + [AIMessage(content=fallback_response)]
    return {"messages": msgs, "next": None, "router_json": state.get("router_json")}