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
        f"❓ '{user_query}' 더 명확한 설명이 필요합니다.\n"
        f"👉 질문을 구체적으로 다시 작성해 주세요.\n"
        f"💡 또는 향수에 관한 멋진 질문을 해보시는 건 어떨까요?"
    )
    
    msgs = state["messages"] + [AIMessage(content=fallback_response)]
    return {"messages": msgs, "next": None, "router_json": state.get("router_json")}