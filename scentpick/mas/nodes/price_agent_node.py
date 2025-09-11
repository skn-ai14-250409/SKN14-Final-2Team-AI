from langchain_core.messages import HumanMessage, AIMessage
from ..state import AgentState
from ..tools.tools_price import price_tool


def price_agent_node(state: AgentState) -> AgentState:
    """Price agent - 직접 도구 호출"""
    user_query = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break
    if not user_query:
        user_query = "(empty)"
    
    try:
        # 직접 price_tool 호출
        price_result = price_tool.invoke({"user_query": user_query})
        
        # 결과를 더 자연스럽게 포맷팅
        final_answer = f"💰 **가격 정보**\n\n{price_result}"
        
        msgs = state["messages"] + [AIMessage(content=final_answer)]
        return {
            "messages": msgs, 
            "next": None, 
            "router_json": state.get("router_json")
        }
    except Exception as e:
        error_msg = f"❌ 가격 조회 중 오류가 발생했습니다: {str(e)}"
        msgs = state["messages"] + [AIMessage(content=error_msg)]
        return {
            "messages": msgs, 
            "next": None, 
            "router_json": state.get("router_json")
        }

