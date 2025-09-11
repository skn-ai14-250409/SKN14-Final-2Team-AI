from ..config import llm
from langchain_core.messages import  HumanMessage, AIMessage
from ..state import AgentState
from ..prompts.faq_prompt import faq_prompt


def FAQ_agent_node(state: AgentState) -> AgentState:
    """FAQ agent - LLM 기본 지식으로 향수 관련 질문 답변"""
    user_query = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break
    if not user_query:
        user_query = "(empty)"
    
    try:
        # LLM에게 향수 지식 전문가로서 답변하도록 프롬프트 설정

        
        chain = faq_prompt | llm
        result = chain.invoke({"question": user_query})
        
        # 결과를 포맷팅
        final_answer = f"📚 **향수 지식**\n\n{result.content}"
        
        msgs = state["messages"] + [AIMessage(content=final_answer)]
        return {
            "messages": msgs, 
            "next": None, 
            "router_json": state.get("router_json")
        }
    except Exception as e:
        error_msg = f"❌ 향수 지식 답변 생성 중 오류가 발생했습니다: {str(e)}"
        msgs = state["messages"] + [AIMessage(content=error_msg)]
        return {
            "messages": msgs, 
            "next": None, 
            "router_json": state.get("router_json")
        }
