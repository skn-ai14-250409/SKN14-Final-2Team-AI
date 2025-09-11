from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from ..state import AgentState
import json
from ..prompts.ML_agent_prompt import ML_agent_system_prompt
from ..tools.tools_recommend import recommend_perfume_simple
from ..config import llm

def ML_agent_node(state: AgentState) -> AgentState:
    """ML agent - recommend_perfume_simple 도구 호출 후, LLM이 설명문 생성까지 수행"""
    user_query = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break
    if not user_query:
        user_query = "(empty)"

    try:
        # 1) ML 도구 호출 (구조화 데이터 보존)
        ml_result = recommend_perfume_simple.invoke({"user_text": user_query})
        # 예상 구조: {"recommendations": [...], "predicted_labels": [...]}
        ml_json_str = json.dumps(ml_result, ensure_ascii=False)

        # 2) LLM에 컨텍스트로 전달하여 자연어 답변 생성
        
        human_prompt = (
            f"사용자 질문:\n{user_query}\n\n"
            f"ML 추천 JSON:\n```json\n{ml_json_str}\n```"
        )

        llm_out = llm.invoke([SystemMessage(content=ML_agent_system_prompt),
                              HumanMessage(content=human_prompt)])

        # 3) 최종 답변을 대화에 추가
        msgs = state["messages"] + [AIMessage(content=llm_out.content)]
        return {
            "messages": msgs,
            "next": None,
            "router_json": state.get("router_json")
        }

    except Exception as e:
        error_msg = f"❌ ML 추천 생성 중 오류가 발생했습니다: {str(e)}"
        msgs = state["messages"] + [AIMessage(content=error_msg)]
        return {
            "messages": msgs,
            "next": None,
            "router_json": state.get("router_json")
        }
    