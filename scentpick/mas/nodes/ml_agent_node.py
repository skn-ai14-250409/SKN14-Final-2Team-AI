from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from ..state import AgentState
import json
from ..prompts.ML_agent_prompt import ML_agent_system_prompt
from ..tools.tools_recommend import recommend_perfume_vdb   # ← 새 도구로 교체
from ..config import llm

def ML_agent_node(state: AgentState) -> AgentState:
    """ML agent - Pinecone VDB를 통해 상위 3개 추천 후, LLM이 설명문 생성"""
    user_query = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break
    if not user_query:
        user_query = "(empty)"

    try:
        # 1) VDB 기반 추천 (라벨 재임베딩 + Pinecone 검색)
        ml_result = recommend_perfume_vdb.invoke({
            "user_text": user_query,
            "topk_labels": 3,
            "top_n_perfumes": 3,
            "use_thresholds": True,
            "alpha_labels": 0.8,
            "index_name": "perfume-vectordb2",
        })
        ml_json_str = json.dumps(ml_result, ensure_ascii=False)

        # 2) LLM 요약/설명 생성
        human_prompt = (
            f"사용자 질문:\n{user_query}\n\n"
            f"ML 추천 JSON:\n```json\n{ml_json_str}\n```"
        )
        llm_out = llm.invoke([
            SystemMessage(content=ML_agent_system_prompt),
            HumanMessage(content=human_prompt)
        ])

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
