from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .nodes.supervisor_node import supervisor_node
from .nodes.llm_parser_node import LLM_parser_node
from .nodes.faq_node import FAQ_agent_node
from .nodes.human_fallback_node import human_fallback_node
from .nodes.price_agent_node import price_agent_node
from .nodes.ml_agent_node import ML_agent_node 
from .state import AgentState

# ----------  Build Graph ----------
graph = StateGraph(AgentState)

# 노드 추가
graph.add_node("supervisor", supervisor_node)
graph.add_node("LLM_parser", LLM_parser_node)
graph.add_node("FAQ_agent", FAQ_agent_node)
graph.add_node("human_fallback", human_fallback_node)
graph.add_node("price_agent", price_agent_node)
graph.add_node("ML_agent", ML_agent_node)

# 시작점 설정
graph.set_entry_point("supervisor")

# 조건부 라우팅 함수
def router_edge(state: AgentState) -> str:
    return state["next"] or "human_fallback"

# 조건부 엣지 추가 (supervisor에서 각 agent로)
graph.add_conditional_edges(
    "supervisor",
    router_edge,
    {
        "LLM_parser": "LLM_parser",
        "FAQ_agent": "FAQ_agent",
        "human_fallback": "human_fallback",
        "price_agent": "price_agent",
        "ML_agent": "ML_agent",
    },
)

# 각 에이전트에서 END로 가는 엣지 추가
for node in ["LLM_parser", "FAQ_agent", "human_fallback", "price_agent", "ML_agent"]:
    graph.add_edge(node, END)

# 그래프 컴파일
app = graph.compile(checkpointer=MemorySaver())


