# scentpick/mas/perfume_chatbot.py  — 복붙용 최종본
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .nodes.supervisor_node import supervisor_node
from .nodes.llm_parser_node import LLM_parser_node
from .nodes.faq_node import FAQ_agent_node
from .nodes.human_fallback_node import human_fallback_node
from .nodes.price_agent_node import price_agent_node
from .nodes.ml_agent_node import ML_agent_node
from .nodes.memory_echo_node import memory_echo_node
from .nodes.rec_echo_node import rec_echo_node   # ← 추가
from .nodes.review_agent_node import review_agent_node, is_review_agent_query # yyh
from .state import AgentState

# ---------- Build Graph ----------
graph = StateGraph(AgentState)

# 노드 추가
graph.add_node("supervisor", supervisor_node)
graph.add_node("LLM_parser", LLM_parser_node)
graph.add_node("FAQ_agent", FAQ_agent_node)
graph.add_node("human_fallback", human_fallback_node)
graph.add_node("price_agent", price_agent_node)
graph.add_node("ML_agent", ML_agent_node)
graph.add_node("memory_echo", memory_echo_node)
graph.add_node("rec_echo", rec_echo_node)  # ← 추가
graph.add_node("review_agent", review_agent_node)

# 시작점
graph.set_entry_point("supervisor")

# 조건부 라우팅 함수 (안전하게 .get 사용)
# yyh
def router_edge(state: AgentState) -> str:
    # --- (향설명 + 가격) 강제 라우팅 가드 ---
    msgs = state.get("messages") or []
    last = msgs[-1] if msgs else None
    q = getattr(last, "content", "") or ""
    if is_review_agent_query(q):
        print(f"[ROUTER] review_agent 강제 선택 (query='{q}')", flush=True)
        return "review_agent"
    # 👇 이 줄 추가 (기존 supervisor 결과 따라가도록)
    return state.get("next") or "human_fallback"
# yyh

# supervisor → 각 에이전트 분기
graph.add_conditional_edges(
    "supervisor",
    router_edge,
    {
        "LLM_parser": "LLM_parser",
        "FAQ_agent": "FAQ_agent",
        "human_fallback": "human_fallback",
        "price_agent": "price_agent",
        "ML_agent": "ML_agent",
        "memory_echo": "memory_echo",
        "rec_echo": "rec_echo", # ← 추가
        "review_agent": "review_agent", # yyh
        
    },
)

# 각 에이전트에서 END로
for node in [
    "LLM_parser",
    "FAQ_agent",
    "human_fallback",
    "price_agent",
    "ML_agent",
    "memory_echo",
    "rec_echo",  # ← 추가
    "review_agent", # yyh
]:
    graph.add_edge(node, END)

# 그래프 컴파일
checkpointer = MemorySaver()   # 프로덕션에선 Redis/SQL checkpointer 권장
app = graph.compile(checkpointer=checkpointer)
