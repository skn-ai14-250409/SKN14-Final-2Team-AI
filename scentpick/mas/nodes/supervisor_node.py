from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from ..config import llm
from ..state import AgentState
from ..prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT
from typing import Any, Dict
import json

def supervisor_node(state: AgentState) -> AgentState:
    """Call the router LLM and return parsed JSON + routing target."""
    user_query = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break
    if not user_query:
        user_query = "(empty)"

    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_SYSTEM_PROMPT),
        ("user", "{query}")
    ])
    chain = prompt | llm

    ai = chain.invoke({"query": user_query})
    text = ai.content

    # JSON strict parse
    chosen = "human_fallback"
    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(text)
        maybe = parsed.get("next")
        if isinstance(maybe, str) and maybe in {"LLM_parser","FAQ_agent","human_fallback","price_agent","ML_agent"}:
            chosen = maybe
    except Exception:
        parsed = {"error": "invalid_json", "raw": text}

    msgs = state["messages"] + [AIMessage(content=text)]
    return {
        "messages": msgs,
        "next": chosen,
        "router_json": parsed
    }