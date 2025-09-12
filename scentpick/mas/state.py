from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from typing import TypedDict, List, Optional, Dict, Any ,Annotated
# ---------- 1) State ----------
class AgentState(TypedDict,total=False):
    messages: Annotated[List[BaseMessage], add_messages]         # conversation log
    next: Optional[str]                   # routing decision key 
    router_json: Optional[Dict[str, Any]] # parsed JSON from router
