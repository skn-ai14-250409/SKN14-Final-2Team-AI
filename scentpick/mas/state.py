from langchain_core.messages import BaseMessage

from typing import TypedDict, List, Optional, Dict, Any
# ---------- 1) State ----------
class AgentState(TypedDict,total=False):
    messages: List[BaseMessage]           # conversation log
    next: Optional[str]                   # routing decision key 
    router_json: Optional[Dict[str, Any]] # parsed JSON from router
