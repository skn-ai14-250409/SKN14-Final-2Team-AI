# scentpick/mas/state.py
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

def safe_dict_merge(prev: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(prev, dict):
        prev = {}
    if not isinstance(new, dict):
        return prev
    merged = prev.copy()
    merged.update(new)
    return merged

def safe_list_concat(prev: Optional[List[Any]], new: Optional[List[Any]]) -> List[Any]:
    if not isinstance(prev, list):
        prev = []
    if not isinstance(new, list):
        return prev
    return prev + new

class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    next: Optional[str]
    router_json: Optional[Dict[str, Any]]

    parsed_slots: Annotated[Dict[str, Any], safe_dict_merge]
    search_results: Annotated[Dict[str, Any], safe_dict_merge]

    # ✅ 추천 히스토리 (마지막 추천을 rec_echo가 읽어감)
    rec_history: Annotated[List[Dict[str, Any]], safe_list_concat]

    # (선택) 지난 턴 마지막 실행 노드명: supervisor 프롬프트에 넣어주기 용도
    last_agent: Optional[str]

    final_answer: Optional[str]
    last_error: Optional[str]

    perfume_list: Optional[List[Dict[str, Any]]]

    image_url: Optional[str] 
