from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from typing import TypedDict, List, Optional, Dict, Any, Annotated

# ---------- 1) State ----------
class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]  # 대화 로그
    next: Optional[str]                                   # 라우팅 결정 키
    router_json: Optional[Dict[str, Any]]                 # supervisor/router 결과

    parsed_slots: Optional[Dict[str, Any]]                # LLM_parser에서 파싱된 슬롯 정보
    search_results: Optional[Dict[str, Any]]              # Pinecone 검색 결과
    final_answer: Optional[str]                           # 최종 생성된 응답 텍스트
