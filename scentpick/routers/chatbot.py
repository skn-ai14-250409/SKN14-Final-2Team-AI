# from fastapi import APIRouter

# router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# @router.get("/")
# def list_users():
#     return {"chatbot": ["Alice", "Bob", "Charlie"]}


# scentpick/routers/chatbot.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

# LangGraph 실행 객체(app) 가져오기
# (perfume_chatbot.py에서 graph.compile() 결과가 app)
from scentpick.mas.perfume_chatbot import app as graph_app

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

class ChatIn(BaseModel):
    query: str
    history: list[str] | None = None  # 필요 없으면 지워도 됨

class ChatOut(BaseModel):
    answer: str

@router.post("/chat", response_model=ChatOut)
def run_chat(body: ChatIn):
    # LangGraph 초기 상태 만들기 (네 그래프 상태 스키마에 맞춤)
    init_state = {
        "messages": [HumanMessage(content=body.query)],
        "next": None,
        "router_json": None
    }

    try:
        out = graph_app.invoke(init_state)  # LangGraph 실행

        # 마지막 AI 응답 추출
        ai_msgs = [m for m in out.get("messages", []) if isinstance(m, AIMessage)]
        if not ai_msgs:
            raise HTTPException(500, "No AI response generated.")
        return ChatOut(answer=ai_msgs[-1].content)

    except Exception as e:
        # 그래프 내부 예외를 안전하게 노출
        raise HTTPException(500, f"graph error: {e}")
