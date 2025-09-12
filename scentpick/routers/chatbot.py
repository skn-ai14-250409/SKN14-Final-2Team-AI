# scentpick/routers/chatbot.py

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text

from langchain_core.messages import HumanMessage, AIMessage
from scentpick.mas.perfume_chatbot import app as graph_app
from database import SessionLocal

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# -----------------------------
# 공용: 서비스 토큰 검증/DB 세션
# -----------------------------
def verify_service_token(x_service_token: Optional[str] = Header(None)):
    expected = os.environ.get("SERVICE_TOKEN")
    if not expected:
        env = os.environ.get("ENVIRONMENT", "development")
        if env == "production":
            raise HTTPException(status_code=500, detail="SERVICE_TOKEN not configured")
        return True
    if x_service_token != expected:
        raise HTTPException(status_code=401, detail="Invalid service token")
    return True

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------
# AI 응답 생성 함수
# -----------------------------
def generate_ai_response(query: str, thread_id: str) -> str:
    init_state = {
        "messages": [HumanMessage(content=query)],
        "next": None,
        "router_json": None,
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        out = graph_app.invoke(init_state, config=config)
        ai_msgs = [m for m in out.get("messages", []) if isinstance(m, AIMessage)]
        if not ai_msgs:
            return "죄송합니다. 응답을 생성하지 못했습니다."
        return ai_msgs[-1].content
    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# Pydantic 모델
# -----------------------------
class ChatRequest(BaseModel):
    user_id: int
    query: str = Field(..., min_length=1)
    conversation_id: Optional[int] = None

class ChatResponse(BaseModel):
    conversation_id: int
    final_answer: str
    success: bool

# -----------------------------
# 메인 엔드포인트 (/chat)
# -----------------------------
@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_service_token)])
def django_chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        now = datetime.utcnow()

        # 1. 기존 대화 확인 또는 새 대화 생성
        if request.conversation_id:
            row = db.execute(
                text("SELECT id, user_id, external_thread_id FROM conversations WHERE id=:cid"),
                {"cid": request.conversation_id},
            ).mappings().first()

            if not row or row["user_id"] != request.user_id:
                raise HTTPException(status_code=404, detail="Conversation not found")

            conv_id = row["id"]
            thread_id = row["external_thread_id"] or str(uuid.uuid4())

            if not row["external_thread_id"]:
                db.execute(
                    text("UPDATE conversations SET external_thread_id=:tid, updated_at=:now WHERE id=:cid"),
                    {"tid": thread_id, "now": now, "cid": conv_id},
                )
        else:
            thread_id = str(uuid.uuid4())
            title = request.query[:15]
            res = db.execute(
                text(
                    """
                    INSERT INTO conversations (user_id, title, external_thread_id, started_at, updated_at)
                    VALUES (:uid, :title, :tid, :now, :now)
                """
                ),
                {"uid": request.user_id, "title": title, "tid": thread_id, "now": now},
            )
            conv_id = res.lastrowid

        # 2. 사용자 메시지 저장
        db.execute(
            text(
                """
                INSERT INTO messages (conversation_id, role, content, model, created_at)
                VALUES (:cid, 'user', :content, NULL, :now)
            """
            ),
            {"cid": conv_id, "content": request.query, "now": now},
        )

        # 3. AI 응답 생성
        ai_response = generate_ai_response(request.query, thread_id)

        # 4. AI 응답 저장
        db.execute(
            text(
                """
                INSERT INTO messages (conversation_id, role, content, model, created_at)
                VALUES (:cid, 'assistant', :content, :model, :now)
            """
            ),
            {"cid": conv_id, "content": ai_response, "model": "fastapi-bot", "now": datetime.utcnow()},
        )

        # 5. 대화 updated_at 갱신
        db.execute(
            text("UPDATE conversations SET updated_at=:now WHERE id=:cid"),
            {"now": datetime.utcnow(), "cid": conv_id},
        )

        db.commit()

        return ChatResponse(conversation_id=conv_id, final_answer=ai_response, success=True)

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        return ChatResponse(conversation_id=0, final_answer=f"Error: {str(e)}", success=False)

# -----------------------------
# 추가 엔드포인트 (/chat.run)
# -----------------------------
@router.post("/chat.run", response_model=ChatResponse, dependencies=[Depends(verify_service_token)])
def django_chat_endpoint_run(request: ChatRequest, db: Session = Depends(get_db)):
    return django_chat_endpoint(request, db)
