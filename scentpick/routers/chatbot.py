# scentpick/routers/chatbot.py

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Optional, Any, Dict, List
import json

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
def generate_ai_response(query: str, thread_id: str) -> dict:
    init_state = {
        "messages": [HumanMessage(content=query)],
        "next": None,
        "router_json": None,
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        out = graph_app.invoke(init_state, config=config)

        # 라우팅된 노드명 추출
        chosen = None
        rj = out.get("router_json")
        if isinstance(rj, dict):
            chosen = rj.get("chosen_agent") or rj.get("agent") or rj.get("next")
        if not chosen:
            chosen = out.get("next")

        ai_msgs = [m for m in out.get("messages", []) if isinstance(m, AIMessage)]
        answer = ai_msgs[-1].content if ai_msgs else "죄송합니다. 응답을 생성하지 못했습니다."

        return {
            "answer": answer,
            "parsed_slots": out.get("parsed_slots", {}) or {},
            "search_results": out.get("search_results", {"matches": []}) or {"matches": []},
            "perfume_list": out.get("perfume_list", []) or [],
            "chosen_agent": chosen,
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "parsed_slots": {},
            "search_results": {"matches": []},
            "perfume_list": [],
            "chosen_agent": None,
        }

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
    perfume_list: Optional[List[Dict[str, Any]]] = None

# -----------------------------
# 메인 엔드포인트 (/chat)
# -----------------------------
@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_service_token)])
def django_chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        now = datetime.utcnow()

        # 1) 기존 대화 확인 또는 새 대화 생성
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
                text("""
                    INSERT INTO conversations (user_id, title, external_thread_id, started_at, updated_at)
                    VALUES (:uid, :title, :tid, :now, :now)
                """),
                {"uid": request.user_id, "title": title, "tid": thread_id, "now": now},
            )
            conv_id = res.lastrowid

        # 2) 사용자 메시지 저장
        res = db.execute(
            text("""
                INSERT INTO messages (conversation_id, role, content, model, created_at)
                VALUES (:cid, 'user', :content, NULL, :now)
            """),
            {"cid": conv_id, "content": request.query, "now": now},
        )
        request_msg_id = res.lastrowid

        # 3) AI 응답 생성 (추천 후보 포함)
        ai_output = generate_ai_response(request.query, thread_id)

        ai_answer      = ai_output["answer"]
        parsed_slots   = ai_output.get("parsed_slots", {})
        search_results = ai_output.get("search_results", {"matches": []})
        chosen_agent   = ai_output.get("chosen_agent")

        # ✅ 추천 리스트 노출 허용 노드
        ALLOW_NODES = {"LLM_parser", "ML_agent", "rec_echo"}
        allow_list = chosen_agent in ALLOW_NODES

        # ✅ perfume_list: 허용 노드일 때만 구성 (아니면 None)
        perfume_list = None
        if allow_list:
            perfume_list = ai_output.get("perfume_list") or []
            if not perfume_list:
                # LLM_parser 등이 search_results만 채웠을 때 fallback
                for m in search_results.get("matches", []):
                    meta = m.get("metadata", {}) or {}
                    pid = meta.get("no")
                    try:
                        pid = int(pid) if pid is not None else None
                    except Exception:
                        pid = None
                    perfume_list.append({
                        "id": pid,
                        "brand": meta.get("brand"),
                        "name": meta.get("name"),
                    })
            if not perfume_list:
                perfume_list = None  # 빈 배열이면 키 제거 효과(프론트에서 버튼 안 뜸)

        # 4) AI 응답 저장
        res = db.execute(
            text("""
                INSERT INTO messages (conversation_id, role, content, model, created_at)
                VALUES (:cid, 'assistant', :content, :model, :now)
            """),
            {"cid": conv_id, "content": ai_answer, "model": "fastapi-bot", "now": now},
        )
        ai_msg_id = res.lastrowid

        # 5) rec_runs 저장 (실제 라우팅된 노드명 기록)
        res = db.execute(
            text("""
                INSERT INTO rec_runs (parsed_slots, agent, model_version, created_at, conversation_id, request_msg_id, user_id, query_text)
                VALUES (:parsed_slots, :agent, :model_version, :now, :cid, :req_mid, :uid, :qtxt)
            """),
            {
                "parsed_slots": json.dumps(parsed_slots if parsed_slots is not None else {}, ensure_ascii=False),
                "agent": chosen_agent or "unknown",
                "model_version": "v1.0",
                "now": now,
                "cid": conv_id,
                "req_mid": request_msg_id,
                "uid": request.user_id,
                "qtxt": request.query,
            },
        )
        run_id = res.lastrowid

        # 6) rec_candidates 저장 (검색 결과 기록은 유지)
        for idx, match in enumerate(search_results.get("matches", []), start=1):
            meta = match.get("metadata", {}) or {}
            pid_val = meta.get("no")
            try:
                pid_val = int(pid_val) if pid_val is not None else None
            except Exception:
                pid_val = None

            try:
                db.execute(
                    text("""
                        INSERT INTO rec_candidates (`rank`, score, reason_summary, reason_detail, retrieved_from, perfume_id, run_rec_id)
                        VALUES (:rank, :score, :summary, :detail, :retrieved, :pid, :rid)
                    """),
                    {
                        "rank": idx,
                        "score": match.get("score", 0.0),
                        "summary": meta.get("text"),
                        "detail": json.dumps({}, ensure_ascii=False),
                        "retrieved": "pinecone",
                        "pid": pid_val,
                        "rid": run_id,
                    },
                )
            except Exception as e:
                print(f"❌ rec_candidates insert error idx={idx}: {e}")

        # 7) 대화 updated_at 갱신
        db.execute(
            text("UPDATE conversations SET updated_at=:now WHERE id=:cid"),
            {"now": now, "cid": conv_id},
        )

        db.commit()

        return ChatResponse(
            conversation_id=conv_id,
            final_answer=ai_answer,
            success=True,
            perfume_list=perfume_list,  # 허용 노드 아닐 때는 None
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        return ChatResponse(
            conversation_id=0,
            final_answer=f"Error: {str(e)}",
            success=False,
            perfume_list=[]
        )

# -----------------------------
# 추가 엔드포인트 (/chat.run)
# -----------------------------
@router.post("/chat.run", response_model=ChatResponse, dependencies=[Depends(verify_service_token)])
def django_chat_endpoint_run(request: ChatRequest, db: Session = Depends(get_db)):
    return django_chat_endpoint(request, db)
