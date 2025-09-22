# scentpick/routers/chatbot.py

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Optional, Any, Dict, List
import json

from fastapi import APIRouter, HTTPException, Header, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text
import asyncio

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
# AI 응답 스트리밍 생성 함수
# -----------------------------
async def generate_ai_response_streaming(query: str, thread_id: str):
    """
    스트리밍 방식으로 AI 응답을 생성합니다.
    각 청크를 yield하여 실시간으로 응답을 전송합니다.
    """
    init_state = {
        "messages": [HumanMessage(content=query)],
        "next": None,
        "router_json": None,
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # 현재는 기존 방식으로 응답을 생성하고 청크로 나누어 전송
        # 향후 LangGraph에서 스트리밍을 지원하면 해당 방식으로 변경 가능
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

        # 응답을 청크 단위로 분할하여 전송
        chunk_size = 10  # 문자 단위로 청크 크기 조정
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i + chunk_size]
            # yield {"content": chunk}
            # await asyncio.sleep(0.05)  # 스트리밍 효과를 위한 지연
            sse_data = json.dumps({"content": chunk}, ensure_ascii=False)
            yield f"data: {sse_data}\n\n"
            await asyncio.sleep(0.01)  # 지연은 짧게 (실제 flush 유도)

        # 추천 리스트 처리
        ALLOW_NODES = {"LLM_parser", "ML_agent", "rec_echo"}
        allow_list = chosen in ALLOW_NODES

        perfume_list = None
        if allow_list:
            perfume_list = out.get("perfume_list") or []
            search_results = out.get("search_results", {"matches": []})

            if not perfume_list:
                # search_results에서 perfume_list 구성
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
                perfume_list = None

        # 완료 신호와 함께 추가 데이터 전송
        yield {
            "done": True,
            "parsed_slots": out.get("parsed_slots", {}) or {},
            "search_results": out.get("search_results", {"matches": []}) or {"matches": []},
            "perfume_list": perfume_list,
            "chosen_agent": chosen,
        }

    except Exception as e:
        yield {"error": f"서버 오류: {str(e)}"}

# -----------------------------
# Pydantic 모델
# -----------------------------
class ChatRequest(BaseModel):
    user_id: int
    query: str = Field(..., min_length=1)
    conversation_id: Optional[int] = None
    stream: Optional[bool] = False

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

        # 6) rec_candidates 저장 (perfume_list → search_results fallback)
        inserted = False

        # search_results를 dict으로 변환해두면 빠르게 매칭 가능
        sr_meta_map = {}
        for match in search_results.get("matches", []):
            try:
                pid = int(match["metadata"].get("no"))
            except Exception:
                pid = None
            if pid:
                sr_meta_map[pid] = {
                    "score": match.get("score", 0.0),
                    "text": match["metadata"].get("text"),
                }

        if perfume_list:
            for idx, item in enumerate(perfume_list, start=1):
                pid = item.get("id")
                if not pid:
                    continue

                # score / text 매칭
                sr_info = sr_meta_map.get(pid, {})
                score_val = item.get("score", sr_info.get("score", 0.0))
                summary_val = item.get("text") or sr_info.get("text") or item.get("name")

                try:
                    db.execute(
                        text("""
                            INSERT INTO rec_candidates 
                                (`rank`, score, reason_summary, reason_detail, retrieved_from, perfume_id, run_rec_id)
                            VALUES 
                                (:rank, :score, :summary, :detail, :retrieved, :pid, :rid)
                        """),
                        {
                            "rank": idx,
                            "score": score_val,
                            "summary": summary_val,
                            "detail": json.dumps({}, ensure_ascii=False),
                            "retrieved": "ml_result",
                            "pid": pid,
                            "rid": run_id,
                        },
                    )
                    inserted = True
                except Exception as e:
                    print(f"❌ rec_candidates insert error (perfume_list idx={idx}): {e}")

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

# -----------------------------
# 스트리밍 엔드포인트 (/stream)
# -----------------------------
@router.post("/chat/stream", dependencies=[Depends(verify_service_token)])
async def chat_stream(
    request: Request,
    x_service_token: str = Header(None, alias="X-Service-Token"),
    db: Session = Depends(get_db)
):
    """
    스트리밍 채팅 엔드포인트
    Django에서 SSE 형식으로 실시간 응답을 받을 수 있도록 구현
    """

    # 인증은 이미 verify_service_token 의존성에서 처리됨

    # 요청 데이터 파싱
    try:
        body = await request.json()
        user_id = body.get("user_id")
        query = body.get("query")
        conversation_id = body.get("conversation_id")
        is_stream = body.get("stream", False)

        if not query:
            async def error_stream():
                yield f"data: {json.dumps({'error': '쿼리가 비어있습니다.'}, ensure_ascii=False)}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")

        if not user_id:
            async def error_stream():
                yield f"data: {json.dumps({'error': '사용자 ID가 필요합니다.'}, ensure_ascii=False)}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")

    except Exception as e:
        async def error_stream():
            yield f"data: {json.dumps({'error': f'요청 파싱 오류: {str(e)}'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # 스트리밍 응답 생성
    async def generate_stream():
        try:
            now = datetime.utcnow()

            # 1) 기존 대화 확인 또는 새 대화 생성
            if conversation_id:
                row = db.execute(
                    text("SELECT id, user_id, external_thread_id FROM conversations WHERE id=:cid"),
                    {"cid": conversation_id},
                ).mappings().first()

                if not row or row["user_id"] != user_id:
                    yield f"data: {json.dumps({'error': 'Conversation not found'}, ensure_ascii=False)}\n\n"
                    return

                conv_id = row["id"]
                thread_id = row["external_thread_id"] or str(uuid.uuid4())

                if not row["external_thread_id"]:
                    db.execute(
                        text("UPDATE conversations SET external_thread_id=:tid, updated_at=:now WHERE id=:cid"),
                        {"tid": thread_id, "now": now, "cid": conv_id},
                    )
            else:
                thread_id = str(uuid.uuid4())
                title = query[:15]
                res = db.execute(
                    text("""
                        INSERT INTO conversations (user_id, title, external_thread_id, started_at, updated_at)
                        VALUES (:uid, :title, :tid, :now, :now)
                    """),
                    {"uid": user_id, "title": title, "tid": thread_id, "now": now},
                )
                conv_id = res.lastrowid

            # 2) 사용자 메시지 저장
            res = db.execute(
                text("""
                    INSERT INTO messages (conversation_id, role, content, model, created_at)
                    VALUES (:cid, 'user', :content, NULL, :now)
                """),
                {"cid": conv_id, "content": query, "now": now},
            )
            request_msg_id = res.lastrowid

            # 3) AI 스트리밍 응답 생성
            full_response = ""
            ai_output = {}

            async for chunk in generate_ai_response_streaming(query, thread_id):
                if chunk.get("content"):
                    full_response += chunk["content"]
                    # Django로 청크 전송
                    yield f"data: {json.dumps({'content': chunk['content']}, ensure_ascii=False)}\n\n"
                elif chunk.get("done"):
                    # AI 응답 완료, 추가 데이터 저장
                    ai_output = chunk
                    break
                elif chunk.get("error"):
                    yield f"data: {json.dumps({'error': chunk['error']}, ensure_ascii=False)}\n\n"
                    return

            # 4) AI 응답 저장
            res = db.execute(
                text("""
                    INSERT INTO messages (conversation_id, role, content, model, created_at)
                    VALUES (:cid, 'assistant', :content, :model, :now)
                """),
                {"cid": conv_id, "content": full_response, "model": "fastapi-bot", "now": now},
            )
            ai_msg_id = res.lastrowid

            # 5) rec_runs 저장
            parsed_slots = ai_output.get("parsed_slots", {})
            chosen_agent = ai_output.get("chosen_agent")

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
                    "uid": user_id,
                    "qtxt": query,
                },
            )
            run_id = res.lastrowid

            # 6) rec_candidates 저장
            perfume_list = ai_output.get("perfume_list")
            search_results = ai_output.get("search_results", {"matches": []})

            if perfume_list:
                sr_meta_map = {}
                for match in search_results.get("matches", []):
                    try:
                        pid = int(match["metadata"].get("no"))
                    except Exception:
                        pid = None
                    if pid:
                        sr_meta_map[pid] = {
                            "score": match.get("score", 0.0),
                            "text": match["metadata"].get("text"),
                        }

                for idx, item in enumerate(perfume_list, start=1):
                    pid = item.get("id")
                    if not pid:
                        continue

                    sr_info = sr_meta_map.get(pid, {})
                    score_val = item.get("score", sr_info.get("score", 0.0))
                    summary_val = item.get("text") or sr_info.get("text") or item.get("name")

                    try:
                        db.execute(
                            text("""
                                INSERT INTO rec_candidates
                                    (`rank`, score, reason_summary, reason_detail, retrieved_from, perfume_id, run_rec_id)
                                VALUES
                                    (:rank, :score, :summary, :detail, :retrieved, :pid, :rid)
                            """),
                            {
                                "rank": idx,
                                "score": score_val,
                                "summary": summary_val,
                                "detail": json.dumps({}, ensure_ascii=False),
                                "retrieved": "ml_result",
                                "pid": pid,
                                "rid": run_id,
                            },
                        )
                    except Exception as e:
                        print(f"❌ rec_candidates insert error (streaming idx={idx}): {e}")

            # 7) 대화 updated_at 갱신
            db.execute(
                text("UPDATE conversations SET updated_at=:now WHERE id=:cid"),
                {"now": now, "cid": conv_id},
            )

            db.commit()

            # 완료 신호와 함께 추가 데이터 전송
            final_data = {
                "done": True,
                "conversation_id": conv_id,
                "perfume_list": perfume_list or []
            }
            yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            db.rollback()
            yield f"data: {json.dumps({'error': f'서버 오류: {str(e)}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Transfer-Encoding": "chunked",
        }
    )
