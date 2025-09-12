# scentpick/routers/chatbot.py

from __future__ import annotations

import os
import uuid
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# LangGraph 실행 객체(app) 가져오기 (graph.compile() 결과)
# 스샷 기준: scentpick/mas/tools/perfume_chatbot.py
# 기존 경로를 쓰고 있었다면 호환 위해 try/except 처리
try:
    from scentpick.mas.tools.perfume_chatbot import app as graph_app  # ✅ 스샷 경로
except Exception:
    from scentpick.mas.perfume_chatbot import app as graph_app        # ⛳️ 기존 경로 호환

# DB 세션 팩토리
from database import SessionLocal

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


# -----------------------------
# 공용: 서비스 토큰 검증/DB 세션
# -----------------------------
def verify_service_token(x_service_token: Optional[str] = Header(None)):
    expected = os.environ.get("SERVICE_TOKEN")
    if expected and x_service_token != expected:
        raise HTTPException(status_code=401, detail="invalid service token")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------
# (기존) 간단 테스트용 엔드포인트
# -----------------------------
class ChatIn(BaseModel):
    query: str
    history: list[str] | None = None  # 필요 없으면 지워도 됨


class ChatOut(BaseModel):
    answer: str


@router.post("/chat", response_model=ChatOut)
def run_chat(body: ChatIn, x_thread_id: Optional[str] = Header(None)):
    """
    단순 테스트용. 멀티턴 확인을 위해 x-thread-id 헤더를 thread_id로 사용.
    """
    init_state = {
        "messages": [HumanMessage(content=body.query)],
        "next": None,
        "router_json": None,
    }
    # ✅ 멀티턴: 헤더로 온 thread_id(없으면 임시 UUID)로 상태 이어받기
    config = {"configurable": {"thread_id": x_thread_id or str(uuid.uuid4())}}

    try:
        out = graph_app.invoke(init_state, config=config)
        ai_msgs = [m for m in out.get("messages", []) if isinstance(m, AIMessage)]
        if not ai_msgs:
            raise HTTPException(500, "No AI response generated.")
        return ChatOut(answer=ai_msgs[-1].content)
    except Exception as e:
        raise HTTPException(500, f"graph error: {e}")


# -----------------------------
# (신규) Django 연동용 /chat.run
# -----------------------------
class ChatMessageIn(BaseModel):
    content: str = Field(..., min_length=1)
    idempotency_key: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRunIn(BaseModel):
    user_id: int
    conversation_id: Optional[int] = None
    external_thread_id: Optional[str] = None  # ✅ 장고에서 온 UUID
    title: Optional[str] = None
    message: ChatMessageIn


class ChatMessageOut(BaseModel):
    role: str
    content: str
    model: Optional[str] = None
    tool_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str


class ChatRunOut(BaseModel):
    conversation_id: int
    external_thread_id: str
    final_answer: str
    messages_appended: List[ChatMessageOut]


# 유틸: 멱등 결과 조회
def find_prev_assistant_by_idem(db: Session, idem_key: str):
    sql = text(
        """
      SELECT m.conversation_id, m.content, m.model, m.tool_name, m.metadata, m.created_at,
             c.external_thread_id
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
       WHERE m.idempotency_key = :idem AND m.role = 'assistant'
       ORDER BY m.created_at DESC
       LIMIT 1
    """
    )
    return db.execute(sql, {"idem": idem_key}).mappings().first()


# 유틸: LangGraph 상태를 JSON 직렬화 가능 형태로 축약
def _serialize_messages(msgs: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs or []:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": m.content})
        elif isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": m.content, "tool_name": getattr(m, "name", None)})
        else:
            out.append({"role": type(m).__name__, "content": str(getattr(m, "content", ""))})
    return out


def snapshot_state(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    snap = {}
    try:
        for k, v in (raw_state or {}).items():
            if k == "messages":
                snap["messages"] = _serialize_messages(v)
            else:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    snap[k] = v
                elif isinstance(v, dict):
                    safe = {}
                    for dk, dv in v.items():
                        safe[dk] = dv if isinstance(dv, (str, int, float, bool, type(None))) else str(dv)
                    snap[k] = safe
                elif isinstance(v, list):
                    snap[k] = [itm if isinstance(itm, (str, int, float, bool, type(None))) else str(itm) for itm in v]
                else:
                    snap[k] = str(v)
    except Exception:
        snap = {"messages": _serialize_messages(raw_state.get("messages", []))}
    return snap


@router.post("/chat.run", response_model=ChatRunOut, dependencies=[Depends(verify_service_token)])
def chat_run(payload: ChatRunIn, db: Session = Depends(get_db), x_idempotency_key: Optional[str] = Header(None)):
    # 0) 멱등키 설정 & 이전 결과 재사용
    idem = payload.message.idempotency_key or x_idempotency_key or str(uuid.uuid4())

    prev = find_prev_assistant_by_idem(db, idem)
    if prev:
        # metadata 역직렬화 (DB가 JSON/text 모두 가능)
        meta_val = prev["metadata"]
        if isinstance(meta_val, (bytes, bytearray)):
            meta_val = meta_val.decode(errors="ignore")
        try:
            meta_dict = json.loads(meta_val) if isinstance(meta_val, str) and meta_val else None
        except Exception:
            meta_dict = None

        created_at = prev["created_at"]
        if getattr(created_at, "tzinfo", None) is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        return ChatRunOut(
            conversation_id=prev["conversation_id"],
            external_thread_id=prev["external_thread_id"],
            final_answer=prev["content"],
            messages_appended=[
                ChatMessageOut(
                    role="assistant",
                    content=prev["content"],
                    model=prev["model"],
                    tool_name=prev["tool_name"],
                    metadata=meta_dict,
                    created_at=created_at.isoformat(),
                )
            ],
        )

    now = datetime.utcnow()

    # 1) 대화 찾기/만들기: conversation_id > external_thread_id > 신규
    if payload.conversation_id:
        row = db.execute(
            text("SELECT id, user_id, external_thread_id FROM conversations WHERE id=:cid"),
            {"cid": payload.conversation_id},
        ).mappings().first()
        if not row or row["user_id"] != payload.user_id:
            raise HTTPException(404, "conversation not found for this user")
        conv_id = row["id"]
        thread_id = row["external_thread_id"] or payload.external_thread_id or str(uuid.uuid4())
        if not row["external_thread_id"] and payload.external_thread_id:
            db.execute(
                text("UPDATE conversations SET external_thread_id=:tid, updated_at=:now WHERE id=:cid"),
                {"tid": payload.external_thread_id, "now": now, "cid": conv_id},
            )

    elif payload.external_thread_id:
        row = db.execute(
            text("SELECT id, user_id, external_thread_id FROM conversations WHERE external_thread_id=:tid"),
            {"tid": payload.external_thread_id},
        ).mappings().first()
        if row:
            if row["user_id"] != payload.user_id:
                raise HTTPException(409, "thread id belongs to another user")
            conv_id = row["id"]
            thread_id = row["external_thread_id"]
        else:
            title = payload.title or payload.message.content[:40]
            res = db.execute(
                text(
                    """
                INSERT INTO conversations (user_id, title, external_thread_id, started_at, updated_at)
                VALUES (:uid, :title, :tid, :now, :now)
            """
                ),
                {"uid": payload.user_id, "title": title, "tid": payload.external_thread_id, "now": now},
            )
            conv_id = res.lastrowid
            thread_id = payload.external_thread_id
    else:
        # 백업: 아무것도 안 왔으면 신규 thread 발급
        thread_id = str(uuid.uuid4())
        title = payload.title or payload.message.content[:40]
        res = db.execute(
            text(
                """
            INSERT INTO conversations (user_id, title, external_thread_id, started_at, updated_at)
            VALUES (:uid, :title, :tid, :now, :now)
        """
            ),
            {"uid": payload.user_id, "title": title, "tid": thread_id, "now": now},
        )
        conv_id = res.lastrowid

    # 2) 사용자 메시지 저장 (role=user) — JSON 직렬화 주의
    meta_json_user = json.dumps(payload.message.metadata or {}, ensure_ascii=False)
    db.execute(
        text(
            """
        INSERT INTO messages (conversation_id, role, content, model, state, idempotency_key, tool_name, metadata, created_at)
        VALUES (:cid, 'user', :content, NULL, NULL, :idem, NULL, :meta, :now)
    """
        ),
        {"cid": conv_id, "content": payload.message.content, "idem": idem, "meta": meta_json_user, "now": now},
    )

    # 3) LangGraph 실행 (✅ thread_id로 멀티턴 상태 이어받기)
    init_state = {
        "messages": [HumanMessage(content=payload.message.content)],
        "next": None,
        "router_json": None,
    }
    config = {"configurable": {"thread_id": thread_id}}  # ✅ 핵심!

    try:
        out_state = graph_app.invoke(init_state, config=config)  # ✅ config 전달
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"graph error: {e}")

    # 상태 스냅샷(직렬화)
    state_snap = snapshot_state(out_state)
    state_json = json.dumps(state_snap, ensure_ascii=False)
    out_messages = out_state.get("messages", []) or []

    appended: List[ChatMessageOut] = []
    final_answer = ""

    # 4) LangGraph 산출 메시지 저장 — JSON 직렬화 주의
    for m in out_messages:
        if isinstance(m, AIMessage):
            role = "assistant"
            content = m.content
            tool_name = None
            model = getattr(m, "model", None) or "gpt-5-thinking"
        elif isinstance(m, ToolMessage):
            role = "tool"
            content = m.content
            tool_name = getattr(m, "name", None)
            model = None
        else:
            # Human/System 등은 저장 스킵(사용자 메시지는 이미 저장됨)
            continue

        ts = datetime.utcnow()
        meta_json_event = json.dumps({}, ensure_ascii=False)

        db.execute(
            text(
                """
            INSERT INTO messages (conversation_id, role, content, model, state, idempotency_key, tool_name, metadata, created_at)
            VALUES (:cid, :role, :content, :model, :state, :idem, :tool, :meta, :ts)
        """
            ),
            {
                "cid": conv_id,
                "role": role,
                "content": content,
                "model": model,
                "state": state_json if role == "assistant" else None,
                "idem": idem,
                "tool": tool_name,
                "meta": meta_json_event,
                "ts": ts,
            },
        )

        created_ts = ts.replace(tzinfo=timezone.utc)
        appended.append(
            ChatMessageOut(
                role=role,
                content=content,
                model=model,
                tool_name=tool_name,
                metadata={},  # 이벤트 메타가 필요하면 적절히 채우세요
                created_at=created_ts.isoformat(),
            )
        )
        if role == "assistant":
            final_answer = content

    if not final_answer:
        db.rollback()
        raise HTTPException(500, "no assistant answer produced")

    # 5) 대화 updated_at 갱신 & 커밋
    db.execute(text("UPDATE conversations SET updated_at=:now WHERE id=:cid"), {"now": datetime.utcnow(), "cid": conv_id})
    db.commit()

    return ChatRunOut(
        conversation_id=conv_id,
        external_thread_id=thread_id,
        final_answer=final_answer,
        messages_appended=appended,
    )
