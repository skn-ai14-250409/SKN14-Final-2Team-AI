from fastapi import FastAPI, Depends
from scentpick.routers import chatbot
from sqlalchemy.orm import Session
from sqlalchemy import text
from dotenv import load_dotenv
from database import SessionLocal

# .env 파일 로드
load_dotenv()

# app = FastAPI()
app = FastAPI(title="Perfume Chat bot API") # yyh

# 라우터 등록
app.include_router(chatbot.router)

@app.get("/")
def read_root():
    return {"msg": "Hi I'm fast api! + FastAPI updated and deployed!!"}

# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# RDS 연결 확인용 엔드포인트
@app.get("/check-db")
def check_db(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT COUNT(*) FROM perfumes")).fetchone()
    return {"perfume_count": result[0]}