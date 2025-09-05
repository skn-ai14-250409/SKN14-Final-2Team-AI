from fastapi import FastAPI
from scentpick.routers import chatbot

app = FastAPI()

# 라우터 등록
app.include_router(chatbot.router)

@app.get("/")
def read_root():
    return {"msg": "Hi I'm fast api! + FastAPI updated and deployed!!"}