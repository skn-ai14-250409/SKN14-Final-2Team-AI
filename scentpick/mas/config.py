import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
load_dotenv()
# ---------- 0) Config ----------
os.environ.setdefault("OPENAI_API_KEY", "PUT_YOUR_KEY_HERE")  # or set in env
MODEL_NAME = "gpt-4o-mini"  # keep it small & fast for routing

# Naver API 설정
naver_client_id = os.getenv("NAVER_CLIENT_ID")
naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
# Pinecone 초기화
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("perfume-vectordb2")

llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
