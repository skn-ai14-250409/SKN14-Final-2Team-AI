from fastapi import APIRouter

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

@router.get("/")
def list_users():
    return {"chatbot": ["Alice", "Bob", "Charlie"]}