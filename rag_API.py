# rag_api.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from rag_core import rag_query_with_history

app = FastAPI(title="RL LLM Agent API")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[Message] = []

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history_dicts: List[Dict[str, str]] = [
        {"role": m.role, "content": m.content} for m in req.history
    ]
    answer = rag_query_with_history(req.question, history_dicts)
    return ChatResponse(answer=answer)

