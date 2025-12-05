from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes_chat import router as chat_router
from app.services.chat import ChatResult, ChatService, get_chat_service


class FakeChatService(ChatService):
    def __init__(self):
        # Skip parent initialization
        pass

    async def answer(self, question: str, user_id: str, tenant_id: str | None):
        return ChatResult(answer="ok", sources=["s1"], latency_ms=1, retrieval_ms=1, llm_ms=1)


def get_fake_chat_service():
    return FakeChatService()


def test_chat_endpoint_returns_response():
    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_chat_service] = get_fake_chat_service

    client = TestClient(app)
    resp = client.post("/chat", json={"question": "hi?", "user_id": "user123"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "ok"
    assert data["sources"] == ["s1"]
