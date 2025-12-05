from __future__ import annotations

from app.db.models.document import Document, DocumentChunk


def test_document_chunk_as_context():
    doc = Document(tenant_id="t1", title="Doc", source="sourceA")
    chunk = DocumentChunk(
        tenant_id="t1",
        document_id=doc.id,
        content="hello world",
        embedding=[0.1, 0.2],
        position=0,
        chunk_metadata={"source": "custom"},
        document=doc,
    )

    ctx = chunk.as_context(score=0.9)

    assert ctx["id"]
    assert ctx["score"] == 0.9
    assert ctx["content"] == "hello world"
    assert ctx["source"] == "custom"
