"""A stand-in for `vllm serve`: speaks the OpenAI chat-completions protocol but answers with ExtractiveStubLLM.
Only for exercising the HTTP path and the API without a GPU. It is not a language model.

    uvicorn scripts.fake_vllm_server:app --port 8000
"""
from fastapi import FastAPI

from insurance_rag.stub_llm import ExtractiveStubLLM

app = FastAPI()
_llm = ExtractiveStubLLM()


@app.post("/v1/chat/completions")
def chat(body: dict):
    return {"id": "fake", "object": "chat.completion", "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": _llm.chat(body["messages"])}}]}
