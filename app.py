import base64
import os
import tempfile
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="My Friend Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    api_key: str
    model: str = "gpt-4o-mini"
    messages: list[ChatMessage]
    with_tts: bool = True


def synthesize_tts_base64(text: str, lang: str = "ko") -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(tmp_path)
        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/api/stt")
async def speech_to_text(
    file: UploadFile = File(...),
    api_key: str = Form(...),
):
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API 키가 필요합니다.")

    suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        client = OpenAI(api_key=api_key)
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return {"text": transcript.text}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/chat")
def chat_and_tts(req: ChatRequest):
    if not req.api_key:
        raise HTTPException(status_code=400, detail="OpenAI API 키가 필요합니다.")

    client = OpenAI(api_key=req.api_key)
    payload = [{"role": m.role, "content": m.content} for m in req.messages]

    completion = client.chat.completions.create(
        model=req.model,
        messages=payload,
    )
    reply = completion.choices[0].message.content or ""
    audio_base64 = synthesize_tts_base64(reply) if req.with_tts else None

    return {
        "reply": reply,
        "audio_base64": audio_base64,
    }