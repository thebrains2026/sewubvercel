import base64
import json
import os
import tempfile
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from openai import OpenAI

app = FastAPI(title="My Friend Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/voice-chat")
async def voice_chat(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    model: str = Form("gpt-4o-mini"),
    messages_json: str = Form("[]"),
):
    if not api_key.strip():
        raise HTTPException(status_code=400, detail="OpenAI API 키를 입력해 주세요.")

    try:
        messages = json.loads(messages_json)
        if not isinstance(messages, list):
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="대화 이력 형식이 올바르지 않습니다.")

    suffix = os.path.splitext(file.filename or "recording.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        client = OpenAI(api_key=api_key.strip())

        # 1. Whisper STT
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        question = (transcript.text or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="음성이 인식되지 않았습니다. 다시 녹음해 주세요.")

        # 2. GPT 응답 생성
        updated_messages = messages + [{"role": "user", "content": question}]
        completion = client.chat.completions.create(
            model=model,
            messages=updated_messages,
        )
        reply = (completion.choices[0].message.content or "").strip()
        updated_messages.append({"role": "assistant", "content": reply})

        # 3. gTTS 음성 합성
        audio_base64 = synthesize_tts_base64(reply, lang="ko")

        return {
            "question": question,
            "reply": reply,
            "audio_base64": audio_base64,
            "messages": updated_messages,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류가 발생했습니다: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)