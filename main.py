import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TopicRequest(BaseModel):
    topic: str

@app.post("/generate-question")
def generate_question(data: TopicRequest):
    prompt = f"""
    Aşağıda verilen konuya göre bir ilkokul öğrencisine yönelik açık uçlu 1 sınav sorusu üretin. Yalnızca soru cümlesini üretin. Cevap 'Soru:' ile başlamalı.

    Konu: {data.topic}

    Soru:
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_API_KEY')}"
    }

    json_data = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 200
        }
    }

    response = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
        headers=headers,
        json=json_data
    )

    if response.status_code != 200:
        return {"response": f"API hatası: {response.status_code}"}

    result = response.json()
    generated = result[0]["generated_text"]

    # Prompt'u ayır
    answer = generated.replace(prompt, "").strip()

    # "Soru:" başlığına ait fazlalıkları sadeleştir
    if answer.lower().startswith("soru: soru:"):
        answer = "Soru:" + answer[11:].strip()
    elif answer.lower().startswith("soru: soru"):
        answer = "Soru:" + answer[9:].strip()

    # Çoklu tekrarları temizle
    lines = answer.split('\n')
    seen = set()
    unique_lines = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            unique_lines.append(line)
    answer = "\n".join(unique_lines).strip()

    # Eğer cevap sadece "Soru:" kalırsa, fallback mesaj döndür
    if answer.strip().lower() == "soru:" or len(answer.strip()) < 10:
        return {"response": "Soru: Belirtilen konuda anlamlı bir sınav sorusu oluşturulamadı. Lütfen konuyu tekrar ifade edin."}

    return {"response": answer}
