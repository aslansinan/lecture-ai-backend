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
    allow_origins=["*"],  # React için açık tut
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TopicRequest(BaseModel):
    topic: str

@app.post("/generate-question")
def generate_question(data: TopicRequest):
    prompt = f"""
    Sen bir ilkokul öğretmenisin ve sınav hazırlıyorsun.
    Aşağıda verilen konuda, **yalnızca Türkçe** olacak şekilde 1 adet açık uçlu sınav sorusu yaz.

    Konu: {data.topic}

    Soruyu "Soru:" ile başlat. Gereksiz tekrar yapma.
    """
    print("TOKEN:", os.getenv("HF_API_KEY"))
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_API_KEY')}"
    }

    json = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 200,
        }
    }

    response = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
        headers=headers,
        json=json,
    )

    if response.status_code != 200:
        return {"response": f"API hatası: {response.status_code}"}

    result = response.json()
    answer = result[0]["generated_text"].replace(prompt, "").strip()

    return {"response": answer}
