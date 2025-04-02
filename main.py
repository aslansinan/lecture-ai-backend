from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model ve tokenizer yükleniyor
tokenizer = AutoTokenizer.from_pretrained("mertcobanov/turkish-question-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("mertcobanov/turkish-question-generator")

class TopicRequest(BaseModel):
    topic: str

@app.post("/generate-question")
def generate_question(data: TopicRequest):
    input_text = f"soru: {data.topic}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_beams=4,
            early_stopping=True
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": f"Soru: {result}"}
