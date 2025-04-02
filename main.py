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
        Aşağıda verilen konuda, yalnızca **Türkçe** dilinde, açık uçlu ve anlamlı bir **sınav sorusu** üret.

        Sadece soruyu döndür. Ekstra açıklama, tekrar veya selamlama verme.

        Konu: {data.topic}

        Soruyu "Soru:" ile başlat ve sadece 1 kez yaz.
        """
        print("TOKEN:", os.getenv("HF_API_KEY"))
        headers = {
            "Authorization": f"Bearer {os.getenv('HF_API_KEY')}"
        }

        json = {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.7,
                "max_new_tokens": 150,
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
        answer = result[0]["generated_text"]

        # Prompt tekrarını çıkar
        if prompt.strip() in answer:
            answer = answer.replace(prompt.strip(), '')

        # Gereksiz tekrar eden "Soru:" ifadelerini temizle
        answer = answer.strip()
        if answer.lower().startswith("soru: soru:"):
            answer = "Soru:" + answer[11:].strip()
        elif answer.lower().startswith("soru: soru"):
            answer = "Soru:" + answer[9:].strip()

        # Eğer iki kez aynı cümleyi yazdıysa bunu sadeleştirme için:
        lines = answer.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() not in unique_lines:
                unique_lines.append(line.strip())
        answer = '\n'.join(unique_lines).strip()
        return {"response": answer}
