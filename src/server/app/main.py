import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from . import config
from .vector import retriever

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://client:8501"],  # Add your client URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ollama_host = os.getenv("OLLAMA_HOST", config.ollama_port)
model = OllamaLLM(model=config.model, base_url=ollama_host)
template = config.template

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/ask")
async def ask(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        if not question:
            return {"error": "Can not find question"}
        information = retriever.invoke(question)
        response = chain.invoke({"information": [information], "question": question})
        return {"response": response}

    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}


if __name__ == "__main__":
    print("Server start")
    print("Documentation API: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
