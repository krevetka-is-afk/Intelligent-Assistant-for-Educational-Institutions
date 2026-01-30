import uvicorn
from fastapi import FastAPI, Request
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from . import config
from .vector import retriever

app = FastAPI()
model = OllamaLLM(model=config.model)
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
        result = chain.invoke({"information": [information], "question": question})
        return {"response": result}

    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}


if __name__ == "__main__":
    print("Server start")
    print("Documentation API: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)