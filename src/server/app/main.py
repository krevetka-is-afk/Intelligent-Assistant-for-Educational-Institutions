from fastapi import FastAPI, Request
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from .vector import retriever

app = FastAPI()
model = OllamaLLM(model="gemma2:2b")
template = """
You are a helpful assistant that helps students find specific information in University sources.

Here is the information you have access to: {information}

Given the student's question,
provide a concise and accurate answer based on the information provided.

Student's question: {question}
"""

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
