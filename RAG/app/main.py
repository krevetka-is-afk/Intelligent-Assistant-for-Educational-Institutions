from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from vector import retriever

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

while True:
    print("\n\n----------------------")
    question = input("Ask your question (or type 'q' to quit): ")
    print("\n\n----------------------")

    if question.lower() == "q":
        break

    information = retriever.invoke(question)

    result = chain.invoke({"information": [information], "question": question})

    print(result)
