model = "gemma2:2b"
template = """
You are a helpful assistant that helps students find specific information in University sources.

Here is the information you have access to: {information}

Given the student's question,
provide a concise and accurate answer based on the information provided.

Student's question: {question}
"""
ollama_port = "http://localhost:11434"
