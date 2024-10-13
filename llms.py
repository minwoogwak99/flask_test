import config

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


# import config

ollama_llm = ChatOllama(model="llama3.2")

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
