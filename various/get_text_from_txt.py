from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain import hub

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llms import ollama_llm


with open("./test_folder/test.txt") as f:
    test_file = f.read()

text_splitter2 = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

documents = text_splitter2.create_documents([test_file])

vectorstore = Chroma.from_documents(
    documents=documents, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 1})

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever,
     "question": RunnablePassthrough()}
    | prompt
    | ollama_llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("what is my name?"):
    print(chunk, end="", flush=True)

vectorstore.delete_collection()
