import bs4
import time

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llms import ollama_llm

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
test_link = "https://lilianweng.github.io/posts/2023-06-23-agent/"


def get_from_link(link, query):
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(link,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    all_splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=HuggingFaceEmbeddings(model_name=embedding_model_name))

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6})

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | ollama_llm
        | StrOutputParser()
    )

    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)
        yield chunk
        time.sleep(0.1)

    chromadb.api.client.SharedSystemClient.clear_system_cache()
