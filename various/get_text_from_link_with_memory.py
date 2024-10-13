import bs4
import time
import chromadb

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
import nest_asyncio

nest_asyncio.apply()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llms import ollama_llm

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"


def get_from_link_with_memory(link, input, prev_chat_log):
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(link),
        # bs_kwargs={"parse_only": bs4_strainer},
    )
    loader.requests_kwargs = {'verify': False}
    docs = loader.aload()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    all_splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=HuggingFaceEmbeddings(model_name=embedding_model_name))

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6})

    system_prompt = """
      human
      You are an assistant for question-answering tasks.
      Use the following pieces of retrieved context to answer the question.
      If you don't know the answer, just say that you don't know.
      Use three sentences maximum and keep the answer concise.
      Context: {context}
    """

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        ollama_llm, retriever, contextualize_q_prompt
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(ollama_llm, qa_prompt)
    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain)

    chat_history = []

    if (prev_chat_log):
        for chat in prev_chat_log:
            if (chat['source'] == 'user'):
                chat_history.append(HumanMessage(content=chat['message']))
            if (chat['source'] == 'bot'):
                chat_history.append(AIMessage(content=chat['message']))

    print(chat_history)
    for chunk in rag_chain.stream(
            {"input": input, "chat_history": chat_history}):
        if answer_chunk := chunk.get("answer"):
            yield answer_chunk
            time.sleep(0.1)

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", system_prompt),
    #     ("human", "{input}"),
    # ])

    # question_answer_chain = create_stuff_documents_chain(
    #     llm=ollama_llm, prompt=prompt)
    # rag_chain = create_retrieval_chain(
    #     retriever=retriever, combine_docs_chain=question_answer_chain)

    # for chunk in rag_chain.stream({"input": input}):
    #     print(chunk['answer'], end="", flush=True)
    #     yield chunk['answer']
    #     time.sleep(0.1)

    # for chunk in rag_chain.stream({"input": input}):
    #     if answer_chunk := chunk.get("answer"):
    #         print(f"{answer_chunk}", end="")

    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)

    # rag_chain = (
    #     {"context": retriever | format_docs,
    #      "question": RunnablePassthrough()}
    #     | prompt
    #     | ollama_llm
    #     | StrOutputParser()
    # )

    chromadb.api.client.SharedSystemClient.clear_system_cache()
