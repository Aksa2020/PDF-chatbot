import os
import shutil
import json
import uuid
import torch
import streamlit as st
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Streamlit Page Setup ---
st.set_page_config(page_title="PDF ChatBot", layout="centered")
st.title("📄💬 PDF + Chat ChatBot")

# --- Directory Setup ---
vector_space_dir = os.path.join(os.getcwd(), "vector_db")
sessions_dir = os.path.join(os.getcwd(), "chat_sessions")
os.makedirs(vector_space_dir, exist_ok=True)
os.makedirs(sessions_dir, exist_ok=True)

# --- Session State ---
if 'session_id' not in st.session_state:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state['session_id'] = f"session_{timestamp}_{uuid.uuid4().hex[:6]}"
    st.session_state['chat_messages'] = []

if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        input_key="input"
    )

session_path = os.path.join(sessions_dir, f"{st.session_state['session_id']}.json")
if os.path.exists(session_path):
    with open(session_path, "r") as f:
        st.session_state['chat_messages'] = json.load(f)

# --- LLM and Embeddings ---
llm = ChatGroq(
    groq_api_key=st.secrets["groq_api_key"],
    model_name="llama3-8b-8192",
    temperature=0
)

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# --- PDF Upload ---
upload_pdf = st.file_uploader("📁 Upload PDF", type=["pdf"], key="upload_pdf")
if upload_pdf and "vectorstore" not in st.session_state:
    with st.spinner("Processing PDF..."):
        pdf_path = os.path.join(os.getcwd(), upload_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(upload_pdf.getbuffer())
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(vector_space_dir)
        st.session_state['vectorstore'] = vectorstore
        st.success("✅ Vector DB created!")

# --- Build Retriever + Chain ---
if "vectorstore" in st.session_state:
    st.session_state["retriever"] = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 3})

if "retriever" in st.session_state and "qa_chain" not in st.session_state:
    retriever = st.session_state["retriever"]

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context, rephrase it as a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the context to answer. If unsure, say 'I don't know'. Keep answers concise."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    qa_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=question_answer_chain,
        memory=st.session_state["memory"]
    )

    st.session_state["qa_chain"] = qa_chain

# --- Handle User Input ---
def handle_user_question():
    user_question = st.session_state['text']
    if not user_question.strip():
        return

    with st.spinner("Thinking..."):
        if "qa_chain" in st.session_state:
            result = st.session_state["qa_chain"].invoke({"input": user_question})
            answer = result["answer"]
        else:
            answer = llm.invoke(user_question)

        st.session_state["chat_messages"].append({"role": "user", "content": user_question})
        st.session_state["chat_messages"].append({"role": "bot", "content": answer})

        with open(session_path, "w") as f:
            json.dump(st.session_state["chat_messages"], f, indent=2)

    st.session_state["text"] = ""

# --- Chat UI ---
if st.session_state['chat_messages']:
    st.markdown("### 💬 Chat History")
    for msg in st.session_state['chat_messages']:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
        st.markdown(f"**{role}:** {msg['content']}")

st.text_input("Ask your question:", key="text", on_change=handle_user_question)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📂 Previous Sessions")
    session_files = [f.replace(".json", "") for f in os.listdir(sessions_dir) if f.endswith(".json")]
    selected_session = st.selectbox("Select a session", options=["-- Select --"] + session_files)
    if selected_session != "-- Select --":
        selected_path = os.path.join(sessions_dir, selected_session + ".json")
        if os.path.exists(selected_path):
            with open(selected_path, "r") as f:
                prev_msgs = json.load(f)
                with st.expander(f"Chat from `{selected_session}`", expanded=True):
                    for msg in prev_msgs:
                        role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
                        st.markdown(f"**{role}:** {msg['content']}")

# --- Clear Session ---
def del_vectordb(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def del_uploaded_pdf(path):
    if path and os.path.exists(path):
        os.remove(path)

if st.button("Clear Session"):
    for key in ['chat_messages', 'text', 'retriever', 'vectorstore', 'pdf_file_path', 'upload_pdf', 'qa_chain']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state['memory'].clear()
    del_vectordb(vector_space_dir)
    del_uploaded_pdf(st.session_state.get('pdf_file_path'))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state['session_id'] = f"session_{timestamp}_{uuid.uuid4().hex[:6]}"
    st.session_state['chat_messages'] = []
    st.success("Session cleared.")
    st.rerun()
