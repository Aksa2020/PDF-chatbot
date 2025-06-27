import os
import uuid
import torch
import streamlit as st
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

import pinecone

# --- Pinecone (v2) Init ---
pinecone.init(
    api_key=st.secrets["api_key"],
    environment=st.secrets["environment"]
)

index_name = st.secrets["index_name"]
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384, metric="cosine")

# --- Streamlit UI ---
st.set_page_config(page_title="PDF ChatBot", layout="centered")
st.title("📄 PDF ChatBot")

# --- Session Setup ---
if 'session_id' not in st.session_state:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state['session_id'] = f"session_{timestamp}_{uuid.uuid4().hex[:6]}"
    st.session_state['chat_messages'] = []

if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

# --- Embeddings ---
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# --- Upload PDF ---
upload_pdf = st.file_uploader("Upload PDF", type=["pdf"])
if upload_pdf and st.session_state['retriever'] is None:
    with st.spinner("Processing PDF..."):
        pdf_path = os.path.join(os.getcwd(), upload_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(upload_pdf.getbuffer())
        st.session_state['pdf_file_path'] = pdf_path

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        vectordb = LC_Pinecone.from_documents(
            documents=documents,
            embedding=embedding_model,
            index_name=index_name,
            namespace=st.session_state['session_id']
        )

        st.session_state['retriever'] = vectordb.as_retriever(search_kwargs={"k": 3})
        st.success("PDF processed and stored in Pinecone.")

# --- LLM via Groq ---
llm = ChatGroq(
    groq_api_key=st.secrets["groq_api_key"],
    model_name="llama3-8b-8192",
    temperature=0
)

# --- QA Chain ---
if st.session_state['retriever']:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state['retriever'],
        memory=st.session_state['memory'],
        return_source_documents=False
    )

# --- Handle Question ---
def handle_user_question():
    user_question = st.session_state["text"]
    if not user_question.strip():
        return
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"question": user_question})
        st.session_state['chat_messages'].append({"role": "user", "content": user_question})
        st.session_state['chat_messages'].append({"role": "bot", "content": result["answer"]})
    st.session_state["text"] = ""

# --- Chat UI ---
if st.session_state['chat_messages']:
    st.markdown("### 💬 Chat History")
    for msg in st.session_state['chat_messages']:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
        st.markdown(f"**{role}:** {msg['content']}")

st.text_input("Ask your question:", key="text", on_change=handle_user_question)

# --- Clear Session ---
def del_uploaded_pdf(path):
    if path and os.path.exists(path):
        os.remove(path)

if st.button("Clear Session"):
    st.session_state['memory'].clear()
    for key in ['chat_messages', 'text', 'retriever', 'pdf_file_path', 'upload_pdf']:
        if key in st.session_state:
            del st.session_state[key]
    del_uploaded_pdf(st.session_state.get('pdf_file_path', None))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state['session_id'] = f"session_{timestamp}_{uuid.uuid4().hex[:6]}"
    st.session_state['chat_messages'] = []
    st.success("Session cleared.")
    st.rerun()
