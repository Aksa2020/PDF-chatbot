import os
import shutil
import json
import uuid
import torch
import streamlit as st
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Streamlit UI Setup ---
st.set_page_config(page_title="PDF ChatBot", layout="centered")
st.title("📄 PDF ChatBot (Supabase)")

# --- Directories for local sessions ---
sessions_dir = os.path.join(os.getcwd(), "chat_sessions")
os.makedirs(sessions_dir, exist_ok=True)

# --- Session Setup ---
if 'session_id' not in st.session_state:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state['session_id'] = f"session_{timestamp}_{uuid.uuid4().hex[:6]}"
    st.session_state['chat_messages'] = []

if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if 'chat_messages' not in st.session_state:
    st.session_state['chat_messages'] = []

session_path = os.path.join(sessions_dir, f"{st.session_state['session_id']}.json")
if os.path.exists(session_path):
    with open(session_path, "r") as f:
        st.session_state['chat_messages'] = json.load(f)

# --- Supabase PGVector Setup ---
connection_string = st.secrets["SUPABASE_DB_URL"]  # format: postgres://user:pass@host/db
collection_name = "documents"  # must match pg table with `embedding vector(...)`

# --- Embedding Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# --- Upload PDF and Create Supabase VectorStore ---
upload_pdf = st.file_uploader("Upload the PDF file", type=["pdf"], key='upload_pdf')
if upload_pdf and st.session_state['retriever'] is None:
    with st.spinner("Loading PDF and indexing to Supabase..."):
        pdf_path = os.path.join(os.getcwd(), upload_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(upload_pdf.getbuffer())
        st.session_state['pdf_file_path'] = pdf_path

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        vectordb = PGVector(
            connection_string=connection_string,
            embedding_function=embedding_model,
            collection_name=collection_name,
            distance_strategy="cosine"
        )
        vectordb.add_documents(documents)
        st.session_state['retriever'] = vectordb.as_retriever(search_kwargs={"k": 3})
        st.success("Vector DB created and stored in Supabase.")

# --- Load Groq LLM ---
llm = ChatGroq(
    groq_api_key=st.secrets["groq_api_key"],
    model_name="llama3-8b-8192",
    temperature=0
)

# --- QA Chain ---
if st.session_state['retriever'] is not None:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state['retriever'],
        memory=st.session_state['memory'],
        return_source_documents=False,
        condense_question_llm=llm
    )

# --- Handle User Question ---
def handle_user_question():
    user_question = st.session_state['text']
    if not user_question.strip():
        return
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"question": user_question})
        st.session_state['chat_messages'].append({"role": "user", "content": user_question})
        st.session_state['chat_messages'].append({"role": "bot", "content": result['answer']})
        with open(session_path, "w") as f:
            json.dump(st.session_state['chat_messages'], f, indent=2)
    st.session_state['text'] = ""

# --- Sidebar Session Viewer ---
with st.sidebar:
    st.markdown("### 📂 View Previous Sessions")
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

# --- Chat UI ---
if st.session_state['chat_messages']:
    st.markdown("### 💬 Current Chat Session")
    for msg in st.session_state['chat_messages']:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
        st.markdown(f"**{role}:** {msg['content']}")

# --- Input Text Box ---
st.text_input("Ask your question:", key="text", on_change=handle_user_question)

# --- Clear Session ---
def del_uploaded_pdf(path):
    if path and os.path.exists(path):
        os.remove(path)

if st.button("Clear Session"):
    if 'memory' in st.session_state:
        st.session_state['memory'].clear()
    for key in ['chat_messages', 'text', 'retriever', 'pdf_file_path', 'upload_pdf']:
        if key in st.session_state:
            del st.session_state[key]
    del_uploaded_pdf(st.session_state.get('pdf_file_path'))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state['session_id'] = f"session_{timestamp}_{uuid.uuid4().hex[:6]}"
    st.session_state['chat_messages'] = []
    st.success("Session and chat history cleared.")
    st.rerun()
