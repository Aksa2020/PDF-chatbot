import os
import streamlit as st
import shutil
import json
import uuid
import torch
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Setup ---
st.set_page_config(page_title="PDF ChatBot", layout="centered")
st.title("PDF ChatBot)")

vector_space_dir = os.path.join(os.getcwd(), "vector_db")
sessions_dir = os.path.join(os.getcwd(), "chat_sessions")
os.makedirs(vector_space_dir, exist_ok=True)
os.makedirs(sessions_dir, exist_ok=True)

# --- Initialize Session ---
if 'session_id' not in st.session_state:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state['session_id'] = f"session_{timestamp}_{uuid.uuid4().hex[:6]}"
    st.session_state['chat_messages'] = []

if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None
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

# --- Load PDF and create vectorstore ---
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

model_attr = getattr(embedding_model, "model", None)
if model_attr and hasattr(model_attr, "to_empty"):
    try:
        model_attr = torch.nn.Module.to_empty(model_attr, device=device)
    except Exception as e:
        print("Warning using to_empty:", e)

upload_pdf = st.file_uploader("Upload the PDF file", type=["pdf"], key='upload_pdf')
if upload_pdf and st.session_state['vectorstore'] is None:
    with st.spinner("Loading PDF and creating vector DB...."):
        pdf_path = os.path.join(os.getcwd(), upload_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(upload_pdf.getbuffer())
        st.session_state['pdf_file_path'] = pdf_path
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(vector_space_dir)
        st.session_state['vectorstore'] = vectorstore
        st.session_state['retriever'] = vectorstore.as_retriever(search_kwargs={"k": 3})
        st.success("Vector DB Created")

# --- Load LLM ---
llm = ChatGroq(
    groq_api_key=st.secrets["groq_api_key"],
    model_name="llama3-8b-8192",
    temperature=0
)

# --- Create QA Chain (before callback) ---
if st.session_state['retriever'] is not None and 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state['retriever'],
        memory=st.session_state['memory'],
        return_source_documents=False,
        condense_question_llm=llm
    )


# if st.session_state['retriever'] is not None:
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=st.session_state['retriever'],
#         memory=st.session_state['memory'],
#         return_source_documents=False,
#         condense_question_llm=llm
#     )

# # --- Handle User Question ---
# def handle_user_question():
#     user_question = st.session_state['text']
#     if not user_question.strip():
#         return

#     with st.spinner("Thinking..."):
#         result = qa_chain.invoke({"question": user_question})
#         st.session_state['chat_messages'].append({"role": "user", "content": user_question})
#         st.session_state['chat_messages'].append({"role": "bot", "content": result['answer']})

#         with open(session_path, "w") as f:
#             json.dump(st.session_state['chat_messages'], f, indent=2)

#     st.session_state['text'] = ""

def handle_user_question():
    user_question = st.session_state['text']
    if not user_question.strip():
        return

    qa_chain = st.session_state.get('qa_chain')
    if not qa_chain:
        st.error("❌ QA chain is not initialized. Please upload a PDF first.")
        return

    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"question": user_question})
        st.session_state['chat_messages'].append({"role": "user", "content": user_question})
        st.session_state['chat_messages'].append({"role": "bot", "content": result['answer']})

        with open(session_path, "w") as f:
            json.dump(st.session_state['chat_messages'], f, indent=2)

    st.session_state['text'] = ""


# --- UI: Sidebar ---
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

# --- UI: Main chat area ---
if st.session_state['chat_messages']:
    st.markdown("### 💬 Current Chat Session")
    for msg in st.session_state['chat_messages']:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
        st.markdown(f"**{role}:** {msg['content']}")

# --- Input box ---
st.text_input("Ask your question:", key="text", on_change=handle_user_question)

# --- Clear Session ---
def del_vectordb(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def del_uploaded_pdf(path):
    if path and os.path.exists(path):
        os.remove(path)

if st.button("Clear Session"):
    # Clear LangChain memory
    if 'memory' in st.session_state:
        st.session_state['memory'].clear()
        # Clear chat history and input
        for key in ['chat_messages', 'text', 'retriever', 'vectorstore', 'pdf_file_path', 'upload_pdf']:
            if key in st.session_state:
                del st.session_state[key]
                # Delete vector DB directory
    del_vectordb(vector_space_dir)
    # Delete uploaded PDF if present
    pdf_p = st.session_state.get('pdf_file_path')
    if pdf_p:
        del_uploaded_pdf(pdf_p)
        # ✅ Reset session_id and create a fresh chat log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state['session_id'] = f"session_{timestamp}_{uuid.uuid4().hex[:6]}"
    st.session_state['chat_messages'] = []
    st.success("Session, PDF, and Vector DB cleared.")
    st.rerun()

# if st.button("Clear Session"):
#     st.session_state['memory'].clear()
#     st.session_state['retriever'] = None
#     st.session_state['vectorstore'] = None
#     st.session_state['chat_messages'] = []
#     del_vectordb(vector_space_dir)
#     del_uploaded_pdf(st.session_state.get('pdf_file_path', None))
#     st.session_state['pdf_file_path'] = None
#     for key in ['upload_pdf']:
#         if key in st.session_state:
#             del st.session_state[key]
#     st.success("Session, PDF, chat, and vector DB are cleared")
#     st.rerun()
