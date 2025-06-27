import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# ✅ Correct import
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from groq import Groq
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
#from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import shutil
from sentence_transformers import SentenceTransformer
import torch

vector_space_dir = os.path.join(os.getcwd(), "vector_db")
os.makedirs(vector_space_dir, exist_ok=True)

# if not os.path.exists(vector_space_dir):
#     os.mkdir(vector_space_dir)

st.set_page_config(page_title="RAG ChatBot", layout="centered")
st.title("RAG ChatBot (Langchain + Groq)")

if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None

upload_pdf = st.file_uploader("Upload the PDF file", type=["pdf"], key='upload_pdf')
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={"device": "cpu"}  # 🔧 Force load directly on CPU
# )
# # Add this to move model from meta device safely if needed
# if hasattr(embedding_model.client, 'to_empty') and isinstance(embedding_model.client, Module):
#     embedding_model.client = Module.to_empty(embedding_model.client, device="cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

# patching to avoid meta tensor issue
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)
model_attr = getattr(embedding_model, "model", None)
if model_attr is not None and hasattr(model_attr, "to_empty"):
    try:
        model_attr = torch.nn.Module.to_empty(model_attr, device=device)
    except Exception as e:
        print("Warning: Failed to use `to_empty()`:", e)

if upload_pdf is not None and st.session_state['vectorstore'] is None:
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

#llm = OllamaLLM(model="llama2")
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatGroq(groq_api_key=st.secrets["groq_api_key"],
               model_name="llama3-8b-8192",
               temperature=0)

if st.session_state['retriever'] is not None:
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever = st.session_state['retriever'], memory = st.session_state['memory'], return_source_documents= False)
    user_question = st.text_input("Ask your question:", key='text')
    if user_question:
        with st.spinner("Thinking...."):
            result = qa_chain.invoke({"question": user_question})
            st.markdown(f"**You:** {user_question}")
            st.markdown(f"**Bot:** {result["answer"]}")

def del_vectordb(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def del_uploaded_pdf(path):
    if os.path.exists(path) and path:
        os.remove(path)

if st.button("Clear Session"):
    st.session_state['memory'].clear()
    st.session_state['retriever'] = None
    st.session_state['vectorstore'] = None
    del_vectordb(vector_space_dir)
    pdf_p = st.session_state.get('pdf_file_path', None)
    del_uploaded_pdf(pdf_p)
    st.session_state['pdf_file_path'] = None
    for key in ['upload_pdf', 'text']:
        if key in st.session_state:
            del st.session_state[key]
    st.success('Session, PDF and VectorDB are cleared')
    st.rerun()



