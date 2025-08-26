import os
import shutil
import json
import uuid
import torch
import streamlit as st
from datetime import datetime
import traceback

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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit.components.v1 as components

# ✅ Google Search Console verification
components.html("""
    <meta name="google-site-verification" content="12b55dIg7kc7farCsRWMYZtAsgc2rSszfhh_-qJ0Wtg" />
""", height=0)



# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="PDF ChatBot", 
    layout="centered",
    page_icon="📄"
)
st.title("📄💬 PDF ChatBot")
st.markdown("Upload a PDF and chat about its contents, or have a general conversation!")

# --- Directory Setup ---
vector_space_dir = os.path.join(os.getcwd(), "vector_db")
sessions_dir = os.path.join(os.getcwd(), "chat_sessions")
uploads_dir = os.path.join(os.getcwd(), "uploads")

for directory in [vector_space_dir, sessions_dir, uploads_dir]:
    os.makedirs(directory, exist_ok=True)

# --- Initialize Session State ---
def initialize_session():
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

initialize_session()

# --- Load Previous Session ---
def load_session():
    session_path = os.path.join(sessions_dir, f"{st.session_state['session_id']}.json")
    if os.path.exists(session_path):
        try:
            with open(session_path, "r") as f:
                data = json.load(f)
                st.session_state['chat_messages'] = data.get('messages', [])
                # Restore memory from messages
                for msg in st.session_state['chat_messages']:
                    if msg["role"] == "user":
                        st.session_state['memory'].chat_memory.add_user_message(msg["content"])
                    else:
                        st.session_state['memory'].chat_memory.add_ai_message(msg["content"])
        except Exception as e:
            st.error(f"Error loading session: {str(e)}")

load_session()

# --- Initialize LLM and Embeddings ---
@st.cache_resource
def initialize_models():
    try:
        llm = ChatGroq(
            groq_api_key=st.secrets["groq_api_key"],
            model_name="llama3-8b-8192",
            temperature=0.1
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device}
        )
        
        return llm, embedding_model
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None, None

llm, embedding_model = initialize_models()

if not llm or not embedding_model:
    st.stop()

# --- Text Splitter ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# --- PDF Processing Functions ---
def validate_pdf(file):
    """Validate uploaded PDF file"""
    if file.size > 50 * 1024 * 1024:  # 50MB limit
        return False, "File size exceeds 50MB limit"
    if not file.name.lower().endswith('.pdf'):
        return False, "Please upload a PDF file"
    return True, "Valid"

def process_pdf(uploaded_file):
    """Process uploaded PDF and create vector store"""
    try:
        # Save uploaded file
        pdf_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and split documents
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        if not documents:
            return None, "No content found in PDF"
        
        # Split documents into chunks
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = FAISS.from_documents(texts, embedding_model)
        vectorstore.save_local(vector_space_dir)
        
        # Clean up uploaded file
        os.remove(pdf_path)
        
        return vectorstore, f"Successfully processed {len(texts)} text chunks"
        
    except Exception as e:
        return None, f"Error processing PDF: {str(e)}"

# --- PDF Upload Section ---
st.markdown("### 📁 Upload PDF Document")
upload_pdf = st.file_uploader(
    "Choose a PDF file", 
    type=["pdf"], 
    key="upload_pdf",
    help="Upload a PDF to chat about its contents"
)

if upload_pdf and "vectorstore" not in st.session_state:
    # Validate file
    is_valid, message = validate_pdf(upload_pdf)
    if not is_valid:
        st.error(message)
        st.stop()
    
    with st.spinner("🔄 Processing PDF... This may take a moment."):
        vectorstore, status_message = process_pdf(upload_pdf)
        
        if vectorstore:
            st.session_state['vectorstore'] = vectorstore
            st.session_state['pdf_processed'] = True
            st.success(f"✅ {status_message}")
        else:
            st.error(f"❌ {status_message}")

# --- Setup Retrieval Chain ---
def setup_retrieval_chain():
    """Setup the retrieval chain for PDF Q&A"""
    if "vectorstore" not in st.session_state:
        return None
    
    try:
        retriever = st.session_state["vectorstore"].as_retriever(
            search_kwargs={"k": 4}
        )
        
        # Contextualize question prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given a chat history and the latest user question which might reference 
            context in the chat history, formulate a standalone question which can be understood 
            without the chat history. Do NOT answer the question, just reformulate it if needed 
            and otherwise return it as is."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contextualize_q_prompt
        )
        
        # Q&A prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant for question-answering tasks. Use the following 
            pieces of retrieved context to answer the question. If you don't know the answer, 
            just say that you don't know. Keep the answer concise and relevant.
            
            Context: {context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=question_answer_chain
        )
        
        return rag_chain
        
    except Exception as e:
        st.error(f"Error setting up retrieval chain: {str(e)}")
        return None

# --- Chat Functions ---
def save_session():
    """Save current session to file"""
    session_path = os.path.join(sessions_dir, f"{st.session_state['session_id']}.json")
    try:
        session_data = {
            'session_id': st.session_state['session_id'],
            'messages': st.session_state['chat_messages'],
            'timestamp': datetime.now().isoformat(),
            'has_pdf': 'vectorstore' in st.session_state
        }
        with open(session_path, "w") as f:
            json.dump(session_data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving session: {str(e)}")

def handle_user_question():
    """Process user question and generate response"""
    user_question = st.session_state.get('text', '').strip()
    if not user_question:
        return
    
    with st.spinner("🤔 Thinking..."):
        try:
            # Add user message to memory
            st.session_state['memory'].chat_memory.add_user_message(user_question)
            
            # Get chat history for context
            chat_history = st.session_state['memory'].chat_memory.messages
            
            if "vectorstore" in st.session_state:
                # Use RAG chain for PDF-based questions
                rag_chain = setup_retrieval_chain()
                if rag_chain:
                    result = rag_chain.invoke({
                        "input": user_question,
                        "chat_history": chat_history
                    })
                    answer = result["answer"]
                else:
                    answer = "Sorry, there was an error processing your question about the PDF."
            else:
                # Use LLM directly for general conversation
                general_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful AI assistant. Provide clear, concise, and helpful responses."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])
                
                chain = general_prompt | llm
                result = chain.invoke({
                    "input": user_question,
                    "chat_history": chat_history
                })
                answer = result.content if hasattr(result, 'content') else str(result)
            
            # Add AI response to memory
            st.session_state['memory'].chat_memory.add_ai_message(answer)
            
            # Update chat messages
            st.session_state["chat_messages"].append({"role": "user", "content": user_question})
            st.session_state["chat_messages"].append({"role": "bot", "content": answer})
            
            # Save session
            save_session()
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.error(error_msg)
            st.session_state["chat_messages"].append({"role": "user", "content": user_question})
            st.session_state["chat_messages"].append({"role": "bot", "content": error_msg})
    
    # Clear input
    st.session_state["text"] = ""

# --- Display Chat History ---
if st.session_state['chat_messages']:
    st.markdown("### 💬 Chat History")
    
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state['chat_messages']):
            if msg["role"] == "user":
                st.markdown(f"""
                <div style='text-align: right; margin: 10px 0;'>
                    <div style='background-color: #007bff; color: white; padding: 10px; 
                               border-radius: 10px; display: inline-block; max-width: 80%;'>
                        🧑 <strong>You:</strong> {msg['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='text-align: left; margin: 10px 0;'>
                    <div style='background-color: #f1f1f1; color: black; padding: 10px; 
                               border-radius: 10px; display: inline-block; max-width: 80%;'>
                        🤖 <strong>Bot:</strong> {msg['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# --- User Input ---
st.markdown("### ✍️ Ask a Question")
st.text_input(
    "Type your question here:", 
    key="text", 
    on_change=handle_user_question,
    placeholder="Ask about the PDF or have a general conversation..."
)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ℹ️ Session Info")
    st.info(f"**Session ID:** {st.session_state['session_id'][:15]}...")
    
    if 'vectorstore' in st.session_state:
        st.success("📄 PDF loaded and ready!")
    else:
        st.warning("📄 No PDF loaded - General chat mode")
    
    st.markdown("---")
    
    st.markdown("### 📂 Previous Sessions")
    try:
        session_files = [f.replace(".json", "") for f in os.listdir(sessions_dir) if f.endswith(".json")]
        session_files.sort(reverse=True)  # Most recent first
        
        if session_files:
            selected_session = st.selectbox(
                "Select a session to view", 
                options=["-- Select --"] + session_files[:10]  # Show last 10 sessions
            )
            
            if selected_session != "-- Select --":
                selected_path = os.path.join(sessions_dir, selected_session + ".json")
                if os.path.exists(selected_path):
                    try:
                        with open(selected_path, "r") as f:
                            session_data = json.load(f)
                            prev_msgs = session_data.get('messages', [])
                            
                        with st.expander(f"Chat from `{selected_session}`", expanded=False):
                            for msg in prev_msgs[-6:]:  # Show last 6 messages
                                role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
                                st.markdown(f"**{role}:** {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                    except Exception as e:
                        st.error(f"Error loading session: {str(e)}")
        else:
            st.info("No previous sessions found")
    except Exception as e:
        st.error(f"Error accessing sessions: {str(e)}")
    
    st.markdown("---")
    
    # --- Clear Session Button ---
    if st.button("🗑️ Clear Current Session", type="primary"):
        try:
            # Clear session state
            keys_to_clear = ['chat_messages', 'text', 'retriever', 'vectorstore', 
                           'pdf_file_path', 'upload_pdf', 'qa_chain', 'pdf_processed']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear memory
            st.session_state['memory'].clear()
            
            # Remove vector database
            if os.path.exists(vector_space_dir):
                shutil.rmtree(vector_space_dir)
                os.makedirs(vector_space_dir, exist_ok=True)
            
            # Create new session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state['session_id'] = f"session_{timestamp}_{uuid.uuid4().hex[:6]}"
            st.session_state['chat_messages'] = []
            
            st.success("✅ Session cleared successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error clearing session: {str(e)}")
    
    # --- App Info ---
    st.markdown("---")
    st.markdown("### 🛠️ App Info")
    st.markdown("""
    - Upload PDF documents to chat about their content
    - Maintains conversation history
    - Supports general conversations
    - Session persistence
    """)
