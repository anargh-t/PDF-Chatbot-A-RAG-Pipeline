"""
Streamlit Web UI for PDF Chatbot RAG Pipeline
A user-friendly interface to upload PDFs, ask questions, and get answers with source citations.
"""

import streamlit as st
import tempfile
import os
from rag_utils import (
    open_and_read_pdf,
    create_chunks_from_pages,
    create_embeddings,
    create_faiss_index,
    setup_llm,
    rag_pipeline
)

# Set page configuration
st.set_page_config(
    page_title="PDF Chatbot - RAG Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #f0f2f6;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .sources {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None


def process_pdf(uploaded_file):
    """Process uploaded PDF and create embeddings"""
    with st.spinner("Processing PDF..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Step 1: Extract text from PDF
            st.info("📄 Extracting text from PDF...")
            pages = open_and_read_pdf(tmp_path)
            st.success(f"✅ Extracted {len(pages)} pages")
            
            # Step 2: Create chunks
            st.info("✂️ Creating text chunks...")
            chunks = create_chunks_from_pages(pages, chunk_size=1000, overlap=200)
            st.success(f"✅ Created {len(chunks)} chunks")
            
            # Step 3: Create embeddings
            st.info("🔢 Creating embeddings...")
            embeddings, embedding_model = create_embeddings(chunks)
            st.success(f"✅ Created embeddings with shape: {embeddings.shape}")
            
            # Step 4: Create FAISS index
            st.info("🔍 Creating search index...")
            index = create_faiss_index(embeddings)
            st.success(f"✅ Search index created with {index.ntotal} vectors")
            
            # Store in session state
            st.session_state.chunks = chunks
            st.session_state.embedding_model = embedding_model
            st.session_state.index = index
            st.session_state.pdf_processed = True
            st.session_state.pdf_name = uploaded_file.name
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)


def setup_model():
    """Setup the LLM model if not already loaded"""
    if st.session_state.model is None:
        with st.spinner("Loading language model (this may take a few minutes)..."):
            st.info("🤖 Loading TinyLlama model...")
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            model, tokenizer = setup_llm(model_name, use_quantization=False)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.success("✅ Language model loaded successfully!")


def display_chat_message(role, content, sources=None):
    """Display a chat message with styling"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = ""
        if sources:
            sources_str = ", ".join(sources)
            sources_html = f'<div class="sources">📚 Sources: {sources_str}</div>'
        
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Bot:</strong> {content}
            {sources_html}
        </div>
        """, unsafe_allow_html=True)


# Main app layout
st.title("📚 PDF Chatbot - RAG Pipeline")
st.markdown("Ask questions about your PDF documents and get answers with source citations!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # PDF Upload
    st.subheader("1. Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to chat with"
    )
    
    if uploaded_file is not None:
        if st.button("🔄 Process PDF", key="process_btn"):
            process_pdf(uploaded_file)
    
    # Show PDF status
    if st.session_state.pdf_processed:
        st.success(f"✅ PDF loaded: {st.session_state.pdf_name}")
        st.info(f"📊 {len(st.session_state.chunks)} chunks indexed")
    
    st.divider()
    
    # Settings
    st.subheader("2. Settings")
    top_k = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="How many relevant text chunks to use for answering"
    )
    
    if st.session_state.pdf_processed:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    st.divider()
    
    # Info
    st.subheader("ℹ️ About")
    st.markdown("""
    This app uses a Retrieval Augmented Generation (RAG) pipeline to answer questions about your PDF documents.
    
    **How it works:**
    1. Upload a PDF document
    2. The text is extracted and split into chunks
    3. Chunks are converted to embeddings
    4. Ask questions in natural language
    5. Relevant chunks are retrieved
    6. An AI generates an answer with sources
    """)

# Main content area
if not st.session_state.pdf_processed:
    st.info("👈 Please upload a PDF file from the sidebar to get started!")
    
    # Show example
    with st.expander("📖 See Example Usage"):
        st.markdown("""
        ### Example Questions:
        - "What are the main topics discussed in this document?"
        - "Can you summarize the key findings?"
        - "What does the document say about [specific topic]?"
        
        ### Features:
        - 📄 Upload any PDF document
        - 💬 Interactive chat interface
        - 📚 Source citations (page numbers)
        - 🔍 Semantic search using embeddings
        - 🤖 AI-powered answer generation
        """)
else:
    # Setup model if needed
    if st.session_state.model is None:
        setup_model()
    
    # Display chat history
    st.subheader("💬 Chat")
    
    # Display previous messages
    for message in st.session_state.chat_history:
        display_chat_message(
            message['role'],
            message['content'],
            message.get('sources')
        )
    
    # Question input
    st.divider()
    question = st.text_input(
        "Ask a question about your PDF:",
        placeholder="E.g., What are the main topics discussed?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🚀 Ask", type="primary")
    
    if ask_button and question.strip():
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': question
        })
        
        # Get answer from RAG pipeline
        with st.spinner("🤔 Thinking..."):
            try:
                result = rag_pipeline(
                    question,
                    st.session_state.embedding_model,
                    st.session_state.index,
                    st.session_state.chunks,
                    st.session_state.model,
                    st.session_state.tokenizer,
                    top_k=top_k
                )
                
                # Add bot message to history
                st.session_state.chat_history.append({
                    'role': 'bot',
                    'content': result['answer'],
                    'sources': result['sources']
                })
                
                # Rerun to display new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try rephrasing your question or check if the PDF was processed correctly.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>
        Built with Streamlit • Powered by Sentence Transformers, FAISS, and TinyLlama
    </small>
</div>
""", unsafe_allow_html=True)
