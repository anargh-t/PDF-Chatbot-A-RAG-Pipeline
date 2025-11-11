# PDF Chatbot: A RAG Pipeline

This project is a demonstration of a simple Retrieval Augmented Generation (RAG) pipeline, allowing you to interactively chat with your PDF documents using natural language. The pipeline is available both as a Jupyter/Colab notebook and as a **user-friendly Streamlit web application**. It leverages state-of-the-art open-source tools for PDF processing, semantic search, and text generation.

## Overview

The pipeline follows these main steps:

1. **PDF Processing**: Extracts text from PDF files using `PyMuPDF`.
2. **Text Chunking**: Splits extracted text into manageable, overlapping chunks for better retrieval and context.
3. **Embeddings**: Converts each text chunk into a numerical vector using a Sentence Transformer model.
4. **Vector Store**: Stores the embeddings in a FAISS index for efficient similarity search.
5. **Retrieval**: Given a user query, retrieves the most relevant text chunks using vector similarity.
6. **LLM Setup**: Loads a Large Language Model (LLM) for answer generation (e.g., TinyLlama, DialoGPT), with optional quantization for efficiency.
7. **Generation**: The LLM generates answers based on the user's query and the retrieved relevant context from the document.

## Key Libraries

* PyMuPDF (`fitz`): For PDF text extraction.
* sentence-transformers: For creating semantic embeddings of text.
* transformers and bitsandbytes: For loading and managing LLMs, supporting quantization for low-memory environments.
* faiss-cpu: For building the vector index and fast similarity search.
* `pandas`, `numpy`, `tqdm`: For data handling and progress bars.

## Setup Instructions

### Option 1: Streamlit Web App (Recommended)

The easiest way to use the PDF Chatbot is through the Streamlit web interface:

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Run the Streamlit App

```bash
streamlit run app.py
```

#### 3. Use the Web Interface

1. Upload a PDF document using the sidebar
2. Click "Process PDF" to index the document
3. Ask questions in the chat interface
4. View answers with source page citations

The Streamlit app provides:

* 📄 Easy PDF upload interface
* 💬 Interactive chat with context
* 📚 Source citations showing which pages were used
* ⚙️ Adjustable settings for retrieval parameters
* 🎨 Clean, responsive UI

### Option 2: Jupyter Notebook

For a more hands-on approach or experimentation, use the Jupyter notebook:

#### 1. Install Dependencies

In your Jupyter or Colab environment, install the required packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes
pip install sentence-transformers
pip install PyMuPDF tqdm pandas
pip install faiss-cpu
```

#### 2. Upload PDF

Use the upload widget in the notebook to upload your PDF file. The path will be set to a variable (e.g., `PDF_PATH`).

#### 3. Run the Pipeline

The notebook will guide you through:

* Extracting and chunking the PDF text.
* Generating embeddings and building the FAISS index.
* Setting up the LLM.
* Chatting with your document!

## Example Usage

```python
# Example: Ask a question about the uploaded PDF
result = rag_pipeline(
    "What are the main topics discussed in this document?",
    embedding_model, index, chunks, model, tokenizer
)
print(result["answer"])
print("Sources:", result["sources"])
```

### Interactive Chat

The notebook provides an interactive chat loop:

```python
interactive_chat()
```

Type your questions to the chatbot and receive answers with references to the relevant pages.

## Features

* 🌐 **User-friendly Web UI** built with Streamlit
* 📄 Supports arbitrary PDF documents
* 📚 Provides cited sources (page numbers) for each answer
* 💬 Interactive chat with context and history
* 🔍 Efficient retrieval and context construction using FAISS and sentence transformers
* 🤖 Memory-efficient LLM inference using quantization (where supported)
* ⚙️ Adjustable retrieval parameters (top_k)
* 🎨 Responsive and clean interface

## Customization

* Swap in different LLMs or embedding models by changing the model name in setup.
* Adjust chunk size and retrieval `top_k` for your use case.

## Project Structure

```
PDF-Chatbot-A-RAG-Pipeline/
├── app.py                              # Streamlit web application
├── rag_utils.py                        # Core RAG pipeline utilities
├── PDF_Chatbot_A_RAG_Pipeline.ipynb   # Original Jupyter notebook
├── requirements.txt                    # Python dependencies
├── DEMO.md                             # Detailed demo guide
├── QUICKSTART.md                       # Quick start guide
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

## Limitations

* Requires adequate system resources (GPU recommended for faster LLM inference, but CPU works too).
* Only processes text-based PDFs (not scanned images/OCR).
* Not intended as a production system; for research/demo purposes.
* First model load may take several minutes and download ~2GB of data.

## Configuration

The application uses the following models:

- **LLM**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Embeddings**: `all-MiniLM-L6-v2` (384 dimensions)
- **Quantization**: Optional for memory efficiency

## Using the Jupyter Notebook

The notebook (`PDF_Chatbot_A_RAG_Pipeline.ipynb`) includes **automatic PDF validation** at each step:

### Validation Features
- ✅ **PDF Upload Verification**: Checks if PDF was uploaded correctly
- ✅ **File Path Validation**: Ensures PDF path is set and file exists  
- ✅ **Extension Check**: Validates PDF file extension
- ✅ **Component Verification**: Verifies all RAG components are loaded before querying

### Helper Functions
- `verify_pdf_setup()` - Check if PDF is properly loaded
- `verify_rag_setup()` - Verify all RAG components are ready

Run these functions at any time to check the status before proceeding.

## Additional Resources

- [DEMO.md](DEMO.md) - Detailed guide on using the Streamlit web interface
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide for getting started in 3 steps
- [Jupyter Notebook](PDF_Chatbot_A_RAG_Pipeline.ipynb) - Interactive implementation with PDF validation

## References

* PDF_Chatbot_A_RAG_Pipeline.ipynb

---

**Author:** [anargh-t](https://github.com/anargh-t)
