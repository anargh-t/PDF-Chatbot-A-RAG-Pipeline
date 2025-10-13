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

- [`PyMuPDF`](https://pymupdf.readthedocs.io/en/latest/) (`fitz`): For PDF text extraction.
- [`sentence-transformers`](https://www.sbert.net/): For creating semantic embeddings of text.
- [`transformers`](https://huggingface.co/docs/transformers/index) and [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes): For loading and managing LLMs, supporting quantization for low-memory environments.
- [`faiss-cpu`](https://github.com/facebookresearch/faiss): For building the vector index and fast similarity search.
- `pandas`, `numpy`, `tqdm`: For data handling and progress bars.

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
- 📄 Easy PDF upload interface
- 💬 Interactive chat with context
- 📚 Source citations showing which pages were used
- ⚙️ Adjustable settings for retrieval parameters
- 🎨 Clean, responsive UI

### Option 2: Jupyter Notebook

For a more hands-on approach or experimentation, use the Jupyter notebook:

#### 1. Install Dependencies

In your Jupyter or Colab environment, install the required packages:

```python
%pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
%pip install -q transformers accelerate bitsandbytes
%pip install -q sentence-transformers
%pip install -q PyMuPDF tqdm pandas
%pip install -q faiss-cpu
```

#### 2. Upload PDF

Use the upload widget in the notebook to upload your PDF file. The path will be set to a variable (e.g., `PDF_PATH`).

#### 3. Run the Pipeline

The notebook will guide you through:

- Extracting and chunking the PDF text.
- Generating embeddings and building the FAISS index.
- Setting up the LLM.
- Chatting with your document!

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

- 🌐 **User-friendly Web UI** built with Streamlit
- 📄 Supports arbitrary PDF documents
- 📚 Provides cited sources (page numbers) for each answer
- 💬 Interactive chat with context and history
- 🔍 Efficient retrieval and context construction using FAISS and sentence transformers
- 🤖 Memory-efficient LLM inference using quantization (where supported)
- ⚙️ Adjustable retrieval parameters (top_k)
- 🎨 Responsive and clean interface

## Customization

- Swap in different LLMs or embedding models by changing the model name in setup.
- Adjust chunk size and retrieval `top_k` for your use case.

## Project Structure

```
PDF-Chatbot-A-RAG-Pipeline/
├── app.py                              # Streamlit web application
├── rag_utils.py                        # Core RAG pipeline utilities
├── PDF_Chatbot_A_RAG_Pipeline.ipynb   # Original Jupyter notebook
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

## Limitations

- Requires adequate system resources (GPU recommended for faster LLM inference, but CPU works too).
- Only processes text-based PDFs (not scanned images/OCR).
- Not intended as a production system; for research/demo purposes.
- First model load may take several minutes and download ~2GB of data.

## References

- [PDF_Chatbot_A_RAG_Pipeline.ipynb](https://github.com/anargh-t/PDF-Chatbot-A-RAG-Pipeline/blob/main/PDF_Chatbot_A_RAG_Pipeline.ipynb)

---

**Author:** [anargh-t](https://github.com/anargh-t)
