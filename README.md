# PDF Chatbot: A RAG Pipeline

This project is a demonstration of a simple Retrieval Augmented Generation (RAG) pipeline, allowing you to interactively chat with your PDF documents using natural language. The pipeline is implemented in a Jupyter/Colab notebook and leverages state-of-the-art open-source tools for PDF processing, semantic search, and text generation.

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

### 1. Install Dependencies

In your Jupyter or Colab environment, install the required packages:

```python
%pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
%pip install -q transformers accelerate bitsandbytes
%pip install -q sentence-transformers
%pip install -q PyMuPDF tqdm pandas
%pip install -q faiss-cpu
```

### 2. Upload PDF

Use the upload widget in the notebook to upload your PDF file. The path will be set to a variable (e.g., `PDF_PATH`).

### 3. Run the Pipeline

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

- Supports arbitrary PDF documents.
- Provides cited sources (page numbers) for each answer.
- Efficient retrieval and context construction using FAISS and sentence transformers.
- Memory-efficient LLM inference using quantization (where supported).

## Customization

- Swap in different LLMs or embedding models by changing the model name in setup.
- Adjust chunk size and retrieval `top_k` for your use case.

## Limitations

- Designed for running in Colab or Jupyter with adequate resources (GPU recommended for LLM inference).
- Only processes text-based PDFs (not scanned images/OCR).
- Not intended as a production system; for research/demo purposes.

## References

- [PDF_Chatbot_A_RAG_Pipeline.ipynb](https://github.com/anargh-t/PDF-Chatbot-A-RAG-Pipeline/blob/main/PDF_Chatbot_A_RAG_Pipeline.ipynb)

---

**Author:** [anargh-t](https://github.com/anargh-t)
