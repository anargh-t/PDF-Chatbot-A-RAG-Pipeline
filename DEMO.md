# Streamlit Web UI Demo Guide

This guide demonstrates how to use the PDF Chatbot Streamlit Web UI.

## Starting the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Using the Web Interface

### 1. Upload a PDF Document

On the left sidebar:
- Click "Browse files" or drag and drop a PDF
- Click the "🔄 Process PDF" button
- Wait while the app:
  - Extracts text from all pages
  - Creates text chunks with overlap
  - Generates embeddings for semantic search
  - Builds a FAISS index for fast retrieval

### 2. Ask Questions

In the main chat area:
- Type your question in the text input box
- Click the "🚀 Ask" button
- The chatbot will:
  - Search for relevant chunks from your PDF
  - Generate a contextual answer using the LLM
  - Display source pages used to answer your question

### 3. Chat History

- All questions and answers are preserved in the chat history
- Scroll up to view previous conversations
- Click "🗑️ Clear Chat History" in the sidebar to start fresh

### 4. Adjust Settings

In the sidebar:
- **Number of sources to retrieve**: Control how many relevant chunks to use (1-10)
- Higher values provide more context but may be slower

## Features Demonstrated

### Interactive Chat Interface
```
You: What are the main topics in this document?

🤖 Bot: The document discusses machine learning concepts, including 
supervised learning, unsupervised learning, and reinforcement learning. 
It also covers applications in healthcare, finance, and transportation.

📚 Sources: Page 1, Page 1, Page 2
```

### Source Citations
Every answer includes page numbers showing which parts of the PDF were used, allowing you to:
- Verify the accuracy of answers
- Reference specific sections of the document
- Understand which context informed the response

### Session Management
- Upload different PDFs without restarting the app
- Chat history is maintained per session
- Models are loaded once and reused for efficiency

## Example Use Cases

### 1. Research Paper Analysis
```
Upload: research_paper.pdf
Ask: "What methodology did the authors use?"
Result: Detailed answer with specific page citations
```

### 2. Legal Document Review
```
Upload: contract.pdf
Ask: "What are the termination conditions?"
Result: Precise answer referencing relevant clauses
```

### 3. Technical Documentation
```
Upload: user_manual.pdf
Ask: "How do I configure the network settings?"
Result: Step-by-step answer with page references
```

## Technical Details

### Processing Pipeline
1. **PDF Upload** → Text extraction with PyMuPDF
2. **Chunking** → 1000 char chunks with 200 char overlap
3. **Embedding** → all-MiniLM-L6-v2 (384 dimensions)
4. **Indexing** → FAISS with cosine similarity
5. **Retrieval** → Top-k semantic search
6. **Generation** → TinyLlama 1.1B chat model

### Performance Notes
- First PDF processing: ~30-60 seconds
- Model loading (first time): ~2-5 minutes
- Subsequent queries: ~2-5 seconds per answer
- Models cached after first load

### Memory Requirements
- Minimum: 4GB RAM
- Recommended: 8GB+ RAM
- GPU: Optional (speeds up generation, not required)

## Troubleshooting

### "Please upload a PDF file to get started"
- Upload a PDF using the sidebar uploader
- Click "Process PDF" button

### Slow response times
- Reduce the number of sources (top_k slider)
- Use a machine with more RAM/CPU
- Consider using GPU for faster inference

### Model download takes long
- First run downloads ~2GB of models
- Models are cached in `~/.cache/huggingface/`
- Subsequent runs use cached models

## Advanced Usage

### Running on Custom Port
```bash
streamlit run app.py --server.port 8080
```

### Running on Remote Server
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Development Mode (auto-reload)
```bash
streamlit run app.py --server.runOnSave true
```

## Keyboard Shortcuts

- `Ctrl + Enter` / `Cmd + Enter`: Submit question
- `R`: Refresh/rerun the app
- `Esc`: Clear text input

## Screenshots

(In a real deployment, screenshots would be included here showing:
- Initial upload screen
- PDF processing progress
- Chat interface with questions and answers
- Source citations display
- Settings sidebar)
