"""
Utility functions for the PDF Chatbot RAG Pipeline.
Extracted from the Jupyter notebook for reuse in the Streamlit app and FastAPI server.
"""

import torch
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import faiss
from tqdm.auto import tqdm
import re
import warnings
warnings.filterwarnings('ignore')


def text_formatter(text: str) -> str:
    """Clean and format text"""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single newline
    return text.strip()


def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF and return list of pages"""
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num in tqdm(range(len(doc)), desc="Reading PDF pages"):
        page = doc[page_num]
        text = page.get_text()
        text = text_formatter(text)

        if text:  # Only add non-empty pages
            pages_text.append({
                'page_number': page_num + 1,
                'text': text,
                'char_count': len(text)
            })

    doc.close()
    return pages_text


def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            search_start = max(start, end - 100)
            sentence_end = text.rfind('.', search_start, end)
            if sentence_end > start:
                end = sentence_end + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


def create_chunks_from_pages(pages: list[dict], chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Create chunks from PDF pages"""
    all_chunks = []

    for page in tqdm(pages, desc="Creating chunks"):
        chunks = split_text_into_chunks(page['text'], chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'page_number': page['page_number'],
                'chunk_id': f"page_{page['page_number']}_chunk_{i+1}",
                'text': chunk,
                'char_count': len(chunk)
            })

    return all_chunks


def create_embeddings(chunks: list[dict], model_name: str = "all-MiniLM-L6-v2") -> tuple:
    """Create embeddings for text chunks"""
    print(f"Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)

    # Extract texts
    texts = [chunk['text'] for chunk in chunks]

    print("Creating embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    return embeddings, embedding_model


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create FAISS index for similarity search"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))

    return index


def retrieve_relevant_chunks(query: str,
                           embedding_model: SentenceTransformer,
                           index: faiss.Index,
                           chunks: list[dict],
                           top_k: int = 5) -> list[dict]:
    """Retrieve most relevant chunks for a query"""
    # Create query embedding
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)

    # Search for similar chunks
    scores, indices = index.search(query_embedding.astype('float32'), top_k)

    # Get relevant chunks with scores
    relevant_chunks = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        chunk = chunks[idx].copy()
        chunk['similarity_score'] = float(score)
        chunk['rank'] = i + 1
        relevant_chunks.append(chunk)

    return relevant_chunks


def setup_llm(model_name: str = "microsoft/DialoGPT-medium", use_quantization: bool = True):
    """Setup LLM for text generation"""
    print(f"Loading LLM: {model_name}")

    # Setup quantization for memory efficiency
    if use_quantization and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_answer(query: str,
                   relevant_chunks: list[dict],
                   model,
                   tokenizer,
                   max_length: int = 512) -> str:
    """Generate answer using retrieved context"""
    # Create context from relevant chunks
    context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])

    # Create prompt
    prompt = f"""Context: {context}

Question: {query}

Answer: Based on the provided context, """

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)

    # Move to same device as model
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the answer part
    answer = response.split("Answer: Based on the provided context, ")[-1]

    return answer.strip()


def rag_pipeline(query: str,
                embedding_model: SentenceTransformer,
                index: faiss.Index,
                chunks: list[dict],
                model,
                tokenizer,
                top_k: int = 5) -> dict:
    """Complete RAG pipeline"""
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, embedding_model, index, chunks, top_k)

    # Generate answer
    answer = generate_answer(query, relevant_chunks, model, tokenizer)

    return {
        'query': query,
        'answer': answer,
        'relevant_chunks': relevant_chunks,
        'sources': [f"Page {chunk['page_number']}" for chunk in relevant_chunks]
    }
