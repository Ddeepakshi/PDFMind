# main.py - FastAPI backend for RAG application (Gemini-only version)
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import asyncio
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from supabase import create_client, Client
import PyPDF2
import io
import re
import hashlib
from collections import Counter
import math
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG PDF Q&A API",
    description="Backend API for RAG-based PDF question answering system (Gemini-only)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure Google Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')  # Use Gemini's latest model for generation
embedding_model = genai.GenerativeModel('embedding-001')

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    user_id: str = "anonymous"

class QuestionResponse(BaseModel):
    success: bool
    answer: str
    sources: List[dict] = []
    qa_id: Optional[int] = None

class HistoryResponse(BaseModel):
    success: bool
    history: List[dict] = []

class DocumentResponse(BaseModel):
    success: bool
    documents: List[dict] = []

class DeleteResponse(BaseModel):
    success: bool
    message: str = ""

# Helper functions
def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        
        chunk = text[start:end]
        
        # Try to break at sentence boundaries
        if end < len(text):
            last_sentence = chunk.rfind('.')
            if last_sentence > chunk_size * 0.5:  # Only if we don't lose too much text
                end = start + last_sentence + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        
        if end >= len(text):
            break
            
        start = end - overlap
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]

async def create_embedding(text: str) -> List[float]:
    """Create embedding using Gemini embedding model"""
    try:
        # Use Gemini's embedding model
        result = await asyncio.to_thread(
            genai.embed_content,
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        # Fallback to TF-IDF based embedding if Gemini embedding fails
        print(f"Gemini embedding failed: {str(e)}, using TF-IDF fallback")
        return create_tfidf_embedding(text)

def create_tfidf_embedding(text: str, max_features: int = 1000) -> List[float]:
    """Create a simple TF-IDF based embedding as fallback"""
    # Simple tokenization
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Create word frequency vector
    word_freq = Counter(words)
    total_words = len(words)
    
    # Get top words for vocabulary
    vocab = [word for word, _ in word_freq.most_common(min(max_features, len(word_freq)))]
    
    # Create TF-IDF vector
    embedding = []
    for word in vocab[:max_features]:
        tf = word_freq.get(word, 0) / total_words
        # Simple IDF approximation
        idf = math.log(1 + 1 / (word_freq.get(word, 0) + 1))
        embedding.append(tf * idf)
    
    # Pad or truncate to fixed size
    while len(embedding) < max_features:
        embedding.append(0.0)
    embedding = embedding[:max_features]
    
    # Normalize
    norm = math.sqrt(sum(x*x for x in embedding))
    if norm > 0:
        embedding = [x/norm for x in embedding]
    
    return embedding

async def create_query_embedding(text: str) -> List[float]:
    """Create embedding for query text"""
    try:
        result = await asyncio.to_thread(
            genai.embed_content,
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Gemini query embedding failed: {str(e)}, using TF-IDF fallback")
        return create_tfidf_embedding(text)

def calculate_similarity(query_embedding: List[float], doc_embeddings: List[List[float]]) -> List[float]:
    """Calculate cosine similarity between query and document embeddings"""
    # Convert doc_embeddings to proper format if they're strings
    processed_embeddings = []
    for embedding in doc_embeddings:
        if isinstance(embedding, str):
            # Handle string representation of numpy array
            try:
                # Remove numpy array wrapper and convert to list
                embedding_str = embedding.replace("np.str_('", "").replace("')", "")
                embedding_list = [float(x) for x in embedding_str.strip('[]').split(',')]
                processed_embeddings.append(embedding_list)
            except:
                # If conversion fails, skip this embedding
                continue
        elif isinstance(embedding, list):
            processed_embeddings.append(embedding)
        else:
            # Try to convert to list
            try:
                processed_embeddings.append(list(embedding))
            except:
                continue
    
    if not processed_embeddings:
        return []
    
    query_array = np.array(query_embedding).reshape(1, -1)
    doc_array = np.array(processed_embeddings)
    similarities = cosine_similarity(query_array, doc_array)[0]
    return similarities.tolist()

async def generate_answer(question: str, context: str) -> str:
    """Generate answer using Gemini AI"""
    try:
        prompt = f"""Based on the following context from uploaded documents, please answer the question.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so and provide what information is available.

Answer:"""

        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

def generate_document_id() -> str:
    """Generate unique document ID"""
    return f"doc_{int(datetime.now().timestamp())}_{str(uuid.uuid4())[:8]}"

# API Routes

@app.get("/", response_model=dict)
async def root():
    """Health check endpoint"""
    return {"message": "RAG PDF Q&A API is running", "status": "healthy"}

@app.post("/api/upload", response_model=dict)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process PDF files"""
    try:
        uploaded_docs = []
        
        for file in files:
            if not file.content_type == "application/pdf":
                continue
                
            # Read file content
            content = await file.read()
            
            # Extract text from PDF
            text = extract_text_from_pdf(content)
            
            if not text.strip():
                continue
                
            # Create text chunks
            chunks = chunk_text(text)
            
            # Generate document ID
            doc_id = generate_document_id()
            
            # Store document metadata
            doc_data = {
                "id": doc_id,
                "filename": file.filename,
                "upload_date": datetime.now().isoformat(),
                "total_chunks": len(chunks)
            }
            
            result = supabase.table("documents").insert(doc_data).execute()
            if not result.data:
                raise HTTPException(status_code=500, detail="Failed to store document metadata")
            
            # Process chunks and create embeddings
            chunk_tasks = []
            for i, chunk in enumerate(chunks):
                chunk_tasks.append(process_chunk(doc_id, i, chunk))
            
            # Process chunks concurrently (but limit concurrency to avoid rate limits)
            chunk_results = []
            batch_size = 5
            for i in range(0, len(chunk_tasks), batch_size):
                batch = chunk_tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch)
                chunk_results.extend(batch_results)
            
            uploaded_docs.append(result.data[0])
        
        return {"success": True, "documents": uploaded_docs}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_chunk(doc_id: str, chunk_index: int, content: str):
    """Process individual chunk - create embedding and store"""
    try:
        # Create embedding
        embedding = await create_embedding(content)
        
        # Ensure embedding is a proper list of floats
        if isinstance(embedding, str):
            # Convert string to list if needed
            embedding = [float(x) for x in embedding.strip('[]').split(',')]
        elif not isinstance(embedding, list):
            embedding = list(embedding)
        
        # Store chunk with embedding
        chunk_data = {
            "document_id": doc_id,
            "chunk_index": chunk_index,
            "content": content,
            "embedding": embedding  # This should now be a proper list
        }
        
        result = supabase.table("document_chunks").insert(chunk_data).execute()
        return result.data[0] if result.data else None
        
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {str(e)}")
        return None

@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Process question and generate answer using RAG"""
    try:
        # Create embedding for the question
        question_embedding = await create_query_embedding(request.question)
        
        # Retrieve all document chunks with embeddings
        result = supabase.table("document_chunks").select(
            "*, documents(filename)"
        ).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="No documents found")
        
        # Calculate similarities
        chunk_embeddings = [chunk["embedding"] for chunk in result.data]
        similarities = calculate_similarity(question_embedding, chunk_embeddings)
        
        # Get top 5 most relevant chunks
        chunk_similarity_pairs = list(zip(result.data, similarities))
        chunk_similarity_pairs.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_similarity_pairs[:5]
        
        # Prepare context
        context_parts = []
        sources = []
        
        for chunk, similarity in top_chunks:
            context_parts.append(f"Document: {chunk['documents']['filename']}\nContent: {chunk['content']}")
            sources.append({
                "filename": chunk['documents']['filename'],
                "similarity": round(float(similarity), 3)
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer using Gemini
        answer = await generate_answer(request.question, context)
        
        # Save Q&A to database
        qa_data = {
            "user_id": request.user_id,
            "question": request.question,
            "answer": answer,
            "context": context,
            "created_at": datetime.now().isoformat()
        }
        
        qa_result = supabase.table("qa_history").insert(qa_data).execute()
        qa_id = qa_result.data[0]["id"] if qa_result.data else None
        
        return QuestionResponse(
            success=True,
            answer=answer,
            sources=sources,
            qa_id=qa_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")

@app.get("/api/history/{user_id}", response_model=HistoryResponse)
async def get_history(user_id: str = "anonymous"):
    """Get Q&A history for a user"""
    try:
        result = supabase.table("qa_history").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        
        return HistoryResponse(
            success=True,
            history=result.data or []
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@app.get("/api/documents", response_model=DocumentResponse)
async def get_documents():
    """Get all uploaded documents"""
    try:
        result = supabase.table("documents").select("*").order("upload_date", desc=True).execute()
        
        return DocumentResponse(
            success=True,
            documents=result.data or []
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")

@app.delete("/api/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """Delete a document and its chunks"""
    try:
        # Delete chunks first (foreign key constraint)
        supabase.table("document_chunks").delete().eq("document_id", document_id).execute()
        
        # Delete document
        result = supabase.table("documents").delete().eq("id", document_id).execute()
        
        return DeleteResponse(
            success=True,
            message="Document deleted successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.delete("/api/history/{user_id}", response_model=DeleteResponse)
async def clear_history(user_id: str):
    """Clear Q&A history for a user"""
    try:
        supabase.table("qa_history").delete().eq("user_id", user_id).execute()
        
        return DeleteResponse(
            success=True,
            message="History cleared successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

@app.get("/api/stats", response_model=dict)
async def get_stats():
    """Get application statistics"""
    try:
        # Get document count
        doc_result = supabase.table("documents").select("id", count="exact").execute()
        doc_count = doc_result.count or 0
        
        # Get chunk count
        chunk_result = supabase.table("document_chunks").select("id", count="exact").execute()
        chunk_count = chunk_result.count or 0
        
        # Get Q&A count
        qa_result = supabase.table("qa_history").select("id", count="exact").execute()
        qa_count = qa_result.count or 0
        
        return {
            "success": True,
            "stats": {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "total_questions": qa_count
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,reload=True)


