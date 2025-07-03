# main.py - Enhanced FastAPI backend for RAG application (PDF + Website support)
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Union
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
import aiohttp
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from typing import Set

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG PDF & Website Q&A API",
    description="Backend API for RAG-based PDF and website question answering system",
    version="1.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # In production, specify your frontend URL
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
model = genai.GenerativeModel('gemini-1.5-flash')
embedding_model = genai.GenerativeModel('embedding-001')

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    user_id: str = "anonymous"

class WebsiteRequest(BaseModel):
    url: HttpUrl
    max_depth: int = 2  # How deep to crawl (1 = single page, 2 = page + direct links)
    max_pages: int = 10  # Maximum pages to crawl
    include_external_links: bool = False  # Whether to follow external links
    user_id: str = "anonymous"

class BulkWebsiteRequest(BaseModel):
    urls: List[HttpUrl]
    max_depth: int = 1  # For bulk, usually just single pages
    max_pages_per_site: int = 5
    user_id: str = "anonymous"

class QuestionResponse(BaseModel):
    success: bool
    answer: str
    sources: List[dict] = []
    qa_id: Optional[int] = None

class WebsiteResponse(BaseModel):
    success: bool
    message: str
    documents: List[dict] = []
    pages_processed: int = 0

class HistoryResponse(BaseModel):
    success: bool
    history: List[dict] = []

class DocumentResponse(BaseModel):
    success: bool
    documents: List[dict] = []

class DeleteResponse(BaseModel):
    success: bool
    message: str = ""

# Website scraping utilities
class WebsiteScraper:
    def __init__(self):
        self.session = None
        self.visited_urls: Set[str] = set()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scrape_url(self, url: str) -> dict:
        """Scrape content from a single URL"""
        try:
            async with self.session.get(str(url)) as response:
                if response.status != 200:
                    return {"error": f"HTTP {response.status}"}
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                
                # Extract title
                title = soup.find('title')
                title = title.get_text().strip() if title else "Untitled"
                
                # Extract main content
                content = ""
                
                # Try to find main content areas
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article'))
                if main_content:
                    content = main_content.get_text()
                else:
                    # Fallback to body
                    body = soup.find('body')
                    if body:
                        content = body.get_text()
                
                # Clean up content
                content = re.sub(r'\s+', ' ', content).strip()
                
                # Extract links for potential crawling
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(str(url), href)
                    links.append(full_url)
                
                return {
                    "url": str(url),
                    "title": title,
                    "content": content,
                    "links": links,
                    "word_count": len(content.split()),
                    "scraped_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    async def crawl_website(self, start_url: str, max_depth: int = 2, max_pages: int = 10, 
                           include_external: bool = False) -> List[dict]:
        """Crawl website with depth and page limits"""
        results = []
        to_visit = [(str(start_url), 0)]  # (url, depth)
        base_domain = urlparse(str(start_url)).netloc
        
        while to_visit and len(results) < max_pages:
            current_url, depth = to_visit.pop(0)
            
            if current_url in self.visited_urls or depth > max_depth:
                continue
                
            self.visited_urls.add(current_url)
            
            # Scrape current URL
            result = await self.scrape_url(current_url)
            
            if "error" not in result and result.get("content"):
                results.append(result)
                
                # Add links to visit if we haven't reached max depth
                if depth < max_depth:
                    for link in result.get("links", []):
                        link_domain = urlparse(link).netloc
                        
                        # Skip if external link and not allowed
                        if not include_external and link_domain != base_domain:
                            continue
                        
                        # Skip if already visited or queued
                        if link not in self.visited_urls and link not in [url for url, _ in to_visit]:
                            to_visit.append((link, depth + 1))
            
            # Add small delay to be respectful
            await asyncio.sleep(0.5)
        
        return results

# Website scraping utilities
class WebsiteScraper:
    def __init__(self):
        self.session = None
        self.visited_urls: Set[str] = set()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scrape_url(self, url: str) -> dict:
        """Scrape content from a single URL"""
        try:
            async with self.session.get(str(url)) as response:
                if response.status != 200:
                    return {"error": f"HTTP {response.status}"}
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                
                # Extract title
                title = soup.find('title')
                title = title.get_text().strip() if title else "Untitled"
                
                # Extract main content
                content = ""
                
                # Try to find main content areas
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article'))
                if main_content:
                    content = main_content.get_text()
                else:
                    # Fallback to body
                    body = soup.find('body')
                    if body:
                        content = body.get_text()
                
                # Clean up content
                content = re.sub(r'\s+', ' ', content).strip()
                
                # Extract links for potential crawling
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(str(url), href)
                    links.append(full_url)
                
                return {
                    "url": str(url),
                    "title": title,
                    "content": content,
                    "links": links,
                    "word_count": len(content.split()),
                    "scraped_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    async def crawl_website(self, start_url: str, max_depth: int = 2, max_pages: int = 10, 
                           include_external: bool = False) -> List[dict]:
        """Crawl website with depth and page limits"""
        results = []
        to_visit = [(str(start_url), 0)]  # (url, depth)
        base_domain = urlparse(str(start_url)).netloc
        
        while to_visit and len(results) < max_pages:
            current_url, depth = to_visit.pop(0)
            
            if current_url in self.visited_urls or depth > max_depth:
                continue
                
            self.visited_urls.add(current_url)
            
            # Scrape current URL
            result = await self.scrape_url(current_url)
            
            if "error" not in result and result.get("content"):
                results.append(result)
                
                # Add links to visit if we haven't reached max depth
                if depth < max_depth:
                    for link in result.get("links", []):
                        link_domain = urlparse(link).netloc
                        
                        # Skip if external link and not allowed
                        if not include_external and link_domain != base_domain:
                            continue
                        
                        # Skip if already visited or queued
                        if link not in self.visited_urls and link not in [url for url, _ in to_visit]:
                            to_visit.append((link, depth + 1))
            
            # Add small delay to be respectful
            await asyncio.sleep(0.5)
        
        return results

# Helper functions (keeping existing PDF functions)
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
            if last_sentence > chunk_size * 0.5:
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
        result = await asyncio.to_thread(
            genai.embed_content,
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Gemini embedding failed: {str(e)}, using TF-IDF fallback")
        return create_tfidf_embedding(text)

def create_tfidf_embedding(text: str, max_features: int = 1000) -> List[float]:
    """Create a simple TF-IDF based embedding as fallback"""
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)
    total_words = len(words)
    
    vocab = [word for word, _ in word_freq.most_common(min(max_features, len(word_freq)))]
    
    embedding = []
    for word in vocab[:max_features]:
        tf = word_freq.get(word, 0) / total_words
        idf = math.log(1 + 1 / (word_freq.get(word, 0) + 1))
        embedding.append(tf * idf)
    
    while len(embedding) < max_features:
        embedding.append(0.0)
    embedding = embedding[:max_features]
    
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
    processed_embeddings = []
    for embedding in doc_embeddings:
        if isinstance(embedding, str):
            try:
                embedding_str = embedding.replace("np.str_('", "").replace("')", "")
                embedding_list = [float(x) for x in embedding_str.strip('[]').split(',')]
                processed_embeddings.append(embedding_list)
            except:
                continue
        elif isinstance(embedding, list):
            processed_embeddings.append(embedding)
        else:
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
        prompt = f"""Based on the following context from uploaded documents and websites, please answer the question.

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

async def process_chunk(doc_id: str, chunk_index: int, content: str):
    """Process individual chunk - create embedding and store"""
    try:
        embedding = await create_embedding(content)
        
        if isinstance(embedding, str):
            embedding = [float(x) for x in embedding.strip('[]').split(',')]
        elif not isinstance(embedding, list):
            embedding = list(embedding)
        
        chunk_data = {
            "document_id": doc_id,
            "chunk_index": chunk_index,
            "content": content,
            "embedding": embedding
        }
        
        result = supabase.table("document_chunks").insert(chunk_data).execute()
        return result.data[0] if result.data else None
        
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {str(e)}")
        return None

# API Routes

@app.get("/", response_model=dict)
async def root():
    """Health check endpoint"""
    return {"message": "RAG PDF & Website Q&A API is running", "status": "healthy"}

@app.post("/api/upload", response_model=dict)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process PDF files"""
    try:
        uploaded_docs = []
        
        for file in files:
            if not file.content_type == "application/pdf":
                continue
                
            content = await file.read()
            text = extract_text_from_pdf(content)
            
            if not text.strip():
                continue
                
            chunks = chunk_text(text)
            doc_id = generate_document_id()
            
            doc_data = {
                "id": doc_id,
                "filename": file.filename,
                "source_type": "pdf",
                "source_url": None,
                "upload_date": datetime.now().isoformat(),
                "total_chunks": len(chunks)
            }
            
            result = supabase.table("documents").insert(doc_data).execute()
            if not result.data:
                raise HTTPException(status_code=500, detail="Failed to store document metadata")
            
            # Process chunks
            chunk_tasks = []
            for i, chunk in enumerate(chunks):
                chunk_tasks.append(process_chunk(doc_id, i, chunk))
            
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

@app.post("/api/website", response_model=WebsiteResponse)
async def scrape_website(request: WebsiteRequest):
    """Scrape and process a website"""
    try:
        async with WebsiteScraper() as scraper:
            # Crawl the website
            scraped_pages = await scraper.crawl_website(
                str(request.url), 
                request.max_depth, 
                request.max_pages,
                request.include_external_links
            )
            
            if not scraped_pages:
                raise HTTPException(status_code=400, detail="No content could be extracted from the website")
            
            processed_docs = []
            
            for page in scraped_pages:
                if not page.get("content"):
                    continue
                
                # Create chunks from page content
                chunks = chunk_text(page["content"])
                doc_id = generate_document_id()
                
                # Store document metadata
                doc_data = {
                    "id": doc_id,
                    "filename": page["title"],
                    "source_type": "website",
                    "source_url": page["url"],
                    "upload_date": datetime.now().isoformat(),
                    "total_chunks": len(chunks),
                    "metadata": {
                        "word_count": page.get("word_count", 0),
                        "scraped_at": page.get("scraped_at")
                    }
                }
                
                result = supabase.table("documents").insert(doc_data).execute()
                if not result.data:
                    continue
                
                # Process chunks
                chunk_tasks = []
                for i, chunk in enumerate(chunks):
                    chunk_tasks.append(process_chunk(doc_id, i, chunk))
                
                # Process chunks in batches
                batch_size = 5
                for i in range(0, len(chunk_tasks), batch_size):
                    batch = chunk_tasks[i:i + batch_size]
                    await asyncio.gather(*batch)
                
                processed_docs.append(result.data[0])
            
            return WebsiteResponse(
                success=True,
                message=f"Successfully processed {len(processed_docs)} pages from the website",
                documents=processed_docs,
                pages_processed=len(processed_docs)
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Website scraping failed: {str(e)}")

@app.post("/api/websites/bulk", response_model=WebsiteResponse)
async def scrape_multiple_websites(request: BulkWebsiteRequest):
    """Scrape multiple websites"""
    try:
        all_processed_docs = []
        total_pages = 0
        
        async with WebsiteScraper() as scraper:
            for url in request.urls:
                try:
                    # Scrape each website
                    scraped_pages = await scraper.crawl_website(
                        str(url), 
                        request.max_depth, 
                        request.max_pages_per_site,
                        False  # Don't follow external links for bulk processing
                    )
                    
                    for page in scraped_pages:
                        if not page.get("content"):
                            continue
                        
                        chunks = chunk_text(page["content"])
                        doc_id = generate_document_id()
                        
                        doc_data = {
                            "id": doc_id,
                            "filename": page["title"],
                            "source_type": "website",
                            "source_url": page["url"],
                            "upload_date": datetime.now().isoformat(),
                            "total_chunks": len(chunks),
                            "metadata": {
                                "word_count": page.get("word_count", 0),
                                "scraped_at": page.get("scraped_at")
                            }
                        }
                        
                        result = supabase.table("documents").insert(doc_data).execute()
                        if not result.data:
                            continue
                        
                        # Process chunks
                        chunk_tasks = []
                        for i, chunk in enumerate(chunks):
                            chunk_tasks.append(process_chunk(doc_id, i, chunk))
                        
                        batch_size = 5
                        for i in range(0, len(chunk_tasks), batch_size):
                            batch = chunk_tasks[i:i + batch_size]
                            await asyncio.gather(*batch)
                        
                        all_processed_docs.append(result.data[0])
                        total_pages += 1
                
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
                    continue
        
        return WebsiteResponse(
            success=True,
            message=f"Successfully processed {total_pages} pages from {len(request.urls)} websites",
            documents=all_processed_docs,
            pages_processed=total_pages
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk website scraping failed: {str(e)}")

@app.post("/api/website", response_model=WebsiteResponse)
async def scrape_website(request: WebsiteRequest):
    """Scrape and process a website"""
    try:
        async with WebsiteScraper() as scraper:
            # Crawl the website
            scraped_pages = await scraper.crawl_website(
                str(request.url), 
                request.max_depth, 
                request.max_pages,
                request.include_external_links
            )
            
            if not scraped_pages:
                raise HTTPException(status_code=400, detail="No content could be extracted from the website")
            
            processed_docs = []
            
            for page in scraped_pages:
                if not page.get("content"):
                    continue
                
                # Create chunks from page content
                chunks = chunk_text(page["content"])
                doc_id = generate_document_id()
                
                # Store document metadata
                doc_data = {
                    "id": doc_id,
                    "filename": page["title"],
                    "source_type": "website",
                    "source_url": page["url"],
                    "upload_date": datetime.now().isoformat(),
                    "total_chunks": len(chunks),
                    "metadata": {
                        "word_count": page.get("word_count", 0),
                        "scraped_at": page.get("scraped_at")
                    }
                }
                
                result = supabase.table("documents").insert(doc_data).execute()
                if not result.data:
                    continue
                
                # Process chunks
                chunk_tasks = []
                for i, chunk in enumerate(chunks):
                    chunk_tasks.append(process_chunk(doc_id, i, chunk))
                
                # Process chunks in batches
                batch_size = 5
                for i in range(0, len(chunk_tasks), batch_size):
                    batch = chunk_tasks[i:i + batch_size]
                    await asyncio.gather(*batch)
                
                processed_docs.append(result.data[0])
            
            return WebsiteResponse(
                success=True,
                message=f"Successfully processed {len(processed_docs)} pages from the website",
                documents=processed_docs,
                pages_processed=len(processed_docs)
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Website scraping failed: {str(e)}")

@app.post("/api/websites/bulk", response_model=WebsiteResponse)
async def scrape_multiple_websites(request: BulkWebsiteRequest):
    """Scrape multiple websites"""
    try:
        all_processed_docs = []
        total_pages = 0
        
        async with WebsiteScraper() as scraper:
            for url in request.urls:
                try:
                    # Scrape each website
                    scraped_pages = await scraper.crawl_website(
                        str(url), 
                        request.max_depth, 
                        request.max_pages_per_site,
                        False  # Don't follow external links for bulk processing
                    )
                    
                    for page in scraped_pages:
                        if not page.get("content"):
                            continue
                        
                        chunks = chunk_text(page["content"])
                        doc_id = generate_document_id()
                        
                        doc_data = {
                            "id": doc_id,
                            "filename": page["title"],
                            "source_type": "website",
                            "source_url": page["url"],
                            "upload_date": datetime.now().isoformat(),
                            "total_chunks": len(chunks),
                            "metadata": {
                                "word_count": page.get("word_count", 0),
                                "scraped_at": page.get("scraped_at")
                            }
                        }
                        
                        result = supabase.table("documents").insert(doc_data).execute()
                        if not result.data:
                            continue
                        
                        # Process chunks
                        chunk_tasks = []
                        for i, chunk in enumerate(chunks):
                            chunk_tasks.append(process_chunk(doc_id, i, chunk))
                        
                        batch_size = 5
                        for i in range(0, len(chunk_tasks), batch_size):
                            batch = chunk_tasks[i:i + batch_size]
                            await asyncio.gather(*batch)
                        
                        all_processed_docs.append(result.data[0])
                        total_pages += 1
                
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
                    continue
        
        return WebsiteResponse(
            success=True,
            message=f"Successfully processed {total_pages} pages from {len(request.urls)} websites",
            documents=all_processed_docs,
            pages_processed=total_pages
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk website scraping failed: {str(e)}")

@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Process question and generate answer using RAG"""
    try:
        # Create embedding for the question
        question_embedding = await create_query_embedding(request.question)
        
        # Retrieve all document chunks with embeddings
        result = supabase.table("document_chunks").select(
            "*, documents(filename, source_type, source_url)"
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
            doc = chunk['documents']
            if doc['source_type'] == 'pdf':
                source_info = f"PDF: {doc['filename']}"
            else:
                source_info = f"Website: {doc['filename']} ({doc['source_url']})"
            
            context_parts.append(f"Source: {source_info}\nContent: {chunk['content']}")
            sources.append({
                "filename": doc['filename'],
                "source_type": doc['source_type'],
                "source_url": doc['source_url'],
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
        # Get document counts by type
        doc_result = supabase.table("documents").select("source_type").execute()
        docs = doc_result.data or []
        
        pdf_count = len([d for d in docs if d.get("source_type") == "pdf"])
        website_count = len([d for d in docs if d.get("source_type") == "website"])
        
        # Get chunk count
        chunk_result = supabase.table("document_chunks").select("id", count="exact").execute()
        chunk_count = chunk_result.count or 0
        
        # Get Q&A count
        qa_result = supabase.table("qa_history").select("id", count="exact").execute()
        qa_count = qa_result.count or 0
        
        return {
            "success": True,
            "stats": {
                "total_documents": len(docs),
                "pdf_documents": pdf_count,
                "website_documents": website_count,
                "total_chunks": chunk_count,
                "total_questions": qa_count
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)