# RAG PDF & Website Q&A System

A powerful **Retrieval-Augmented Generation (RAG)** system that allows you to upload PDF documents, scrape websites, and ask intelligent questions about the content. Built with **FastAPI**, **React**, **Supabase**, and **Google Gemini AI**.

![RAG System Demo](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![React](https://img.shields.io/badge/React-18+-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688)

## ✨ Features

### 📄 **Multiple Content Sources**
- **PDF Upload**: Upload and process PDF documents
- **Website Scraping**: Extract content from websites with configurable depth
- **Bulk Processing**: Process multiple websites simultaneously
- **Smart Content Extraction**: Focuses on main content areas, removes navigation/ads

### 🤖 **AI-Powered Q&A**
- **Gemini AI Integration**: Powered by Google's latest Gemini model
- **Semantic Search**: Vector similarity search using embeddings
- **Context-Aware Answers**: Provides comprehensive answers with source attribution
- **Multi-Source Responses**: Combines information from PDFs and websites

### 🔧 **Advanced Features**
- **Real-time Processing**: Async processing for better performance
- **Configurable Crawling**: Control depth, page limits, and external link following
- **Source Attribution**: Clear indication of whether answers come from PDFs or websites
- **Q&A History**: Track all your questions and answers
- **Statistics Dashboard**: Monitor your document library

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │   Supabase DB   │
│                 │    │                 │    │                 │
│  • Upload UI    │◄──►│  • PDF Processing│◄──►│  • Documents    │
│  • Chat Interface│    │  • Web Scraping │    │  • Chunks       │
│  • History      │    │  • Embeddings   │    │  • Q&A History  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Gemini AI     │
                       │                 │
                       │  • Embeddings   │
                       │  • Text Gen     │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Node.js 16+**
- **Supabase Account**
- **Google Gemini API Key**

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-pdf-website-qa.git
cd rag-pdf-website-qa
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

Edit `.env` with your credentials:
```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
GEMINI_API_KEY=your_gemini_api_key
```

### 3. Database Setup

Run this SQL in your Supabase SQL editor:

```sql
-- Create documents table
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    source_type TEXT DEFAULT 'pdf',
    source_url TEXT,
    upload_date TIMESTAMP DEFAULT NOW(),
    total_chunks INTEGER DEFAULT 0,
    metadata JSONB
);

-- Create document chunks table
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768)  -- Adjust size based on your embedding model
);

-- Create Q&A history table
CREATE TABLE qa_history (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    context TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_qa_history_user_id ON qa_history(user_id);
CREATE INDEX idx_qa_history_created_at ON qa_history(created_at);
```

### 4. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create .env file
cp .env.example .env
```

Edit frontend `.env`:
```env
REACT_APP_API_URL=http://localhost:8000/api
```

### 5. Run the Application

**Start Backend:**
```bash
python main.py
```

**Start Frontend (in a new terminal):**
```bash
cd frontend
npm start
```

Visit `http://localhost:3000` to use the application!

## 📁 Project Structure

```
rag-pdf-website-qa/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   └── index.js        # React entry point
│   ├── public/
│   ├── package.json        # Node.js dependencies
│   └── .env                # Frontend environment variables
└── README.md
```

## 🔧 Configuration

### Backend Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `SUPABASE_URL` | Your Supabase project URL | ✅ |
| `SUPABASE_SERVICE_KEY` | Your Supabase service role key | ✅ |
| `GEMINI_API_KEY` | Google Gemini API key | ✅ |

### Website Scraping Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_depth` | How deep to crawl links | 2 |
| `max_pages` | Maximum pages per site | 10 |
| `include_external_links` | Follow external links | false |
| `max_pages_per_site` | Pages per site (bulk) | 5 |

## 📚 API Documentation

### Core Endpoints

#### Upload PDF
```http
POST /api/upload
Content-Type: multipart/form-data

files: PDF files
```

#### Scrape Website
```http
POST /api/website
Content-Type: application/json

{
  "url": "https://example.com",
  "max_depth": 2,
  "max_pages": 10,
  "include_external_links": false
}
```

#### Ask Question
```http
POST /api/ask
Content-Type: application/json

{
  "question": "What is the main topic?",
  "user_id": "user123"
}
```

#### Get Statistics
```http
GET /api/stats
```

Full API documentation available at `http://localhost:8000/docs` when running.

## 🎯 Usage Examples

### 1. Upload and Query PDFs

```bash
# Upload PDF
curl -X POST "http://localhost:8000/api/upload" \
  -F "files=@document.pdf"

# Ask question
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "user_id": "user123"}'
```

### 2. Scrape Website and Query

```bash
# Scrape website
curl -X POST "http://localhost:8000/api/website" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "max_depth": 1, "max_pages": 5}'

# Ask question about scraped content
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this website about?", "user_id": "user123"}'
```

### 3. Bulk Website Processing

```bash
curl -X POST "http://localhost:8000/api/websites/bulk" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://site1.com", "https://site2.com"],
    "max_depth": 1,
    "max_pages_per_site": 3
  }'
```

## 🛠️ Dependencies

### Backend
- **FastAPI** - Web framework
- **aiohttp** - Async HTTP client for web scraping
- **beautifulsoup4** - HTML parsing
- **google-generativeai** - Gemini AI integration
- **supabase** - Database client
- **PyPDF2** - PDF processing
- **scikit-learn** - Vector similarity calculations
- **numpy** - Numerical operations

### Frontend
- **React** - UI framework
- **lucide-react** - Icons
- **Tailwind CSS** - Styling (via CDN)

## 🚨 Troubleshooting

### Common Issues

#### 1. "Website not added" Error
- **Check database schema**: Ensure `source_type`, `source_url`, and `metadata` columns exist
- **Test with simple sites**: Try `https://example.com` first
- **Check logs**: Look at FastAPI console for error messages

#### 2. CORS Errors
- **Frontend URL**: Ensure CORS settings include your frontend URL
- **Port mismatch**: Frontend (3000) and backend (8000) ports

#### 3. Database Connection Issues
- **Credentials**: Verify Supabase URL and service key
- **Permissions**: Ensure service key has necessary permissions
- **Network**: Check firewall/network restrictions

#### 4. Gemini API Errors
- **API Key**: Verify your Gemini API key is valid
- **Quota**: Check API usage limits
- **Model**: Ensure you're using the correct model name

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Test individual components:
```bash
# Test backend health
curl http://localhost:8000/

# Test scraping
curl http://localhost:8000/api/test-scrape

# Test database
python -c "from supabase import create_client; print('DB OK')"
```

## 🔐 Security Considerations

- **API Keys**: Never commit API keys to version control
- **Database**: Use service keys with minimal required permissions
- **Rate Limiting**: Implement rate limiting for production use
- **Input Validation**: All user inputs are validated
- **CORS**: Configure CORS for your specific domain in production

## 📈 Performance Optimization

### Backend
- **Async Processing**: All I/O operations are async
- **Batch Processing**: Chunks processed in batches
- **Connection Pooling**: Reuse HTTP connections
- **Caching**: Consider Redis for frequently accessed data

### Frontend
- **Lazy Loading**: Load components as needed
- **Debouncing**: Debounce search inputs
- **Error Boundaries**: Handle errors gracefully
- **Loading States**: Show progress indicators

## 🗺️ Roadmap

### Planned Features
- [ ] **Advanced Search**: Filters, sorting, faceted search
- [ ] **Document Management**: Folders, tags, metadata
- [ ] **User Authentication**: Multi-user support
- [ ] **Export Options**: PDF, Word, CSV exports
- [ ] **Analytics**: Usage statistics, popular queries
- [ ] **Integrations**: Google Drive, Dropbox, SharePoint
- [ ] **Mobile App**: React Native implementation

### Performance Improvements
- [ ] **Caching Layer**: Redis for embeddings and results
- [ ] **Search Optimization**: Elasticsearch integration
- [ ] **Batch Processing**: Queue system for large uploads
- [ ] **CDN**: Static asset optimization

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini AI** for powerful language models
- **Supabase** for excellent database and backend services
- **FastAPI** for the amazing web framework
- **React** team for the frontend framework
- **Beautiful Soup** for HTML parsing capabilities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/rag-pdf-website-qa/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/rag-pdf-website-qa/discussions)
- **Email**: your.email@example.com

---

**⭐ Star this repo if you found it helpful!**

Made with ❤️ by [Your Name](https://github.com/yourusername)
