// src/App.js - Updated React frontend with real backend integration
import React, { useState, useEffect } from 'react';
import { Upload, MessageCircle, History, FileText, Send, Trash2, AlertCircle, CheckCircle } from 'lucide-react';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const RAGApp = () => {
  const [documents, setDocuments] = useState([]);
  const [qaHistory, setQaHistory] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [notification, setNotification] = useState(null);
  const [userId] = useState(() => {
    // Generate or retrieve user ID (in production, use proper auth)
    let id = localStorage.getItem('userId');
    if (!id) {
      id = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('userId', id);
    }
    return id;
  });

  // Load initial data
  useEffect(() => {
    loadDocuments();
    loadHistory();
  }, []);

  const showNotification = (message, type = 'info') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 5000);
  };

  const loadDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents`);
      const data = await response.json();
      if (data.success) {
        setDocuments(data.documents);
      }
    } catch (error) {
      console.error('Error loading documents:', error);
      showNotification('Failed to load documents', 'error');
    }
  };

  const loadHistory = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/history/${userId}`);
      const data = await response.json();
      if (data.success) {
        setQaHistory(data.history);
      }
    } catch (error) {
      console.error('Error loading history:', error);
      showNotification('Failed to load Q&A history', 'error');
    }
  };

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    setIsLoading(true);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      files.forEach(file => {
        if (file.type === 'application/pdf') {
          formData.append('files', file);
        }
      });

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (data.success) {
        showNotification(`Successfully uploaded ${data.documents.length} document(s)`, 'success');
        await loadDocuments();
        setActiveTab('chat');
      } else {
        throw new Error(data.detail || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      showNotification('Failed to upload PDF files', 'error');
    } finally {
      setIsLoading(false);
      setUploadProgress(0);
      event.target.value = '';
    }
  };

  const handleQuestionSubmit = async () => {
    if (!currentQuestion.trim() || documents.length === 0) return;

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: currentQuestion,
          user_id: userId
        }),
      });

      const data = await response.json();
      
      if (data.success) {
        setCurrentQuestion('');
        await loadHistory();
        setActiveTab('history');
        showNotification('Question answered successfully!', 'success');
      } else {
        throw new Error(data.error || 'Failed to process question');
      }
    } catch (error) {
      console.error('Question processing error:', error);
      showNotification('Failed to process your question', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const deleteDocument = async (docId) => {
    if (!window.confirm('Are you sure you want to delete this document?')) return;

    try {
      const response = await fetch(`${API_BASE_URL}/documents/${docId}`, {
        method: 'DELETE',
      });

      const data = await response.json();
      
      if (data.success) {
        await loadDocuments();
        showNotification('Document deleted successfully', 'success');
      } else {
        throw new Error(data.error || 'Failed to delete document');
      }
    } catch (error) {
      console.error('Delete error:', error);
      showNotification('Failed to delete document', 'error');
    }
  };

  const clearHistory = async () => {
    if (!window.confirm('Are you sure you want to clear all Q&A history?')) return;
    
    // This would require a backend endpoint to clear history
    // For now, just reload
    setQaHistory([]);
    showNotification('History cleared', 'success');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuestionSubmit();
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-50 min-h-screen">
      {/* Notification */}
      {notification && (
        <div className={`fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 flex items-center space-x-2 ${
          notification.type === 'success' ? 'bg-green-100 text-green-800' :
          notification.type === 'error' ? 'bg-red-100 text-red-800' :
          'bg-blue-100 text-blue-800'
        }`}>
          {notification.type === 'success' ? <CheckCircle className="h-5 w-5" /> :
           notification.type === 'error' ? <AlertCircle className="h-5 w-5" /> :
           <AlertCircle className="h-5 w-5" />}
          <span>{notification.message}</span>
        </div>
      )}

      <div className="bg-white rounded-lg shadow-lg">
        {/* Header */}
        <div className="border-b border-gray-200 p-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">RAG-based PDF Q&A System</h1>
          <p className="text-gray-600">Upload PDFs, ask questions, and get AI-powered answers with context retrieval using Gemini API.</p>
          <div className="mt-2 text-sm text-gray-500">
            User ID: {userId} | Documents: {documents.length} | Q&A History: {qaHistory.length}
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex border-b border-gray-200">
          <button
            onClick={() => setActiveTab('upload')}
            className={`px-6 py-3 font-medium border-b-2 transition-colors ${
              activeTab === 'upload'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <Upload className="inline w-5 h-5 mr-2" />
            Upload Documents
          </button>
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-6 py-3 font-medium border-b-2 transition-colors ${
              activeTab === 'chat'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <MessageCircle className="inline w-5 h-5 mr-2" />
            Ask Questions
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`px-6 py-3 font-medium border-b-2 transition-colors ${
              activeTab === 'history'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <History className="inline w-5 h-5 mr-2" />
            Q&A History ({qaHistory.length})
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <div className="space-y-6">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors">
                <input
                  type="file"
                  multiple
                  accept=".pdf"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                  disabled={isLoading}
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                  <p className="text-lg font-medium text-gray-900 mb-2">Upload PDF Documents</p>
                  <p className="text-gray-500">Click to select or drag and drop your PDF files here</p>
                  <p className="text-sm text-gray-400 mt-2">Maximum file size: 10MB per file</p>
                </label>
              </div>

              {/* Upload Progress */}
              {isLoading && (
                <div className="bg-blue-50 p-4 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                    <div className="flex-1">
                      <p className="text-blue-800 font-medium">Processing PDF files...</p>
                      <p className="text-blue-600 text-sm">Extracting text and creating embeddings</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Document List */}
              {documents.length > 0 && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-gray-900">Uploaded Documents ({documents.length})</h3>
                  <div className="grid gap-4">
                    {documents.map((doc) => (
                      <div key={doc.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <FileText className="h-8 w-8 text-red-500" />
                          <div>
                            <p className="font-medium text-gray-900">{doc.filename}</p>
                            <p className="text-sm text-gray-500">
                              Uploaded {new Date(doc.upload_date).toLocaleDateString()} • {doc.total_chunks} chunks
                            </p>
                          </div>
                        </div>
                        <button
                          onClick={() => deleteDocument(doc.id)}
                          className="p-2 text-gray-400 hover:text-red-500 transition-colors"
                          title="Delete document"
                        >
                          <Trash2 className="h-5 w-5" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Chat Tab */}
          {activeTab === 'chat' && (
            <div className="space-y-6">
              {documents.length === 0 ? (
                <div className="text-center py-12">
                  <FileText className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                  <p className="text-lg text-gray-600">No documents uploaded yet</p>
                  <p className="text-gray-500">Upload some PDF documents first to start asking questions</p>
                  <button
                    onClick={() => setActiveTab('upload')}
                    className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Upload Documents
                  </button>
                </div>
              ) : (
                <>
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <p className="text-blue-800">
                      <strong>{documents.length}</strong> document(s) loaded and indexed. Ask questions about their content!
                    </p>
                    <p className="text-blue-600 text-sm mt-1">
                      Powered by Gemini AI with vector similarity search
                    </p>
                  </div>

                  <div className="space-y-4">
                    <div className="flex space-x-4">
                      <textarea
                        value={currentQuestion}
                        onChange={(e) => setCurrentQuestion(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Ask a question about your documents... (Press Enter to send)"
                        className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                        rows="3"
                        disabled={isLoading}
                      />
                      <button
                        onClick={handleQuestionSubmit}
                        disabled={isLoading || !currentQuestion.trim()}
                        className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center space-x-2 h-fit"
                      >
                        {isLoading ? (
                          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                        ) : (
                          <Send className="h-5 w-5" />
                        )}
                        <span>{isLoading ? 'Processing...' : 'Ask'}</span>
                      </button>
                    </div>
                    
                    {isLoading && (
                      <div className="bg-yellow-50 p-4 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-yellow-600"></div>
                          <div>
                            <p className="text-yellow-800 font-medium">Processing your question...</p>
                            <p className="text-yellow-600 text-sm">Finding relevant context and generating answer</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Recent Q&A Preview */}
                  {qaHistory.length > 0 && (
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="font-medium text-gray-900 mb-2">Most Recent Q&A:</h4>
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-gray-700">Q: {qaHistory[0].question}</p>
                        <p className="text-sm text-gray-600">A: {qaHistory[0].answer.substring(0, 200)}...</p>
                        <p className="text-xs text-gray-500">
                          {new Date(qaHistory[0].created_at).toLocaleString()}
                        </p>
                      </div>
                      <button
                        onClick={() => setActiveTab('history')}
                        className="mt-2 text-blue-600 hover:text-blue-800 text-sm font-medium"
                      >
                        View all history →
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* History Tab */}
          {activeTab === 'history' && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-900">Q&A History</h3>
                {qaHistory.length > 0 && (
                  <button
                    onClick={clearHistory}
                    className="px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors flex items-center space-x-2"
                  >
                    <Trash2 className="h-4 w-4" />
                    <span>Clear History</span>
                  </button>
                )}
              </div>

              {qaHistory.length === 0 ? (
                <div className="text-center py-12">
                  <History className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                  <p className="text-lg text-gray-600">No questions asked yet</p>
                  <p className="text-gray-500">Start asking questions about your documents to build your history</p>
                </div>
              ) : (
                <div className="space-y-6">
                  {qaHistory.map((qa) => (
                    <div key={qa.id} className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                      <div className="flex justify-between items-start mb-4">
                        <h4 className="font-medium text-gray-900">Question:</h4>
                        <span className="text-sm text-gray-500">
                          {new Date(qa.created_at).toLocaleDateString()} at{' '}
                          {new Date(qa.created_at).toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-gray-700 mb-4 bg-blue-50 p-3 rounded-lg">{qa.question}</p>
                      
                      <h4 className="font-medium text-gray-900 mb-2">Answer:</h4>
                      <div className="text-gray-700 whitespace-pre-wrap bg-gray-50 p-4 rounded-lg mb-4">
                        {qa.answer}
                      </div>
                      
                      {qa.context && (
                        <details className="mt-4">
                          <summary className="cursor-pointer text-sm font-medium text-gray-600 hover:text-gray-800">
                            View source context
                          </summary>
                          <div className="mt-2 p-3 bg-gray-100 rounded text-xs text-gray-600 max-h-40 overflow-y-auto">
                            {qa.context}
                          </div>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Setup Instructions */}
      <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Setup Status & Instructions</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-800 mb-2">Backend Status:</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span>React Frontend Connected</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
                <span>Express Backend (Run server.js)</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
                <span>Supabase Database</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
                <span>Gemini API Integration</span>
              </div>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-800 mb-2">Quick Start:</h4>
            <ol className="list-decimal list-inside space-y-1 text-sm text-gray-600">
              <li>Set up Supabase database with provided schema</li>
              <li>Configure environment variables</li>
              <li>Run backend server: npm run dev</li>
              <li>Upload PDF documents</li>
              <li>Start asking questions!</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RAGApp;

// ===== PACKAGE.JSON for React frontend =====
/*
{
  "name": "rag-frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "@testing-library/jest-dom": "^5.17.0",
    "@testing-library/react": "^13.4.0",
    "@testing-library/user-event": "^13.5.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "lucide-react": "^0.263.1",
    "web-vitals": "^2.1.4"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "proxy": "http://localhost:5000"
}
*/

// ===== .ENV for React frontend =====
/*
REACT_APP_API_URL=http://localhost:5000/api
*/