"""
Pydantic Data Models Module

This module defines all request and response models for the API.
Pydantic models provide:
- Automatic validation
- Type checking
- JSON serialization/deserialization
- API documentation in Swagger UI

These models define the "contract" between frontend and backend.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """
    Request Model for Question Answering
    
    Used when the user asks a question about their uploaded documents.
    
    Attributes:
        question: The user's question
        chat_history: Optional list of previous Q&A pairs for context
    
    Example JSON:
        {
            "question": "What is the main topic of the document?",
            "chat_history": [
                ["Previous question?", "Previous answer."]
            ]
        }
    """
    question: str = Field(
        ..., 
        description="The question to ask about the uploaded documents",
        min_length=1,
        max_length=500,
        example="What are the key findings in this document?"
    )
    
    chat_history: Optional[List[List[str]]] = Field(
        default=[],
        description="Previous conversation history as [question, answer] pairs",
        example=[["What is this about?", "This document discusses AI technology."]]
    )
    
    class Config:
        """Configuration for this model"""
        json_schema_extra = {
            "example": {
                "question": "What are the main topics covered?",
                "chat_history": []
            }
        }


class QueryResponse(BaseModel):
    """
    Response Model for Question Answering
    
    Returned after processing a user's question.
    
    Attributes:
        answer: The AI-generated answer
        sources: List of text chunks that were used to generate the answer
        confidence: Optional confidence score (0-1)
    
    Example JSON:
        {
            "answer": "The document discusses...",
            "sources": ["Relevant text chunk 1", "Relevant text chunk 2"],
            "confidence": 0.95
        }
    """
    answer: str = Field(
        ..., 
        description="The generated answer to the question"
    )
    
    sources: List[str] = Field(
        default=[],
        description="List of source text chunks used to generate the answer"
    )
    
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    
    class Config:
        """Configuration for this model"""
        json_schema_extra = {
            "example": {
                "answer": "The main topics are AI, machine learning, and data science.",
                "sources": [
                    "Chapter 1 introduces artificial intelligence...",
                    "Machine learning is a subset of AI..."
                ],
                "confidence": 0.92
            }
        }


class UploadResponse(BaseModel):
    """
    Response Model for File Upload
    
    Returned after successfully uploading and processing a PDF.
    
    Attributes:
        message: Success message
        filename: Name of the uploaded file
        chunks_created: Number of text chunks created from the document
        collection_name: Name of the Qdrant collection where vectors are stored
    
    Example JSON:
        {
            "message": "PDF uploaded and processed successfully",
            "filename": "document.pdf",
            "chunks_created": 45,
            "collection_name": "pdf_documents"
        }
    """
    message: str = Field(
        ...,
        description="Status message"
    )
    
    filename: str = Field(
        ...,
        description="Name of the uploaded file"
    )
    
    chunks_created: int = Field(
        ...,
        description="Number of text chunks created and stored",
        ge=0
    )
    
    collection_name: str = Field(
        ...,
        description="Qdrant collection name where vectors are stored"
    )
    
    class Config:
        """Configuration for this model"""
        json_schema_extra = {
            "example": {
                "message": "PDF uploaded and processed successfully",
                "filename": "research_paper.pdf",
                "chunks_created": 45,
                "collection_name": "pdf_documents"
            }
        }


class HealthResponse(BaseModel):
    """
    Response Model for Health Check
    
    Simple response to verify the API is running.
    
    Attributes:
        status: Health status (typically "healthy")
        message: Additional information
    
    Example JSON:
        {
            "status": "healthy",
            "message": "RAG Chatbot API is running"
        }
    """
    status: str = Field(
        ...,
        description="Health status of the API"
    )
    
    message: str = Field(
        ...,
        description="Additional status information"
    )
    
    class Config:
        """Configuration for this model"""
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "RAG Chatbot API is running"
            }
        }


class ErrorResponse(BaseModel):
    """
    Response Model for Errors
    
    Standardized error response for all API errors.
    
    Attributes:
        error: Error type or category
        detail: Detailed error message
        status_code: HTTP status code
    
    Example JSON:
        {
            "error": "ValidationError",
            "detail": "Question cannot be empty",
            "status_code": 422
        }
    """
    error: str = Field(
        ...,
        description="Error type"
    )
    
    detail: str = Field(
        ...,
        description="Detailed error message"
    )
    
    status_code: int = Field(
        ...,
        description="HTTP status code"
    )
    
    class Config:
        """Configuration for this model"""
        json_schema_extra = {
            "example": {
                "error": "ProcessingError",
                "detail": "Failed to process PDF file",
                "status_code": 500
            }
        }