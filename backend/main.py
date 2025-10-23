"""
FastAPI Main Application

This is the entry point of the backend application.
It defines all API endpoints and handles HTTP requests/responses.

Key Components:
- FastAPI app initialization
- CORS middleware for frontend communication
- API endpoints for upload, query, and health check
- Error handling and logging
- RAG service integration

To run: uvicorn main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import List

# Import our modules
from config import Settings, get_settings
from models import (
    QueryRequest, 
    QueryResponse, 
    UploadResponse, 
    HealthResponse,
    ErrorResponse
)
from rag_service import RAGService

# Configure logging
# This helps debug issues in production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A RAG-based chatbot API that answers questions about uploaded PDF documents",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# Initialize settings
settings = get_settings()

# Initialize RAG Service (singleton)
# This is created once when the app starts
rag_service = RAGService(settings)

# Configure CORS (Cross-Origin Resource Sharing)
# This allows the frontend to communicate with the backend
# Without CORS, browsers block requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,  # Production frontend
        "http://localhost:3000",  # Local development
        "http://127.0.0.1:3000"   # Alternative localhost
    ],
    allow_credentials=True,  # Allow cookies
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.on_event("startup")
async def startup_event():
    """
    Startup Event Handler
    
    Runs once when the application starts.
    Used for:
    - Initializing connections
    - Running health checks
    - Logging startup info
    """
    logger.info("Starting RAG Chatbot API...")
    logger.info(f"Frontend URL: {settings.frontend_url}")
    logger.info(f"Qdrant URL: {settings.qdrant_url}")
    logger.info(f"Collection: {settings.collection_name}")
    logger.info(f"LLM Model: {settings.llm_model}")
    
    try:
        # Verify connections on startup
        await rag_service.health_check()
        logger.info("All services healthy!")
    except Exception as e:
        logger.error(f"Startup health check failed: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown Event Handler
    
    Runs when the application is shutting down.
    Used for:
    - Closing connections
    - Cleanup
    - Logging shutdown info
    """
    logger.info("Shutting down RAG Chatbot API...")


@app.get("/", response_model=HealthResponse)
async def root():
    """
    Root Endpoint
    
    Simple endpoint to verify the API is running.
    
    Returns:
        HealthResponse: Status message
    """
    return HealthResponse(
        status="healthy",
        message="RAG Chatbot API is running. Visit /docs for API documentation."
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health Check Endpoint
    
    Verifies that all services (Qdrant, Gemini) are accessible.
    Used by:
    - Monitoring systems
    - Load balancers
    - Deployment platforms (Render, AWS, etc.)
    
    Returns:
        HealthResponse: Health status
        
    Raises:
        HTTPException: If any service is unavailable
    """
    try:
        await rag_service.health_check()
        return HealthResponse(
            status="healthy",
            message="All services are operational"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload and process")
):
    """
    Upload PDF Endpoint
    
    This endpoint handles PDF uploads and processes them for RAG.
    
    Process:
    1. Validate file is PDF
    2. Read file content
    3. Extract text using PyPDF2
    4. Split text into chunks
    5. Create embeddings
    6. Store in Qdrant vector database
    
    Args:
        file: Uploaded PDF file
        
    Returns:
        UploadResponse: Success message with processing details
        
    Raises:
        HTTPException: If file is invalid or processing fails
        
    Example:
        curl -X POST "http://localhost:8000/upload-pdf" \
             -H "Content-Type: multipart/form-data" \
             -F "file=@document.pdf"
    """
    try:
        logger.info(f"📥 UPLOAD START: Received file '{file.filename}'")
        
        # Validate file type
        logger.info(f"🔍 STEP 1/6: Validating file type...")
        if not file.filename.endswith('.pdf'):
            logger.error(f"❌ Invalid file type: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        logger.info(f"✅ File type validated: PDF")
        
        # Validate file size (max 10MB)
        logger.info(f"🔍 STEP 2/6: Reading file content...")
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        logger.info(f"✅ File read successfully: {file_size_mb:.2f}MB")
        
        if file_size_mb > 10:
            logger.error(f"❌ File too large: {file_size_mb:.2f}MB")
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({file_size_mb:.2f}MB). Maximum size is 10MB"
            )
        
        logger.info(f"🔍 STEP 3/6: Starting PDF processing...")
        
        # Process PDF
        chunks_created = await rag_service.process_pdf(
            pdf_file=file_content,
            filename=file.filename
        )
        
        logger.info(f"🎉 UPLOAD COMPLETE: {file.filename} - {chunks_created} chunks created")
        
        return UploadResponse(
            message="PDF uploaded and processed successfully",
            filename=file.filename,
            chunks_created=chunks_created,
            collection_name=settings.collection_name
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"💥 ERROR processing PDF: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query Documents Endpoint
    
    This endpoint answers questions about uploaded documents using RAG.
    
    Process:
    1. Receive question from user
    2. Convert question to embedding
    3. Search Qdrant for relevant document chunks (retrieval)
    4. Send chunks + question to Gemini (generation)
    5. Return answer with sources
    
    Args:
        request: QueryRequest with question and optional chat history
        
    Returns:
        QueryResponse: Answer with source documents
        
    Raises:
        HTTPException: If query fails
        
    Example:
        curl -X POST "http://localhost:8000/query" \
             -H "Content-Type: application/json" \
             -d '{
                "question": "What is the main topic?",
                "chat_history": []
             }'
    """
    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Convert chat history to list of tuples
        chat_history = [
            (q, a) for q, a in (request.chat_history or [])
        ]
        
        # Query RAG service
        answer, sources = await rag_service.query(
            question=request.question,
            chat_history=chat_history
        )
        
        logger.info(f"Query successful. Sources used: {len(sources)}")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=None  # Can add confidence scoring later
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global Exception Handler
    
    Catches any unhandled exceptions and returns a structured error response.
    This prevents the API from returning raw error messages.
    
    Args:
        request: The request that caused the exception
        exc: The exception that was raised
        
    Returns:
        JSONResponse: Formatted error response
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred. Please try again.",
            "status_code": 500
        }
    )


# For local development and testing
if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    # This is only used when running: python main.py
    # In production, we use: uvicorn main:app
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,
        reload=True  # Auto-reload on code changes (development only)
    )