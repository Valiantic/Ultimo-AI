"""
Configuration Management Module

This module handles all application configuration using Pydantic Settings.
It loads environment variables from .env file and provides type-safe access
to configuration values throughout the application.

Key Features:
- Automatically loads .env file
- Provides default values
- Type validation
- Easy access to all config values
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application Settings
    
    This class defines all configuration variables needed for the application.
    Pydantic automatically loads these from environment variables or .env file.
    
    Attributes:
        gemini_api_key: Google Gemini API key for embeddings and LLM (REQUIRED)
        qdrant_url: Qdrant cloud cluster URL (includes port :6333)
        qdrant_api_key: Qdrant API key for authentication
        frontend_url: Frontend URL for CORS configuration
        collection_name: Name of the Qdrant collection to store vectors
        chunk_size: Size of text chunks for splitting documents
        chunk_overlap: Overlap between chunks to maintain context
        embedding_model: Google Gemini embedding model to use
        llm_model: Google Gemini LLM model for generation
    """
    
    # API Keys - REQUIRED
    # Google Gemini API key - REQUIRED
    gemini_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    
    # CORS Configuration
    frontend_url: str = "http://localhost:3000"
    
    # Qdrant Configuration
    collection_name: str = "pdf_documents"
    
    # Document Processing Configuration
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap to maintain context between chunks
    
    # Model Configuration
    embedding_model: str = "models/text-embedding-004"  # Google Gemini embedding model
    llm_model: str = "gemini-2.0-flash-exp"  # Latest Gemini model
    
    # Model Parameters
    temperature: float = 0.3  # Lower = more focused, Higher = more creative
    max_output_tokens: int = 2048  # Maximum response length
    
    class Config:
        """
        Pydantic Config Class
        
        Tells Pydantic where to find the .env file.
        env_file: Path to .env file (relative to where app runs)
        """
        env_file = ".env"
        case_sensitive = False  # Allow case-insensitive env var names


# Create a global settings instance
# This single instance is imported throughout the application
# Benefits:
# 1. Only loads .env file once
# 2. Singleton pattern - one source of truth
# 3. Easy to test by mocking this instance
settings = Settings()


def get_settings() -> Settings:
    """
    Dependency Injection Function
    
    This function is used with FastAPI's dependency injection system.
    It allows easy testing by overriding this function with mock settings.
    
    Returns:
        Settings: The global settings instance
    
    Example Usage:
        @app.get("/some-endpoint")
        async def endpoint(settings: Settings = Depends(get_settings)):
            api_key = settings.gemini_api_key
    """
    return settings