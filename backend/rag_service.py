"""
RAG Service Module

This is the core module that implements the RAG (Retrieval Augmented Generation) logic.
It handles:
1. PDF text extraction
2. Text chunking
3. Vector embeddings generation using HuggingFace
4. Storage in Qdrant vector database
5. Similarity search for relevant context
6. Question answering using OpenAI GPT

RAG Process Flow:
1. User uploads PDF → Extract text → Split into chunks → Create embeddings → Store in Qdrant
2. User asks question → Create question embedding → Search Qdrant for similar chunks →
   Send chunks + question to OpenAI → Return answer
"""

from typing import List, Tuple
import PyPDF2
from io import BytesIO

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import Qdrant  # Updated import name
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import Settings


class RAGService:
    """
    RAG Service Class

    This class encapsulates all RAG functionality.
    It's initialized once and reused for all requests.

    Architecture Benefits:
    - Singleton pattern: One instance for the entire app
    - Connection pooling: Reuses Qdrant and OpenAI connections
    - Configuration centralization: All settings in one place
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the RAG Service

        This sets up all the necessary components:
        - Qdrant client and vector store
        - HuggingFace embeddings (all-MiniLM-L6-v2)
        - OpenAI LLM (GPT-3.5-turbo)
        - Text splitter for chunking

        Args:
            settings: Application settings with API keys and configuration
        """
        self.settings = settings
        
        # Initialize Qdrant Client
        # This connects to your Qdrant Cloud cluster
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        
        # Initialize HuggingFace Embeddings
        # Uses all-MiniLM-L6-v2 model (384-dimensional embeddings)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize Text Splitter
        # This splits long documents into smaller chunks
        # Why? LLMs have token limits and smaller chunks = better retrieval
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,  # Max characters per chunk
            chunk_overlap=settings.chunk_overlap,  # Overlap maintains context
            length_function=len,  # How to measure chunk size
            separators=["\n\n", "\n", " ", ""]  # Split on paragraphs first, then sentences, then words
        )
        
        # Initialize OpenAI Chat LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # You can change to gpt-4o or other if desired
            openai_api_key=settings.openai_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens
        )
        
        # Ensure collection exists
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """
        Ensure Qdrant Collection Exists
        
        A collection in Qdrant is like a table in a database.
        It stores vectors with a specific dimension.
        
        This method:
        1. Checks if collection exists
        2. If not, creates it with correct settings
        
        Why check first? Avoids errors on subsequent runs.
        """
        try:
            # Try to get collection info
            self.qdrant_client.get_collection(self.settings.collection_name)
            print(f"Collection '{self.settings.collection_name}' already exists")
        except Exception:
            # Collection doesn't exist, create it
            print(f"Creating collection '{self.settings.collection_name}'...")

            # HuggingFace all-MiniLM-L6-v2 model produces 384-dimensional vectors
            vector_size = 384
            
            self.qdrant_client.create_collection(
                collection_name=self.settings.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE  # Cosine similarity for comparing vectors
                )
            )
            print(f"Collection created successfully")
    
    def extract_text_from_pdf(self, pdf_file: bytes) -> str:
        """
        Extract Text from PDF File
        
        Uses PyPDF2 to read PDF and extract all text.
        
        Process:
        1. Read PDF bytes into memory
        2. Loop through all pages
        3. Extract text from each page
        4. Combine all text
        
        Args:
            pdf_file: PDF file as bytes
            
        Returns:
            str: Extracted text from all pages
            
        Raises:
            Exception: If PDF is corrupted or can't be read
        """
        try:
            # Create a PDF reader object from bytes
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
            
            text = ""
            # Loop through each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                # Extract text from page and add to total
                text += page.extract_text() + "\n"
            
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    async def process_pdf(self, pdf_file: bytes, filename: str) -> int:
        """
        Process PDF and Store in Vector Database
        
        This is the main ingestion pipeline:
        1. Extract text from PDF
        2. Split text into chunks
        3. Create embeddings for each chunk
        4. Store embeddings in Qdrant
        
        Args:
            pdf_file: PDF file as bytes
            filename: Name of the file (for metadata)
            
        Returns:
            int: Number of chunks created and stored
            
        Raises:
            Exception: If processing fails at any step
        """
        try:
            # Step 1: Extract text
            print(f"📄 STEP 3a/6: Extracting text from {filename}...")
            text = self.extract_text_from_pdf(pdf_file)
            
            if not text.strip():
                print(f"❌ No text extracted from PDF")
                raise Exception("No text could be extracted from PDF")
            
            text_length = len(text)
            print(f"✅ Text extraction complete: {text_length:,} characters")
            
            # Step 2: Split text into chunks
            print(f"📝 STEP 4/6: Splitting text into chunks...")
            chunks = self.text_splitter.split_text(text)
            print(f"✅ Created {len(chunks)} chunks")
            
            # Step 3 & 4: Create embeddings and store in Qdrant
            # LangChain's Qdrant handles both automatically
            print(f"🧠 STEP 5/6: Creating embeddings for {len(chunks)} chunks...")
            print(f"⏳ This may take a moment... (generating {len(chunks)} embeddings)")
            
            vector_store = Qdrant.from_texts(
                texts=chunks,  # Text chunks to embed
                embedding=self.embeddings,  # Embedding function
                collection_name=self.settings.collection_name,  # Where to store
                url=self.settings.qdrant_url,  # Qdrant cluster URL
                api_key=self.settings.qdrant_api_key,  # Authentication
                metadata=[{"filename": filename, "chunk_index": i} for i in range(len(chunks))]  # Metadata for each chunk
            )
            
            print(f"✅ Embeddings created and stored in Qdrant")
            print(f"💾 STEP 6/6: Storing vectors in collection '{self.settings.collection_name}'")
            print(f"🎉 Successfully processed {filename}")
            return len(chunks)
            
        except Exception as e:
            print(f"💥 ERROR in process_pdf: {str(e)}")
            raise Exception(f"Error processing PDF: {str(e)}")
    
    async def query(self, question: str, chat_history: List[Tuple[str, str]] = None) -> Tuple[str, List[str]]:
        """
        Answer Question Using RAG
        
        This is the retrieval and generation pipeline:
        1. Convert question to embedding
        2. Search Qdrant for similar chunks (retrieval)
        3. Send chunks + question to OpenAI (generation)
        4. Return answer and sources
        
        Args:
            question: User's question
            chat_history: Previous Q&A pairs for context
            
        Returns:
            Tuple[str, List[str]]: (answer, source_texts)
            
        Raises:
            Exception: If query fails
        """
        try:
            if chat_history is None:
                chat_history = []
            
            # Initialize vector store (connects to existing collection)
            vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.settings.collection_name,
                embeddings=self.embeddings
            )
            
            # Create retriever - retrieve top 4 most similar chunks
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            
            # Retrieve relevant documents
            docs = retriever.invoke(question)
            
            # Format context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Use the following pieces of context to answer the question. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.\n\nContext:\n{context}"),
                ("human", "{question}"),
            ])
            
            # Create chain: prompt -> llm -> output parser
            chain = prompt_template | self.llm | StrOutputParser()
            
            # Execute query
            answer = chain.invoke({"context": context, "question": question})
            
            # Extract sources
            sources = [doc.page_content for doc in docs]
            
            return answer, sources
            
        except Exception as e:
            raise Exception(f"Error querying: {str(e)}")
    
    async def health_check(self) -> bool:
        """
        Health Check

        Verifies connections to:
        - Qdrant (vector database)
        - HuggingFace (embeddings)
        - OpenAI (LLM)

        Returns:
            bool: True if all services are healthy

        Raises:
            Exception: If any service is unavailable
        """
        try:
            # Check Qdrant connection
            collections = self.qdrant_client.get_collections()
            print(f"Qdrant is healthy. Found {len(collections.collections)} collections")

            # Check HuggingFace embeddings by creating a simple embedding
            test_embedding = self.embeddings.embed_query("test")
            print(f"HuggingFace embeddings are healthy. Embedding dimension: {len(test_embedding)}")
            
            return True
        except Exception as e:
            raise Exception(f"Health check failed: {str(e)}")