from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config.settings import get_settings

settings = get_settings()

class VectorStoreManager:
    def __init__(
        self,
        collection_name: str,
        embedding_model: str = None,
        persist_directory: str = None
    ):
        """
        Initialize the Vector Store Manager.
        
        Args:
            collection_name: Name of the collection to use
            embedding_model: Model to use for embeddings
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.persist_directory = persist_directory or settings.VECTOR_STORE_PATH
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(texts)
        self.vector_store.persist()
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional filter criteria
            
        Returns:
            List of similar documents
        """
        k = k or settings.MAX_RETRIEVAL_DOCS
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def delete_collection(self) -> None:
        """Delete the current collection."""
        self.vector_store.delete_collection()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        return {
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "persist_directory": self.persist_directory
        } 