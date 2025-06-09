"""
VectorStore Manager for document storage and retrieval.
"""
import os
import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    PDFLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorstoreManager:
    """
    Manager for creating, loading, and managing vector stores.
    """
    
    def __init__(self, persist_directory: str = "data/vector_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def initialize(self):
        """初始化向量存储"""
        self.vector_store = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
        )
        return self

    def load_documents(self, directory: str, file_types: List[str] = None):
        """加载文档到向量存储"""
        if file_types is None:
            file_types = [".txt", ".pdf", ".docx"]

        # 创建文档加载器
        loaders = {
            ".txt": TextLoader,
            ".pdf": PDFLoader,
            ".docx": Docx2txtLoader,
        }

        # 加载文档
        documents = []
        for file_type in file_types:
            if file_type in loaders:
                loader = DirectoryLoader(
                    directory,
                    glob=f"**/*{file_type}",
                    loader_cls=loaders[file_type]
                )
                documents.extend(loader.load())

        # 分割文档
        texts = self.text_splitter.split_documents(documents)

        # 添加到向量存储
        if self.vector_store is None:
            self.initialize()
        
        self.vector_store.add_documents(texts)
        self.vector_store.persist()
        return len(texts)

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        if self.vector_store is None:
            self.initialize()
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]

    def clear(self):
        """清除向量存储"""
        if self.vector_store is not None:
            self.vector_store.delete_collection()
            self.vector_store = None

    def get_stats(self) -> Dict[str, Any]:
        """获取向量存储统计信息"""
        if self.vector_store is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "collection_count": self.vector_store._collection.count(),
            "persist_directory": str(self.persist_directory)
        }

    def get_vectorstore_path(self, collection_name: str) -> str:
        """
        Get the path to a specific vector store collection.
        
        Args:
            collection_name: The name of the collection.
            
        Returns:
            The path to the collection.
        """
        return os.path.join(self.persist_directory, collection_name)
    
    def create_or_load_vectorstore(
        self,
        collection_name: str,
        documents_path: Optional[str] = None,
        force_reload: bool = False
    ) -> VectorStore:
        """
        Create or load a vector store.
        
        Args:
            collection_name: The name of the collection.
            documents_path: The path to the documents to load.
            force_reload: Whether to force a reload of the documents.
            
        Returns:
            The vector store.
        """
        collection_path = self.get_vectorstore_path(collection_name)
        
        # Check if the vector store already exists
        exists = os.path.exists(collection_path) and os.listdir(collection_path)
        
        # If it exists and we don't need to reload, just load it
        if exists and not force_reload:
            logger.info(f"Loading existing vectorstore from {collection_path}")
            return self._load_vectorstore(collection_name)
        
        # If it doesn't exist or we need to reload, create it
        if not documents_path:
            raise ValueError("documents_path must be provided when creating a new vectorstore")
        
        logger.info(f"Creating new vectorstore from {documents_path}")
        return self._create_vectorstore(collection_name, documents_path)
    
    def _load_vectorstore(self, collection_name: str) -> VectorStore:
        """
        Load a vector store from disk.
        
        Args:
            collection_name: The name of the collection.
            
        Returns:
            The loaded vector store.
        """
        collection_path = self.get_vectorstore_path(collection_name)
        
        return Chroma(
            persist_directory=collection_path,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
    
    def _create_vectorstore(self, collection_name: str, documents_path: str) -> VectorStore:
        """
        Create a new vector store from documents.
        
        Args:
            collection_name: The name of the collection.
            documents_path: The path to the documents to load.
            
        Returns:
            The created vector store.
        """
        # Load documents
        documents = self._load_documents(documents_path)
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        collection_path = self.get_vectorstore_path(collection_name)
        
        # Create the vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=collection_path,
            collection_name=collection_name
        )
        vectorstore.persist()
        return vectorstore
    
    def _load_documents(self, documents_path: str) -> List:
        """
        Load documents from a directory.
        
        Args:
            documents_path: The path to the documents to load.
            
        Returns:
            A list of loaded documents.
        """
        loaders = []
        
        # Text files
        loaders.append(
            DirectoryLoader(
                documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
        )
        
        # PDF files
        loaders.append(
            DirectoryLoader(
                documents_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
        )
        
        # CSV files
        loaders.append(
            DirectoryLoader(
                documents_path,
                glob="**/*.csv",
                loader_cls=CSVLoader
            )
        )
        
        # Markdown files
        loaders.append(
            DirectoryLoader(
                documents_path,
                glob="**/*.md",
                loader_cls=UnstructuredMarkdownLoader
            )
        )
        
        documents = []
        for loader in loaders:
            try:
                documents.extend(loader.load())
            except Exception as e:
                logger.warning(f"Error loading documents with {loader.__class__.__name__}: {e}")
        
        return documents
    
    def get_retriever(
        self,
        collection_name: str,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        """
        Get a retriever for a specific collection.
        
        Args:
            collection_name: The name of the collection.
            search_type: The type of search to perform.
            search_kwargs: Additional arguments for the search.
            
        Returns:
            A retriever.
        """
        vectorstore = self._load_vectorstore(collection_name)
        search_kwargs = search_kwargs or {"k": 5}
        
        return vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    
    def create_mental_health_kb(self, force_reload: bool = False) -> VectorStore:
        """
        Create or load the mental health knowledge base.
        
        Args:
            force_reload: Whether to force a reload of the documents.
            
        Returns:
            The mental health knowledge base vector store.
        """
        return self.create_or_load_vectorstore(
            collection_name=settings.MENTAL_HEALTH_KB_NAME,
            documents_path=settings.MENTAL_HEALTH_DOCS_PATH,
            force_reload=force_reload
        )
    
    def create_communication_kb(self, force_reload: bool = False) -> VectorStore:
        """
        Create or load the communication knowledge base.
        
        Args:
            force_reload: Whether to force a reload of the documents.
            
        Returns:
            The communication knowledge base vector store.
        """
        return self.create_or_load_vectorstore(
            collection_name=settings.COMMUNICATION_KB_NAME,
            documents_path=settings.COMMUNICATION_DOCS_PATH,
            force_reload=force_reload
        ) 