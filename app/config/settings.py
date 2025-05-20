from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict, Any


class Settings(BaseSettings):
    """
    Application settings managed with Pydantic. 
    Values are loaded from environment variables and .env file.
    """
    # General settings
    APP_NAME: str = "LumiMind"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # Default LLM provider
    DEFAULT_LLM_PROVIDER: str = "openai"
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_DEFAULT_MODEL: str = "gpt-4-turbo"
    
    # Google (Gemini)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_DEFAULT_MODEL: str = "gemini-pro"
    
    # DeepSeek
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_DEFAULT_MODEL: str = "deepseek-chat"
    
    # SiliconFlow
    SILICONFLOW_API_KEY: Optional[str] = None
    SILICONFLOW_DEFAULT_MODEL: str = "sf-chat"
    
    # InternLM
    INTERNLM_API_KEY: Optional[str] = None
    INTERNLM_DEFAULT_MODEL: str = "internlm-chat-20b"
    
    # iFlyTek Spark
    IFLYTEK_SPARK_APPID: Optional[str] = None
    IFLYTEK_SPARK_API_KEY: Optional[str] = None
    IFLYTEK_SPARK_API_SECRET: Optional[str] = None
    IFLYTEK_SPARK_DEFAULT_MODEL: str = "spark-3.5"
    
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL: str = "llama3"
    
    # Vector database settings
    VECTORDB_TYPE: str = "chroma"  # "chroma" or "faiss"
    VECTORDB_PATH: str = "./vectordb"
    
    # RAG settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Mental health knowledge base
    MENTAL_HEALTH_DOCS_PATH: str = "./app/knowledge_base/mental_health_docs"
    MENTAL_HEALTH_KB_NAME: str = "mental_health_kb"
    
    # Communication knowledge base
    COMMUNICATION_DOCS_PATH: str = "./app/knowledge_base/communication_docs"
    COMMUNICATION_KB_NAME: str = "communication_kb"
    
    # Crisis detection settings
    CRISIS_KEYWORDS_PATH: str = "./app/core/utils/crisis_keywords.txt"
    CRISIS_DETECTION_THRESHOLD: float = 0.7
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Create a singleton instance
settings = Settings() 