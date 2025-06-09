from typing import Optional
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    SILICONFLOW_API_KEY: Optional[str] = None
    INTERNLM_API_KEY: Optional[str] = None
    IFLYTEK_SPARK_APPID: Optional[str] = None
    IFLYTEK_SPARK_API_KEY: Optional[str] = None
    IFLYTEK_SPARK_API_SECRET: Optional[str] = None
    
    # LLM Configuration
    DEFAULT_LLM_PROVIDER: str = "openai"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL: str = "llama2"
    
    # Vector Database
    VECTOR_DB_PATH: Path = Path("data/vector_db")
    
    # Crisis Detection
    CRISIS_KEYWORDS_PATH: Path = Path("data/crisis_keywords.json")
    
    # Knowledge Base
    MENTAL_HEALTH_KB_PATH: Path = Path("knowledge_base/mental_health_docs")
    COMMUNICATION_KB_PATH: Path = Path("knowledge_base/communication_docs")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_settings() -> Settings:
    """Get application settings."""
    return Settings() 