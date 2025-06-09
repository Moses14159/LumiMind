"""
LLM Factory module for managing different LLM providers.
"""
from typing import Optional, Dict, Any, Union
from langchain_core.language_models.llms import LLM
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

# For providers that might not be fully implemented yet
from langchain_core.language_models.chat_models import BaseChatModel

from app.config.settings import get_settings

settings = get_settings()


class LLMNotConfiguredError(ValueError):
    """Exception raised when LLM is not properly configured."""
    pass


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> Union[BaseChatModel, BaseLLM]:
    """
    Factory function to get an LLM instance based on the provider.
    
    Args:
        provider: The LLM provider to use. Defaults to settings.DEFAULT_LLM_PROVIDER.
        model_name: Specific model name for the provider.
        temperature: Controls randomness in the output. Higher values make output more random.
        max_tokens: Maximum number of tokens to generate.
    
    Returns:
        An instance of the LLM.
    
    Raises:
        ValueError: If the provider is not supported or API key is missing.
    """
    selected_provider = provider or settings.DEFAULT_LLM_PROVIDER
    
    if selected_provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set in environment variables.")
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=model_name or "gpt-3.5-turbo",
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    elif selected_provider == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ValueError("Gemini API key is not set in environment variables.")
        return ChatGoogleGenerativeAI(
            google_api_key=settings.GEMINI_API_KEY,
            model=model_name or "gemini-pro",
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    
    elif selected_provider == "deepseek":
        if not settings.DEEPSEEK_API_KEY:
            raise LLMNotConfiguredError("DeepSeek API key is not configured")
        
        # Placeholder for DeepSeek implementation
        raise NotImplementedError("DeepSeek integration is not implemented yet")
    
    elif selected_provider == "siliconflow":
        if not settings.SILICONFLOW_API_KEY:
            raise LLMNotConfiguredError("SiliconFlow API key is not configured")
        
        # Placeholder for SiliconFlow implementation
        raise NotImplementedError("SiliconFlow integration is not implemented yet")
    
    elif selected_provider == "internlm":
        if not settings.INTERNLM_API_KEY:
            raise LLMNotConfiguredError("InternLM API key is not configured")
        
        # Placeholder for InternLM implementation
        raise NotImplementedError("InternLM integration is not implemented yet")
    
    elif selected_provider == "spark":
        if not all([
            settings.IFLYTEK_SPARK_APPID,
            settings.IFLYTEK_SPARK_API_KEY,
            settings.IFLYTEK_SPARK_API_SECRET
        ]):
            raise LLMNotConfiguredError("iFlyTek Spark API credentials are not configured")
        
        # Placeholder for iFlyTek Spark implementation
        raise NotImplementedError("iFlyTek Spark integration is not implemented yet")
    
    elif selected_provider == "ollama":
        if not settings.OLLAMA_BASE_URL:
            raise ValueError("Ollama base URL is not set.")
        return Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=model_name or settings.OLLAMA_DEFAULT_MODEL,
            temperature=temperature
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {selected_provider}")


def get_default_llm() -> Union[BaseChatModel, BaseLLM]:
    """Get the default LLM instance based on settings."""
    return get_llm() 