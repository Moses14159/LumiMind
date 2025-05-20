"""
LLM Factory module for managing different LLM providers.
"""
from typing import Optional, Dict, Any, Union
from langchain_core.language_models.llms import LLM
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama

# For providers that might not be fully implemented yet
from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import settings


class LLMNotConfiguredError(ValueError):
    """Exception raised when LLM is not properly configured."""
    pass


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any
) -> Union[BaseChatModel, LLM]:
    """
    Get a LLM instance based on the specified provider.
    
    Args:
        provider: The LLM provider to use. If None, uses the default provider from settings.
        model_name: The model name to use. If None, uses the default model for the provider.
        **kwargs: Additional arguments to pass to the LLM constructor.
        
    Returns:
        A LangChain LLM instance.
        
    Raises:
        ValueError: If the API key or configuration is missing.
        NotImplementedError: If the provider is not implemented.
    """
    if provider is None:
        provider = settings.DEFAULT_LLM_PROVIDER
        
    provider = provider.lower()
    
    # OpenAI
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise LLMNotConfiguredError("OpenAI API key is not configured")
        
        model = model_name or settings.OPENAI_DEFAULT_MODEL
        return ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=model,
            **kwargs
        )
    
    # Google (Gemini)
    elif provider == "gemini":
        if not settings.GEMINI_API_KEY:
            raise LLMNotConfiguredError("Gemini API key is not configured")
        
        model = model_name or settings.GEMINI_DEFAULT_MODEL
        return ChatGoogleGenerativeAI(
            google_api_key=settings.GEMINI_API_KEY,
            model=model,
            **kwargs
        )
    
    # DeepSeek
    elif provider == "deepseek":
        if not settings.DEEPSEEK_API_KEY:
            raise LLMNotConfiguredError("DeepSeek API key is not configured")
        
        # Placeholder for DeepSeek implementation
        raise NotImplementedError("DeepSeek integration is not implemented yet")
    
    # SiliconFlow
    elif provider == "siliconflow":
        if not settings.SILICONFLOW_API_KEY:
            raise LLMNotConfiguredError("SiliconFlow API key is not configured")
        
        # Placeholder for SiliconFlow implementation
        raise NotImplementedError("SiliconFlow integration is not implemented yet")
    
    # InternLM
    elif provider == "internlm":
        if not settings.INTERNLM_API_KEY:
            raise LLMNotConfiguredError("InternLM API key is not configured")
        
        # Placeholder for InternLM implementation
        raise NotImplementedError("InternLM integration is not implemented yet")
    
    # iFlyTek Spark
    elif provider == "spark":
        if not all([
            settings.IFLYTEK_SPARK_APPID,
            settings.IFLYTEK_SPARK_API_KEY,
            settings.IFLYTEK_SPARK_API_SECRET
        ]):
            raise LLMNotConfiguredError("iFlyTek Spark API credentials are not configured")
        
        # Placeholder for iFlyTek Spark implementation
        raise NotImplementedError("iFlyTek Spark integration is not implemented yet")
    
    # Ollama
    elif provider == "ollama":
        model = model_name or settings.OLLAMA_DEFAULT_MODEL
        return Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=model,
            **kwargs
        )
    
    # Unknown provider
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_default_llm(**kwargs: Any) -> Union[BaseChatModel, LLM]:
    """
    Get the default LLM instance based on the configuration.
    
    Args:
        **kwargs: Additional arguments to pass to the LLM constructor.
        
    Returns:
        A LangChain LLM instance.
    """
    return get_llm(
        provider=settings.DEFAULT_LLM_PROVIDER,
        **kwargs
    ) 