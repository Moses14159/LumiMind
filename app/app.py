"""
LumiMind: LLM-driven Mental Health Consultation and Communication Coaching Platform

Main Streamlit application for the LumiMind platform.
"""
import os
import logging
import streamlit as st
from langchain_core.language_models import BaseChatModel

# Import modules
from modules.mental_health_page import render_mental_health_page
from modules.communication_page import render_communication_page

# Import utility functions
from core.utils.llm_factory import get_llm, get_default_llm

# Import settings
from config.settings import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set page config
st.set_page_config(
    page_title="LumiMind",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_available_llm_providers():
    """
    Get a list of available LLM providers based on configured API keys.
    
    Returns:
        A list of available LLM providers.
    """
    available_providers = []
    
    # Check for API keys
    if settings.OPENAI_API_KEY:
        available_providers.append("openai")
    
    if settings.GEMINI_API_KEY:
        available_providers.append("gemini")
    
    if settings.DEEPSEEK_API_KEY:
        available_providers.append("deepseek")
    
    if settings.SILICONFLOW_API_KEY:
        available_providers.append("siliconflow")
    
    if settings.INTERNLM_API_KEY:
        available_providers.append("internlm")
    
    if all([
        settings.IFLYTEK_SPARK_APPID,
        settings.IFLYTEK_SPARK_API_KEY,
        settings.IFLYTEK_SPARK_API_SECRET
    ]):
        available_providers.append("spark")
    
    # Ollama is always available (assuming it's running locally)
    available_providers.append("ollama")
    
    return available_providers


def initialize_session_state():
    """
    Initialize the session state variables.
    """
    # LLM provider selection
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = settings.DEFAULT_LLM_PROVIDER
    
    # LLM instance
    if "llm" not in st.session_state:
        st.session_state.llm = get_default_llm()
    
    # Current module
    if "current_module" not in st.session_state:
        st.session_state.current_module = "mental_health"


def render_sidebar():
    """
    Render the sidebar with navigation and settings.
    """
    st.sidebar.title("LumiMind 智能助手")
    st.sidebar.caption("大模型驱动的心理健康与沟通支持平台")
    
    # Module selection
    st.sidebar.subheader("导航")
    module = st.sidebar.radio(
        "选择模块",
        ["心理健康支持", "沟通辅导"],
        index=0 if st.session_state.current_module == "mental_health" else 1,
        key="module_selection"
    )
    
    # Update current module based on selection
    if module == "心理健康支持":
        st.session_state.current_module = "mental_health"
    else:
        st.session_state.current_module = "communication"
    
    # LLM provider selection
    st.sidebar.subheader("设置")
    # 支持的所有 provider
    provider_display = {
        "openai": "OpenAI",
        "gemini": "Google Gemini",
        "ollama": "本地 Ollama",
        "deepseek": "DeepSeek（敬请期待）",
        "siliconflow": "SiliconFlow（敬请期待）",
        "internlm": "InternLM（敬请期待）",
        "spark": "讯飞星火（敬请期待）"
    }
    implemented_providers = ["openai", "gemini", "ollama"]
    available_providers = get_available_llm_providers()
    # 保证所有 provider 都能显示
    all_providers = [p for p in provider_display.keys() if p in available_providers or p in implemented_providers]
    provider_options = [provider_display.get(p, p) for p in all_providers]
    # 当前 provider 的 index
    current_index = all_providers.index(st.session_state.llm_provider) if st.session_state.llm_provider in all_providers else 0
    provider = st.sidebar.selectbox(
        "选择大模型服务商",
        provider_options,
        index=current_index
    )
    
    # 反向映射
    provider_map = {v: k for k, v in provider_display.items()}
    selected_provider = provider_map.get(provider, provider)
    if selected_provider != st.session_state.llm_provider:
        if selected_provider not in implemented_providers:
            st.sidebar.warning(f"{provider} 暂未开放，敬请期待！")
        else:
            st.session_state.llm_provider = selected_provider
            try:
                st.session_state.llm = get_llm(provider=selected_provider)
                # 强制重建所有 chain
                st.session_state.empathetic_chain = None
                st.session_state.cbt_exercise_chain = None
                st.session_state.response_coach_chain = None
                st.session_state.role_play_chain = None
                st.sidebar.success(f"已切换到 {provider}！")
            except Exception as e:
                st.sidebar.error(f"切换到 {provider} 时出错: {str(e)}")
                st.session_state.llm = get_default_llm()
                st.session_state.llm_provider = settings.DEFAULT_LLM_PROVIDER
    
    # About section
    st.sidebar.subheader("关于")
    st.sidebar.info(
        """
        LumiMind 是一个基于大模型的心理健康与沟通辅导平台。
        
        **本系统不能替代专业心理健康服务。**
        
        如遇危机，请及时联系专业机构或拨打紧急求助热线。
        """
    )
    
    # Version info
    st.sidebar.caption(f"版本 {settings.APP_VERSION}")


def main():
    """
    Main function to run the Streamlit app.
    """
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render current module
    if st.session_state.current_module == "mental_health":
        render_mental_health_page(st.session_state.llm)
    else:
        render_communication_page(st.session_state.llm)


if __name__ == "__main__":
    main() 