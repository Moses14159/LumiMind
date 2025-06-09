import streamlit as st
import os
from pathlib import Path
import sys
import logging
from datetime import datetime
from app.core.knowledge_base import KnowledgeManager
from app.core.utils.document_processor import DocumentProcessor
from app.core.utils.security_handler import SecurityHandler
from app.core.utils.error_handler import ErrorHandler

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from app.core.utils.crisis_detector import CrisisDetector
from app.core.chains.mental_health.cbt_chain import CBTExerciseChain
from app.core.chains.communication.response_coach import ResponseCoachChain
from app.core.utils.emotion_analyzer import EmotionAnalyzer
from app.config.settings import get_settings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化设置
settings = get_settings()

# 设置页面配置
st.set_page_config(
    page_title="LumiMind - AI驱动的心理健康与沟通辅导平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载自定义CSS
def load_css():
    css_file = Path("static/style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 初始化会话状态
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'current_module' not in st.session_state:
    st.session_state.current_module = "心理健康咨询"

# 加载CSS
load_css()

# 初始化处理器
@st.cache_resource
def init_handlers():
    return {
        'kb_manager': KnowledgeManager(),
        'doc_processor': DocumentProcessor(),
        'security_handler': SecurityHandler(),
        'error_handler': ErrorHandler()
    }

handlers = init_handlers()

# 侧边栏
with st.sidebar:
    st.image("static/logo.png", width=200)
    st.title("LumiMind")
    
    # 模块选择
    st.subheader("选择模块")
    module = st.radio(
        "选择功能模块",
        ["心理健康咨询", "沟通辅导"],
        label_visibility="collapsed"
    )
    st.session_state.current_module = module
    
    # AI模型选择
    st.subheader("AI模型")
    model = st.selectbox(
        "选择语言模型",
        ["GPT-4", "Gemini Pro", "DeepSeek", "Ollama"],
        label_visibility="collapsed"
    )
    
    # 知识库管理
    st.subheader("知识库管理")
    uploaded_files = st.file_uploader(
        "上传文档",
        type=["txt", "pdf", "docx", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            with st.spinner(f"处理文件: {file.name}"):
                try:
                    # 处理文件
                    result = handlers['doc_processor'].process_uploaded_file(
                        file,
                        st.session_state.current_module
                    )
                    
                    if result['success']:
                        # 添加到知识库
                        handlers['kb_manager'].add_document(
                            st.session_state.current_module,
                            result['content'],
                            result['metadata']
                        )
                        st.success(f"成功处理文件: {file.name}")
                    else:
                        st.error(f"处理文件失败: {result['message']}")
                        
                except Exception as e:
                    error_result = handlers['error_handler'].handle_error(e)
                    st.error(f"处理文件时出错: {error_result['message']}")

# 主界面
st.title(f"LumiMind - {st.session_state.current_module}")

# 聊天界面
chat_container = st.container()

# 显示历史消息
for message in st.session_state.messages:
    with chat_container:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

# 用户输入
user_input = st.text_input("输入您的问题或想法...", key="user_input")

if user_input:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 显示用户消息
    with chat_container:
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-content">{user_input}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 处理用户输入
    with st.spinner("思考中..."):
        try:
            # 根据当前模块选择不同的处理逻辑
            if st.session_state.current_module == "心理健康咨询":
                # 心理健康咨询逻辑
                response = "这是一个心理健康咨询的回复示例。"
            else:
                # 沟通辅导逻辑
                response = "这是一个沟通辅导的回复示例。"
            
            # 添加助手消息
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # 显示助手消息
            with chat_container:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-content">{response}</div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            error_result = handlers['error_handler'].handle_error(e)
            st.error(f"处理请求时出错: {error_result['message']}")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>LumiMind - AI驱动的心理健康与沟通辅导平台</p>
        <p>© 2024 LumiMind. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
) 