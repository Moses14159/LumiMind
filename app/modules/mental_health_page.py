"""
Mental Health Support module for the LumiMind application.
"""
import streamlit as st
from langchain_core.language_models import BaseChatModel
import os

from core.chains.mental_health_chain import EmpatheticConversationChain
from core.chains.cbt_exercise_chain import CBTExerciseChain
from core.utils.crisis_detection import CrisisDetector


def initialize_state():
    """
    Initialize session state variables for the mental health page.
    """
    if "mental_health_messages" not in st.session_state:
        st.session_state.mental_health_messages = []
    
    if "empathetic_chain" not in st.session_state:
        st.session_state.empathetic_chain = None
    
    if "cbt_exercise_chain" not in st.session_state:
        st.session_state.cbt_exercise_chain = None
    
    if "crisis_detector" not in st.session_state:
        st.session_state.crisis_detector = None
    
    if "mode" not in st.session_state:
        st.session_state.mode = "chat"  # "chat" or "cbt"


def display_messages():
    """
    Display the conversation messages.
    """
    for message in st.session_state.mental_health_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def display_crisis_alert():
    """
    Display a crisis alert message.
    """
    st.error("""
    **重要提示：请寻求专业帮助**
    
    检测到你可能正处于严重的心理困扰。AI 助手无法替代专业的心理健康支持。
    
    建议：
    * 联系心理健康专业人士
    * 向信任的朋友或家人寻求帮助
    * 拨打危机热线：
        * 全国生命危机干预热线：988 或 1-800-273-8255
        * 危机短信服务：发送 HOME 至 741741
    你的健康很重要，专业的支持可以带来积极改变。
    """)


def handle_file_upload():
    st.sidebar.subheader("上传知识文档")
    uploaded_file = st.sidebar.file_uploader(
        "上传 txt/pdf/csv/md 文件以扩展心理健康知识库",
        type=["txt", "pdf", "csv", "md"]
    )
    if uploaded_file is not None:
        save_dir = "app/knowledge_base/mental_health_docs"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"文件 {uploaded_file.name} 已成功上传！")
        st.sidebar.info("请点击下方按钮刷新知识库以使新内容生效。")
        if st.sidebar.button("刷新知识库"):
            from core.rag.vectorstore_manager import VectorstoreManager
            manager = VectorstoreManager()
            manager.create_mental_health_kb(force_reload=True)
            st.sidebar.success("知识库已刷新！")


def render_mental_health_page(llm: BaseChatModel):
    """
    Render the mental health support page.
    
    Args:
        llm: The language model to use.
    """
    # Initialize state
    initialize_state()
    
    # Instantiate chains if needed
    if st.session_state.empathetic_chain is None:
        st.session_state.empathetic_chain = EmpatheticConversationChain(llm=llm)
    
    if st.session_state.cbt_exercise_chain is None:
        st.session_state.cbt_exercise_chain = CBTExerciseChain(llm=llm)
    
    if st.session_state.crisis_detector is None:
        st.session_state.crisis_detector = CrisisDetector(llm=llm)
    
    # Page header
    st.title("💭 心理健康支持")
    
    # Mode selection
    mode = st.radio(
        "选择模式",
        ["共情对话", "CBT 练习"],
        horizontal=True,
        index=0 if st.session_state.mode == "chat" else 1
    )
    
    # Update mode
    st.session_state.mode = "chat" if mode == "共情对话" else "cbt"
    
    # Display description
    if st.session_state.mode == "chat":
        st.write("""
        这里是一个安全的空间，你可以表达你的想法和感受。
        AI 助手会以共情和支持的方式倾听你。
        
        *注意：本系统不能替代专业心理健康服务。*
        """)
    else:
        st.write("""
        认知行为疗法（CBT）可以帮助你识别和挑战消极思维。
        本练习将引导你逐步完成一个具体情境的自我探索。
        
        *注意：本系统不能替代专业心理健康服务。*
        """)
    
    # Display conversation
    display_messages()
    
    # Input for user message
    user_input = st.chat_input("请输入你的消息...")
    
    if user_input:
        # Add user message to conversation
        st.session_state.mental_health_messages.append({"role": "user", "content": user_input})
        
        # Display updated conversation
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Crisis detection (synchronous for simplicity in this demo)
        crisis_result = st.session_state.crisis_detector.detect_crisis_sync(user_input)
        
        if crisis_result["is_crisis"]:
            # Display crisis alert
            display_crisis_alert()
            
            # Add assistant message to conversation
            response = "I notice you may be experiencing significant distress. This AI assistant is not equipped to provide crisis support. Please consider reaching out to a mental health professional or crisis helpline. Your wellbeing is important."
            st.session_state.mental_health_messages.append({"role": "assistant", "content": response})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            # Process normal response based on mode
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    if st.session_state.mode == "chat":
                        # Use empathetic conversation chain
                        response = st.session_state.empathetic_chain.invoke({"input": user_input})
                    else:
                        # Use CBT exercise chain
                        result = st.session_state.cbt_exercise_chain.invoke({"input": user_input})
                        response = result["response"]
                        
                        # Display current stage if in CBT mode
                        if "stage" in result:
                            st.caption(f"当前阶段：{result['stage']}")
                    
                    # Display assistant response
                    st.markdown(response)
            
            # Add assistant message to conversation
            st.session_state.mental_health_messages.append({"role": "assistant", "content": response})
    
    # Reset buttons
    with st.sidebar:
        if st.button("清空对话", key="clear_mental_health"):
            # Reset conversation
            st.session_state.mental_health_messages = []
            
            # Reset chains
            if st.session_state.mode == "chat":
                st.session_state.empathetic_chain.clear_memory()
            else:
                st.session_state.cbt_exercise_chain.reset()
            
            st.rerun()
    
    handle_file_upload() 