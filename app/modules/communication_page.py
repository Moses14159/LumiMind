"""
Communication Coach module for the LumiMind application.
"""
import streamlit as st
from langchain_core.language_models import BaseChatModel

from core.chains.communication_coach_chain import ResponseCoachChain
from core.chains.role_play_chain import RolePlayChain, STANDARD_SCENARIOS
from core.utils.crisis_detection import CrisisDetector


def initialize_state():
    """
    Initialize session state variables for the communication page.
    """
    if "communication_messages" not in st.session_state:
        st.session_state.communication_messages = []
    
    if "response_coach_chain" not in st.session_state:
        st.session_state.response_coach_chain = None
    
    if "role_play_chain" not in st.session_state:
        st.session_state.role_play_chain = None
    
    if "crisis_detector" not in st.session_state:
        st.session_state.crisis_detector = None
    
    if "comm_mode" not in st.session_state:
        st.session_state.comm_mode = "response_coach"  # "response_coach" or "role_play"
    
    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = "salary_negotiation"
    
    if "in_role_play" not in st.session_state:
        st.session_state.in_role_play = False


def display_messages():
    """
    Display the conversation messages.
    """
    for message in st.session_state.communication_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def display_crisis_alert():
    """
    Display a crisis alert message when crisis is detected in communication module.
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


def render_communication_page(llm: BaseChatModel):
    """
    Render the communication coach page.
    
    Args:
        llm: The language model to use.
    """
    # Initialize state
    initialize_state()
    
    # Instantiate chains if needed
    if st.session_state.response_coach_chain is None:
        st.session_state.response_coach_chain = ResponseCoachChain(llm=llm)
    
    if st.session_state.role_play_chain is None:
        st.session_state.role_play_chain = RolePlayChain(llm=llm)
    
    if st.session_state.crisis_detector is None:
        st.session_state.crisis_detector = CrisisDetector(llm=llm)
    
    # Page header
    st.title("💬 沟通辅导")
    
    # Mode selection
    mode = st.radio(
        "选择模式",
        ["回应教练", "情景演练"],
        horizontal=True,
        index=0 if st.session_state.comm_mode == "response_coach" else 1
    )
    
    # Update mode
    if mode == "回应教练":
        # Only reset if we're changing modes
        if st.session_state.comm_mode != "response_coach":
            st.session_state.communication_messages = []
            st.session_state.response_coach_chain.clear_memory()
            st.session_state.in_role_play = False
        st.session_state.comm_mode = "response_coach"
    else:
        # Only reset if we're changing modes
        if st.session_state.comm_mode != "role_play":
            st.session_state.communication_messages = []
            st.session_state.in_role_play = False
        st.session_state.comm_mode = "role_play"
    
    # Display description based on mode
    if st.session_state.comm_mode == "response_coach":
        st.write("""
        请描述你在沟通中遇到的困惑或需要帮助的情境，AI 会分析并给出多种回应建议及解释。
        """)
    else:
        st.write("""
        在这里你可以选择一个情景，与 AI 进行角色扮演练习。你可以随时输入"我表现得怎么样？"来获得反馈，或输入"结束演练"来结束本次对话。
        """)
        
        # Display scenario selection for role play mode
        if not st.session_state.in_role_play:
            # Let user select a scenario
            scenario_names = list(STANDARD_SCENARIOS.keys())
            scenario_display_names = [s.name for s in STANDARD_SCENARIOS.values()]
            
            scenario_index = scenario_names.index(st.session_state.selected_scenario) if st.session_state.selected_scenario in scenario_names else 0
            
            selected_display_name = st.selectbox(
                "选择一个练习情景",
                scenario_display_names,
                index=scenario_index
            )
            
            # Map display name back to scenario key
            for key, scenario in STANDARD_SCENARIOS.items():
                if scenario.name == selected_display_name:
                    st.session_state.selected_scenario = key
                    break
            
            # Display scenario description
            scenario = STANDARD_SCENARIOS[st.session_state.selected_scenario]
            st.info(f"""
            **情景**：{scenario.name}
            
            **你的角色**：自己
            
            **AI 的角色**：{scenario.character}
            
            **情景描述**：{scenario.scenario_description}
            """)
            
            # Button to start role play
            if st.button("开始演练"):
                # Set the scenario for the role play chain
                st.session_state.role_play_chain.set_scenario(scenario)
                
                # Add a system message to start
                system_msg = f"情景演练已开始。你正在练习 {scenario.name}，AI 扮演 {scenario.character}。"
                st.session_state.communication_messages.append({"role": "system", "content": system_msg})
                
                # Set in_role_play to true
                st.session_state.in_role_play = True
                
                # Rerun to refresh the UI
                st.rerun()
    
    # Display conversation
    display_messages()
    
    # Input for user message
    user_input = st.chat_input(
        "请输入你的沟通困惑或回复..." if st.session_state.comm_mode == "response_coach" else 
        "请输入你的回复，继续对话..."
    )
    
    if user_input:
        # Check for crisis
        crisis_result = st.session_state.crisis_detector.detect_crisis_sync(user_input)
        
        if crisis_result["is_crisis"]:
            # Display crisis alert
            display_crisis_alert()
            
            # Add user message to conversation
            st.session_state.communication_messages.append({"role": "user", "content": user_input})
            
            # Display updated conversation
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Add assistant message about crisis
            response = "检测到你可能正处于严重的心理困扰。AI 助手无法提供危机干预，请及时联系专业人士或危机热线。"
            st.session_state.communication_messages.append({"role": "assistant", "content": response})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            # Add user message to conversation
            st.session_state.communication_messages.append({"role": "user", "content": user_input})
            
            # Display updated conversation
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Process response based on mode
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    if st.session_state.comm_mode == "response_coach":
                        # Use response coach chain
                        response = st.session_state.response_coach_chain.invoke({"input": user_input})
                    else:
                        # Use role play chain
                        result = st.session_state.role_play_chain.invoke({"input": user_input})
                        response = result["response"]
                        
                        # If not in character, this is feedback - style it differently
                        if not result.get("in_character", True):
                            st.info(response)
                        else:
                            st.markdown(response)
                        
                        # For visual clarity, only add markdown if we didn't use st.info
                        if result.get("in_character", True):
                            # Display assistant response
                            st.markdown(response)
            
            # Add assistant message to conversation
            st.session_state.communication_messages.append({"role": "assistant", "content": response})
    
    # Reset buttons
    with st.sidebar:
        if st.session_state.comm_mode == "role_play" and st.session_state.in_role_play:
            if st.button("结束演练"):
                # Generate ending message
                ending_response = st.session_state.role_play_chain._end_role_play()
                
                # Add to conversation
                st.session_state.communication_messages.append({"role": "system", "content": "情景演练已结束。"})
                st.session_state.communication_messages.append({"role": "assistant", "content": ending_response})
                
                # Reset role play state
                st.session_state.in_role_play = False
                
                # Rerun to refresh UI
                st.rerun()
        
        if st.button("清空对话", key="clear_communication"):
            # Reset conversation
            st.session_state.communication_messages = []
            
            # Reset chains
            if st.session_state.comm_mode == "response_coach":
                st.session_state.response_coach_chain.clear_memory()
            elif st.session_state.in_role_play:
                st.session_state.role_play_chain.clear_memory()
                st.session_state.in_role_play = False
            
            st.rerun() 