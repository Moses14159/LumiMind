from typing import Dict, List, Any, Optional
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.core.utils.llm_factory import get_llm
from app.config.settings import get_settings

settings = get_settings()

class ResponseOption(BaseModel):
    """Structure for a response option."""
    text: str = Field(..., description="The response text")
    tone: str = Field(..., description="The tone of the response")
    potential_impact: str = Field(..., description="Potential impact on the conversation")
    reasoning: str = Field(..., description="Reasoning behind this response")

class CommunicationScenario(BaseModel):
    """Structure for a communication scenario analysis."""
    context: str = Field(..., description="The communication context")
    goal: str = Field(..., description="The communication goal")
    current_state: str = Field(..., description="Current state of the conversation")
    response_options: List[ResponseOption] = Field(..., description="Possible response options")
    recommended_approach: str = Field(..., description="Recommended approach")
    follow_up_questions: List[str] = Field(..., description="Questions to clarify the situation")

class ResponseCoachChain:
    def __init__(
        self,
        provider: str = None,
        model_name: str = None,
        temperature: float = 0.7,
        max_tokens: int = None
    ):
        """
        Initialize the Response Coach Chain.
        
        Args:
            provider: LLM provider to use
            model_name: Specific model name
            temperature: Controls randomness in output
            max_tokens: Maximum tokens to generate
        """
        self.llm = get_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.memory = ConversationBufferWindowMemory(
            k=5,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # 创建输出解析器
        self.output_parser = PydanticOutputParser(pydantic_object=CommunicationScenario)
        
        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a communication coach helping users navigate difficult conversations and respond effectively.

Your task is to:
1. Analyze the communication context
2. Understand the user's goals
3. Generate multiple response options
4. Explain the potential impact of each option
5. Provide reasoning and recommendations
6. Suggest follow-up questions for clarification

Respond in a structured format that can be parsed into a CommunicationScenario object:
{
    "context": "description of the communication context",
    "goal": "the user's communication goal",
    "current_state": "current state of the conversation",
    "response_options": [
        {
            "text": "the response text",
            "tone": "the tone of the response",
            "potential_impact": "potential impact on the conversation",
            "reasoning": "reasoning behind this response"
        }
    ],
    "recommended_approach": "overall recommended approach",
    "follow_up_questions": ["list of questions to clarify the situation"]
}

Be practical and specific. Focus on actionable advice and concrete examples."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and generate communication coaching response.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dictionary containing the communication scenario analysis
        """
        try:
            response = self.chain.predict(input=user_input)
            # 解析响应为 CommunicationScenario 对象
            scenario = self.output_parser.parse(response)
            return scenario.dict()
        except Exception as e:
            return {
                "error": f"Error processing communication scenario: {str(e)}",
                "context": user_input,
                "goal": "",
                "current_state": "",
                "response_options": [],
                "recommended_approach": "",
                "follow_up_questions": []
            }
    
    def get_guidance(self, scenario_type: str) -> str:
        """
        Get guidance for different communication scenarios.
        
        Args:
            scenario_type: Type of communication scenario
            
        Returns:
            Guidance for the scenario
        """
        guidance = {
            "conflict": """在处理冲突时，建议：
1. 保持冷静，避免情绪化反应
2. 使用"我"语句表达感受
3. 积极倾听对方观点
4. 寻找共同点
5. 提出具体解决方案""",
            
            "feedback": """在给予反馈时，建议：
1. 具体描述观察到的行为
2. 说明行为的影响
3. 提出改进建议
4. 使用建设性语言
5. 保持开放对话""",
            
            "request": """在提出请求时，建议：
1. 清晰说明需求
2. 解释原因
3. 提供具体细节
4. 表达对对方时间的尊重
5. 保持灵活性""",
            
            "apology": """在道歉时，建议：
1. 真诚承认错误
2. 具体说明错误
3. 表达理解和共情
4. 提出弥补方案
5. 承诺改进"""
        }
        
        return guidance.get(scenario_type, "让我们分析一下这个沟通情境，找出最合适的回应方式。")
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Get the current chat history.
        
        Returns:
            List of chat messages
        """
        return self.memory.chat_memory.messages 