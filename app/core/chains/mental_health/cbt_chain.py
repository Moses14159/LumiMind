from typing import Dict, List, Any, Optional
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.core.utils.llm_factory import get_llm
from app.config.settings import get_settings

settings = get_settings()

class CBTExercise(BaseModel):
    """CBT exercise structure."""
    situation: str = Field(..., description="The triggering situation")
    thoughts: List[str] = Field(..., description="Automatic thoughts")
    emotions: List[str] = Field(..., description="Emotions experienced")
    evidence_for: List[str] = Field(..., description="Evidence supporting the thoughts")
    evidence_against: List[str] = Field(..., description="Evidence against the thoughts")
    alternative_thoughts: List[str] = Field(..., description="More balanced thoughts")
    action_plan: List[str] = Field(..., description="Concrete steps to take")

class CBTExerciseChain:
    def __init__(
        self,
        provider: str = None,
        model_name: str = None,
        temperature: float = 0.7,
        max_tokens: int = None
    ):
        """
        Initialize the CBT Exercise Chain.
        
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
        self.output_parser = PydanticOutputParser(pydantic_object=CBTExercise)
        
        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a CBT (Cognitive Behavioral Therapy) exercise guide. Help users work through their thoughts and feelings using CBT techniques.

Your task is to guide users through a structured CBT exercise, helping them:
1. Identify the triggering situation
2. Recognize automatic thoughts
3. Identify associated emotions
4. Examine evidence for and against their thoughts
5. Develop more balanced thoughts
6. Create an action plan

Respond in a structured format that can be parsed into a CBTExercise object:
{
    "situation": "description of the triggering situation",
    "thoughts": ["list of automatic thoughts"],
    "emotions": ["list of emotions experienced"],
    "evidence_for": ["evidence supporting the thoughts"],
    "evidence_against": ["evidence against the thoughts"],
    "alternative_thoughts": ["more balanced thoughts"],
    "action_plan": ["concrete steps to take"]
}

Be supportive and non-judgmental. Help users explore their thoughts and feelings without pushing them too hard."""),
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
        Process user input and generate a CBT exercise response.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dictionary containing the CBT exercise structure
        """
        try:
            response = self.chain.predict(input=user_input)
            # 解析响应为 CBTExercise 对象
            exercise = self.output_parser.parse(response)
            return exercise.dict()
        except Exception as e:
            return {
                "error": f"Error processing CBT exercise: {str(e)}",
                "situation": user_input,
                "thoughts": [],
                "emotions": [],
                "evidence_for": [],
                "evidence_against": [],
                "alternative_thoughts": [],
                "action_plan": []
            }
    
    def get_next_step(self, current_step: str) -> str:
        """
        Get guidance for the next step in the CBT exercise.
        
        Args:
            current_step: Current step in the exercise
            
        Returns:
            Guidance for the next step
        """
        step_guidance = {
            "situation": "请描述一下触发您情绪的具体情境。发生了什么？在哪里？和谁在一起？",
            "thoughts": "在这个情境中，您脑海中闪过了哪些想法？请尽量具体地描述。",
            "emotions": "这些想法让您感受到了什么情绪？请尝试用具体的词语描述。",
            "evidence": "让我们来检验这些想法。有什么证据支持这些想法？又有什么证据表明这些想法可能不完全准确？",
            "alternatives": "基于我们的讨论，您能想到一些更平衡或更现实的想法吗？",
            "action": "基于这些新的想法，您觉得可以采取哪些具体的行动来改善这个情况？"
        }
        
        return step_guidance.get(current_step, "让我们继续探索您的想法和感受。")
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Get the current chat history.
        
        Returns:
            List of chat messages
        """
        return self.memory.chat_memory.messages 