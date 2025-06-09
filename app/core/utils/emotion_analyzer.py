from typing import Dict, List, Any, Optional
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.core.utils.llm_factory import get_llm
from app.config.settings import get_settings

settings = get_settings()

class Emotion(BaseModel):
    """Structure for an emotion analysis."""
    primary_emotion: str = Field(..., description="The primary emotion detected")
    intensity: float = Field(..., description="Intensity of the emotion (0-1)")
    secondary_emotions: List[str] = Field(..., description="Secondary emotions detected")
    triggers: List[str] = Field(..., description="Potential triggers for these emotions")
    coping_suggestions: List[str] = Field(..., description="Suggested coping strategies")

class EmotionAnalyzer:
    def __init__(
        self,
        provider: str = None,
        model_name: str = None,
        temperature: float = 0.7,
        max_tokens: int = None
    ):
        """
        Initialize the Emotion Analyzer.
        
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
        
        # 创建输出解析器
        self.output_parser = PydanticOutputParser(pydantic_object=Emotion)
        
        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an emotion analysis expert helping to identify and understand emotional states.

Your task is to:
1. Identify primary and secondary emotions
2. Assess emotional intensity
3. Identify potential triggers
4. Suggest appropriate coping strategies

Respond in a structured format that can be parsed into an Emotion object:
{
    "primary_emotion": "the main emotion detected",
    "intensity": "intensity level (0-1)",
    "secondary_emotions": ["list of secondary emotions"],
    "triggers": ["list of potential triggers"],
    "coping_suggestions": ["list of coping strategies"]
}

Be empathetic and supportive in your analysis."""),
            ("human", "{input}")
        ])
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True
        )
    
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotions in the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing the emotion analysis
        """
        try:
            response = self.chain.predict(input=text)
            # 解析响应为 Emotion 对象
            emotion = self.output_parser.parse(response)
            return emotion.dict()
        except Exception as e:
            return {
                "error": f"Error analyzing emotions: {str(e)}",
                "primary_emotion": "unknown",
                "intensity": 0.0,
                "secondary_emotions": [],
                "triggers": [],
                "coping_suggestions": []
            }
    
    def get_emotion_guidance(self, emotion: str) -> str:
        """
        Get guidance for handling specific emotions.
        
        Args:
            emotion: The emotion to get guidance for
            
        Returns:
            Guidance for handling the emotion
        """
        guidance = {
            "anger": """处理愤怒情绪的建议：
1. 深呼吸，数到10
2. 暂时离开触发情境
3. 进行身体活动释放能量
4. 使用"我"语句表达感受
5. 寻求支持或专业帮助""",
            
            "anxiety": """处理焦虑情绪的建议：
1. 练习深呼吸和冥想
2. 进行渐进式肌肉放松
3. 写下担忧并分析
4. 保持规律作息
5. 寻求专业支持""",
            
            "sadness": """处理悲伤情绪的建议：
1. 允许自己感受情绪
2. 与信任的人分享
3. 保持基本生活规律
4. 进行适度运动
5. 寻求专业帮助""",
            
            "stress": """处理压力情绪的建议：
1. 识别压力源
2. 制定优先级清单
3. 练习时间管理
4. 保持健康生活方式
5. 学习放松技巧"""
        }
        
        return guidance.get(emotion.lower(), "让我们先理解这种情绪，然后一起找到合适的应对方式。")
    
    def get_emotion_intensity_scale(self) -> Dict[str, List[str]]:
        """
        Get the emotion intensity scale.
        
        Returns:
            Dictionary containing intensity levels and descriptions
        """
        return {
            "low": [
                "轻微的情绪波动",
                "可以正常应对",
                "不影响日常生活"
            ],
            "medium": [
                "明显的情绪反应",
                "需要一些调节",
                "可能影响部分活动"
            ],
            "high": [
                "强烈的情绪体验",
                "需要积极调节",
                "显著影响日常生活"
            ],
            "severe": [
                "极端的情绪状态",
                "需要专业帮助",
                "严重影响功能"
            ]
        } 