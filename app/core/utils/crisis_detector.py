from typing import Dict, List, Tuple
import json
from pathlib import Path
from langchain_core.language_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from app.config.settings import get_settings
from app.core.utils.llm_factory import get_llm

settings = get_settings()

class CrisisDetector:
    def __init__(
        self,
        provider: str = None,
        model_name: str = None,
        temperature: float = 0.1  # 使用较低的温度以获得更确定的判断
    ):
        """
        Initialize the Crisis Detector.
        
        Args:
            provider: LLM provider to use
            model_name: Specific model name
            temperature: Controls randomness in output
        """
        self.llm = get_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature
        )
        
        # 加载危机关键词
        self.crisis_keywords = self._load_crisis_keywords()
        
        # 创建危机检测提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a crisis detection system. Your task is to analyze user input for potential crisis signals.
            
            Consider the following aspects:
            1. Explicit mentions of self-harm or suicide
            2. Strong expressions of hopelessness or despair
            3. References to specific plans or methods
            4. Sudden changes in emotional state
            5. Isolation or withdrawal signals
            
            Respond with a JSON object containing:
            {
                "risk_level": "high" | "medium" | "low",
                "confidence": float between 0 and 1,
                "reasoning": "brief explanation",
                "keywords_found": ["list", "of", "found", "keywords"]
            }
            
            Be conservative in your assessment. When in doubt, classify as higher risk."""),
            ("human", "{input}")
        ])
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True
        )
    
    def _load_crisis_keywords(self) -> Dict[str, List[str]]:
        """Load crisis keywords from file."""
        keywords_path = Path(settings.CRISIS_KEYWORDS_PATH)
        if not keywords_path.exists():
            # 如果文件不存在，使用默认关键词
            return {
                "high_risk": [
                    "自杀", "自残", "结束生命", "不想活了",
                    "suicide", "self-harm", "end my life"
                ],
                "medium_risk": [
                    "绝望", "痛苦", "活不下去", "没有希望",
                    "hopeless", "despair", "can't go on"
                ],
                "low_risk": [
                    "难过", "抑郁", "焦虑", "压力",
                    "sad", "depressed", "anxious", "stress"
                ]
            }
        
        with open(keywords_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def detect_crisis(self, text: str) -> Dict:
        """
        Detect potential crisis signals in the input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing risk assessment
        """
        try:
            # 使用 LLM 进行分析
            response = self.chain.predict(input=text)
            assessment = json.loads(response)
            
            # 验证响应格式
            required_fields = ["risk_level", "confidence", "reasoning", "keywords_found"]
            if not all(field in assessment for field in required_fields):
                raise ValueError("Invalid response format from LLM")
            
            # 确保风险等级是有效的
            if assessment["risk_level"] not in ["high", "medium", "low"]:
                assessment["risk_level"] = "medium"  # 保守处理
            
            # 确保置信度在有效范围内
            assessment["confidence"] = max(0.0, min(1.0, float(assessment["confidence"])))
            
            return assessment
            
        except Exception as e:
            # 发生错误时返回保守的评估
            return {
                "risk_level": "medium",
                "confidence": 0.5,
                "reasoning": f"Error in crisis detection: {str(e)}",
                "keywords_found": []
            }
    
    def get_crisis_response(self, risk_level: str) -> str:
        """
        Get appropriate crisis response based on risk level.
        
        Args:
            risk_level: Detected risk level
            
        Returns:
            Appropriate response message
        """
        responses = {
            "high": """我注意到您可能正在经历一些非常困难的时刻。请记住，您并不孤单，专业帮助随时可用。

紧急求助热线：
- 全国心理援助热线：400-161-9995
- 北京心理危机研究与干预中心：010-82951332
- 上海心理援助热线：021-63798990

建议您：
1. 立即联系专业心理咨询师或精神科医生
2. 与信任的亲友分享您的感受
3. 如果情况紧急，请拨打急救电话 120 或前往最近的医院急诊室""",
            
            "medium": """我理解您现在可能感到非常困扰。请记住，寻求帮助是勇敢的表现。

您可以：
1. 与信任的亲友倾诉
2. 预约专业心理咨询师
3. 拨打心理援助热线：400-161-9995

如果您愿意，我们可以继续聊聊您的感受。""",
            
            "low": """感谢您的分享。如果您感到困扰，随时可以：
1. 与信任的人倾诉
2. 寻求专业心理咨询
3. 使用放松技巧，如深呼吸或冥想

需要我为您提供更多支持吗？"""
        }
        
        return responses.get(risk_level, responses["medium"]) 