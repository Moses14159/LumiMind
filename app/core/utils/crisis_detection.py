"""
Crisis detection utility for identifying potential mental health crises in user inputs.
"""
import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings

logger = logging.getLogger(__name__)


class CrisisDetector:
    """
    Detects potential mental health crises in user inputs.
    
    Uses a combination of keyword matching and LLM-based detection to identify
    expressions of self-harm, suicidal ideation, or other crisis situations.
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        keywords_path: Optional[str] = None,
        threshold: Optional[float] = None
    ):
        """
        Initialize the crisis detector.
        
        Args:
            llm: The LLM to use for advanced detection.
            keywords_path: Path to the file containing crisis keywords.
            threshold: The threshold for considering a message as indicating a crisis.
        """
        self.llm = llm
        self.keywords_path = keywords_path or settings.CRISIS_KEYWORDS_PATH
        self.threshold = threshold or settings.CRISIS_DETECTION_THRESHOLD
        self.crisis_keywords = self._load_crisis_keywords()
        
        # Define the crisis detection prompt
        self.crisis_detection_prompt = PromptTemplate.from_template(
            """You are an AI assistant trained to detect signs of mental health crises in text. 
Your task is to analyze the following message and determine if it indicates a potential crisis 
such as suicidal ideation, self-harm intentions, or severe psychological distress.

User message: "{message}"

Based on this message, is the user expressing thoughts or intentions of self-harm, 
suicide, or showing signs of a severe mental health crisis that requires immediate attention?

Respond with ONLY "YES" if you detect a potential crisis, or "NO" if you do not. 
Do not include any other text in your response."""
        )
    
    def _load_crisis_keywords(self) -> List[str]:
        """
        Load crisis keywords from a file.
        
        Returns:
            A list of crisis keywords.
        """
        if not os.path.exists(self.keywords_path):
            logger.warning(f"Crisis keywords file not found at {self.keywords_path}. Using default keywords.")
            return [
                "suicide", "kill myself", "end my life", "take my own life",
                "self-harm", "cut myself", "hurt myself", "self harm",
                "can't go on", "don't want to live", "better off dead",
                "no reason to live", "want to die"
            ]
        
        with open(self.keywords_path, "r", encoding="utf-8") as f:
            return [line.strip().lower() for line in f if line.strip()]
    
    def _keyword_detection(self, message: str) -> Tuple[bool, List[str]]:
        """
        Detect crisis keywords in a message.
        
        Args:
            message: The message to analyze.
            
        Returns:
            A tuple containing:
                - Whether any keywords were detected.
                - A list of detected keywords.
        """
        message_lower = message.lower()
        detected_keywords = []
        
        for keyword in self.crisis_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', message_lower):
                detected_keywords.append(keyword)
        
        return bool(detected_keywords), detected_keywords
    
    async def _llm_detection(self, message: str) -> bool:
        """
        Use an LLM to detect potential crises.
        
        Args:
            message: The message to analyze.
            
        Returns:
            Whether the LLM detected a potential crisis.
        """
        if not self.llm:
            logger.warning("No LLM provided for crisis detection. Skipping LLM detection.")
            return False
        
        chain = self.crisis_detection_prompt | self.llm | StrOutputParser()
        
        try:
            result = await chain.ainvoke({"message": message})
            return result.strip().upper() == "YES"
        except Exception as e:
            logger.error(f"Error in LLM crisis detection: {e}")
            return False
    
    async def detect_crisis(self, message: str) -> Dict[str, Union[bool, List[str], float]]:
        """
        Detect potential crises in a message.
        
        Args:
            message: The message to analyze.
            
        Returns:
            A dictionary containing:
                - `is_crisis`: Whether a crisis was detected.
                - `detected_keywords`: A list of detected crisis keywords.
                - `llm_detection`: Whether the LLM detected a crisis (if available).
                - `confidence`: A confidence score (0-1) for the crisis detection.
        """
        # Keyword detection
        keyword_detected, detected_keywords = self._keyword_detection(message)
        
        # LLM detection (if available)
        llm_detected = False
        if self.llm and (keyword_detected or len(message.split()) > 10):
            llm_detected = await self._llm_detection(message)
        
        # Calculate confidence score
        confidence = 0.0
        if keyword_detected:
            # Base confidence on number of keywords detected
            confidence = min(0.7, 0.3 + (len(detected_keywords) * 0.1))
        
        if llm_detected:
            # Increase confidence if LLM also detected crisis
            confidence = max(confidence, 0.8)
        
        # Determine final crisis status
        is_crisis = confidence >= self.threshold
        
        return {
            "is_crisis": is_crisis,
            "detected_keywords": detected_keywords,
            "llm_detection": llm_detected,
            "confidence": confidence
        }
    
    def detect_crisis_sync(self, message: str) -> Dict[str, Union[bool, List[str], float]]:
        """
        Synchronous version of crisis detection (without LLM).
        
        Args:
            message: The message to analyze.
            
        Returns:
            A dictionary containing:
                - `is_crisis`: Whether a crisis was detected.
                - `detected_keywords`: A list of detected crisis keywords.
                - `confidence`: A confidence score (0-1) for the crisis detection.
        """
        # Keyword detection
        keyword_detected, detected_keywords = self._keyword_detection(message)
        
        # Calculate confidence score based on keywords only
        confidence = 0.0
        if keyword_detected:
            # Base confidence on number of keywords detected
            confidence = min(0.7, 0.3 + (len(detected_keywords) * 0.1))
        
        # Determine final crisis status
        is_crisis = confidence >= self.threshold
        
        return {
            "is_crisis": is_crisis,
            "detected_keywords": detected_keywords,
            "confidence": confidence
        } 