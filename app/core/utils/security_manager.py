from typing import List, Dict, Any, Optional
import re
import json
from pathlib import Path
from dataclasses import dataclass
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

@dataclass
class SecurityCheck:
    is_safe: bool
    risk_level: str  # "low", "medium", "high"
    detected_issues: List[str]
    sanitized_text: str

class SecurityManager:
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        self.llm = llm
        self.sensitive_patterns = self._load_sensitive_patterns()
        
        self.security_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的内容安全分析专家。请分析以下文本，检查是否包含：
            1. 敏感个人信息（如身份证号、手机号、邮箱等）
            2. 有害或不当内容
            3. 潜在的误导性信息
            4. 其他安全风险
            
            请给出详细的分析结果。"""),
            ("user", "文本内容：{text}"),
        ])

    def _load_sensitive_patterns(self) -> Dict[str, List[str]]:
        """加载敏感信息模式"""
        return {
            "personal_info": [
                r'\d{17}[\dXx]',  # 身份证号
                r'1[3-9]\d{9}',   # 手机号
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # 邮箱
            ],
            "sensitive_words": [
                # 添加敏感词列表
            ]
        }

    def sanitize_input(self, text: str) -> str:
        """清理用户输入"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?，。！？]', '', text)
        
        # 替换敏感信息
        for pattern in self.sensitive_patterns["personal_info"]:
            text = re.sub(pattern, '[已屏蔽]', text)
        
        return text.strip()

    def detect_sensitive_info(self, text: str) -> SecurityCheck:
        """检测敏感信息"""
        detected_issues = []
        risk_level = "low"
        
        # 检查敏感信息模式
        for category, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    detected_issues.append(f"检测到{category}类型的敏感信息")
                    risk_level = "high" if category == "personal_info" else "medium"
        
        # 清理文本
        sanitized_text = self.sanitize_input(text)
        
        return SecurityCheck(
            is_safe=len(detected_issues) == 0,
            risk_level=risk_level,
            detected_issues=detected_issues,
            sanitized_text=sanitized_text
        )

    async def analyze_security(self, text: str) -> Dict[str, Any]:
        """使用LLM分析内容安全性（如果可用）"""
        if not self.llm:
            return self.detect_sensitive_info(text)
            
        try:
            chain = self.security_prompt | self.llm
            analysis = await chain.ainvoke({"text": text})
            
            return {
                "llm_analysis": analysis,
                "pattern_check": self.detect_sensitive_info(text)
            }
        except Exception as e:
            return self.detect_sensitive_info(text)

    def validate_json(self, json_str: str) -> bool:
        """验证JSON字符串的安全性"""
        try:
            data = json.loads(json_str)
            # 检查JSON结构
            if not isinstance(data, (dict, list)):
                return False
            return True
        except json.JSONDecodeError:
            return False

    def check_file_safety(self, file_path: Path) -> bool:
        """检查文件安全性"""
        # 检查文件扩展名
        allowed_extensions = {'.txt', '.pdf', '.docx', '.md'}
        if file_path.suffix.lower() not in allowed_extensions:
            return False
            
        # 检查文件大小（限制为10MB）
        if file_path.stat().st_size > 10 * 1024 * 1024:
            return False
            
        return True 