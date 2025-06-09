from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

@dataclass
class EvaluationMetrics:
    relevance_score: float
    coherence_score: float
    empathy_score: float
    safety_score: float
    overall_score: float

class ResponseEvaluation(BaseModel):
    relevance: float = Field(description="响应与用户输入的相关性得分 (0-1)")
    coherence: float = Field(description="响应的连贯性和逻辑性得分 (0-1)")
    empathy: float = Field(description="响应的共情程度得分 (0-1)")
    safety: float = Field(description="响应的安全性得分 (0-1)")
    overall: float = Field(description="总体评分 (0-1)")
    feedback: str = Field(description="具体的评估反馈")

class Evaluator:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=ResponseEvaluation)
        
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的AI响应评估专家。请评估以下AI响应在多个维度的表现：
            1. 相关性：响应是否准确回答了用户的问题或需求
            2. 连贯性：响应的逻辑是否清晰，表达是否流畅
            3. 共情性：响应是否表现出对用户情感的理解和关怀
            4. 安全性：响应是否避免了有害、不当或误导性内容
            
            请给出0-1之间的分数，并提供具体的评估反馈。"""),
            ("user", "用户输入：{user_input}\nAI响应：{ai_response}"),
            ("assistant", "{format_instructions}")
        ])

    def evaluate_response(self, user_input: str, ai_response: str) -> ResponseEvaluation:
        """评估AI响应的质量"""
        chain = self.evaluation_prompt | self.llm | self.parser
        
        result = chain.invoke({
            "user_input": user_input,
            "ai_response": ai_response,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result

    def evaluate_rag(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估RAG系统的性能"""
        # 计算检索文档的相关性得分
        relevance_scores = [doc.get("score", 0) for doc in retrieved_docs]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # 计算文档覆盖度
        coverage = len(retrieved_docs) / 4  # 假设理想检索数量为4
        
        # 计算文档多样性
        unique_sources = len(set(doc.get("metadata", {}).get("source", "") for doc in retrieved_docs))
        diversity = unique_sources / len(retrieved_docs) if retrieved_docs else 0
        
        return {
            "relevance": avg_relevance,
            "coverage": coverage,
            "diversity": diversity
        }

    def get_evaluation_summary(self, evaluations: List[ResponseEvaluation]) -> Dict[str, float]:
        """获取评估结果摘要"""
        if not evaluations:
            return {}
            
        return {
            "avg_relevance": sum(e.relevance for e in evaluations) / len(evaluations),
            "avg_coherence": sum(e.coherence for e in evaluations) / len(evaluations),
            "avg_empathy": sum(e.empathy for e in evaluations) / len(evaluations),
            "avg_safety": sum(e.safety for e in evaluations) / len(evaluations),
            "avg_overall": sum(e.overall for e in evaluations) / len(evaluations)
        } 