from typing import Dict, List, Optional
from pathlib import Path
from .vectorstore_manager import VectorStoreManager
from app.config.settings import get_settings

settings = get_settings()

class KnowledgeManager:
    def __init__(self):
        self.mental_health_kb = VectorStoreManager(
            persist_directory=str(settings.VECTOR_DB_PATH / "mental_health")
        )
        self.communication_kb = VectorStoreManager(
            persist_directory=str(settings.VECTOR_DB_PATH / "communication")
        )
        self.initialized = False

    def initialize(self):
        """初始化知识库"""
        if not self.initialized:
            self.mental_health_kb.initialize()
            self.communication_kb.initialize()
            self.initialized = True
        return self

    def load_knowledge_base(self):
        """加载知识库文档"""
        # 加载心理健康知识库
        mental_health_docs = self.mental_health_kb.load_documents(
            str(settings.MENTAL_HEALTH_KB_PATH)
        )
        
        # 加载沟通辅导知识库
        communication_docs = self.communication_kb.load_documents(
            str(settings.COMMUNICATION_KB_PATH)
        )
        
        return {
            "mental_health_docs": mental_health_docs,
            "communication_docs": communication_docs
        }

    def search_mental_health(self, query: str, k: int = 4) -> List[Dict]:
        """搜索心理健康知识库"""
        return self.mental_health_kb.search(query, k=k)

    def search_communication(self, query: str, k: int = 4) -> List[Dict]:
        """搜索沟通辅导知识库"""
        return self.communication_kb.search(query, k=k)

    def get_kb_stats(self) -> Dict:
        """获取知识库统计信息"""
        return {
            "mental_health": self.mental_health_kb.get_stats(),
            "communication": self.communication_kb.get_stats()
        }

    def clear_knowledge_base(self):
        """清除知识库"""
        self.mental_health_kb.clear()
        self.communication_kb.clear()
        self.initialized = False 