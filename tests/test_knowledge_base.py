import unittest
from pathlib import Path
from app.core.rag.knowledge_manager import KnowledgeManager
from app.config.settings import get_settings

settings = get_settings()

class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb_manager = KnowledgeManager()
        self.kb_manager.initialize()

    def test_mental_health_search(self):
        """测试心理健康知识库搜索"""
        query = "什么是CBT？"
        results = self.kb_manager.search_mental_health(query)
        
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        self.assertIn("content", results[0])
        self.assertIn("score", results[0])

    def test_communication_search(self):
        """测试沟通辅导知识库搜索"""
        query = "如何有效沟通？"
        results = self.kb_manager.search_communication(query)
        
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        self.assertIn("content", results[0])
        self.assertIn("score", results[0])

    def test_kb_stats(self):
        """测试知识库统计信息"""
        stats = self.kb_manager.get_kb_stats()
        
        self.assertIn("mental_health", stats)
        self.assertIn("communication", stats)
        self.assertIn("status", stats["mental_health"])
        self.assertIn("status", stats["communication"])

    def test_crisis_detection(self):
        """测试危机检测相关内容检索"""
        query = "自杀风险信号"
        results = self.kb_manager.search_mental_health(query)
        
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        self.assertIn("content", results[0])
        self.assertIn("score", results[0])

if __name__ == '__main__':
    unittest.main() 