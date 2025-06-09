#!/usr/bin/env python
"""
LumiMind 知识库初始化脚本
用于初始化向量数据库和加载示例文档
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.core.knowledge_base import KnowledgeManager
from app.core.utils.document_processor import DocumentProcessor
from app.core.utils.security_handler import SecurityHandler
from app.core.utils.error_handler import ErrorHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_knowledge_base():
    """初始化知识库"""
    try:
        # 初始化处理器
        kb_manager = KnowledgeManager()
        doc_processor = DocumentProcessor()
        security_handler = SecurityHandler()
        error_handler = ErrorHandler()
        
        # 初始化向量数据库
        logger.info("初始化向量数据库...")
        kb_manager.initialize()
        
        # 加载示例文档
        logger.info("加载示例文档...")
        example_docs = {
            'mental_health': [
                'knowledge_base/mental_health_docs/cbt_basics.md',
                'knowledge_base/mental_health_docs/crisis_intervention.md'
            ],
            'communication': [
                'knowledge_base/communication_docs/effective_communication.md'
            ]
        }
        
        for kb_type, docs in example_docs.items():
            logger.info(f"处理 {kb_type} 知识库文档...")
            for doc_path in docs:
                try:
                    # 检查文件是否存在
                    if not Path(doc_path).exists():
                        logger.warning(f"文件不存在: {doc_path}")
                        continue
                    
                    # 处理文档
                    with open(doc_path, 'rb') as f:
                        result = doc_processor.process_uploaded_file(f, kb_type)
                        
                        if result['success']:
                            # 添加到知识库
                            kb_manager.add_document(
                                kb_type,
                                result['content'],
                                result['metadata']
                            )
                            logger.info(f"成功加载文档: {doc_path}")
                        else:
                            logger.error(f"处理文档失败: {doc_path} - {result['message']}")
                            
                except Exception as e:
                    error_result = error_handler.handle_error(e, {
                        'file_path': doc_path,
                        'kb_type': kb_type
                    })
                    logger.error(f"处理文档时出错: {error_result['message']}")
        
        # 显示知识库统计
        stats = kb_manager.get_stats()
        logger.info("知识库统计:")
        logger.info(f"心理健康文档数: {stats['mental_health']['document_count']}")
        logger.info(f"沟通辅导文档数: {stats['communication']['document_count']}")
        
        logger.info("知识库初始化完成！")
        
    except Exception as e:
        logger.error(f"初始化知识库时出错: {str(e)}")
        sys.exit(1)

def main():
    """主函数"""
    logger.info("开始初始化 LumiMind 知识库...")
    initialize_knowledge_base()

if __name__ == "__main__":
    main() 