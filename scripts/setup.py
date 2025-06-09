#!/usr/bin/env python
"""
LumiMind 环境设置脚本
用于自动创建必要的目录结构和配置文件
"""

import os
import sys
import shutil
from pathlib import Path
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """创建必要的目录结构"""
    directories = [
        'app/core/chains/mental_health',
        'app/core/chains/communication',
        'app/core/rag',
        'app/core/utils',
        'app/config',
        'app/modules',
        'knowledge_base/mental_health_docs',
        'knowledge_base/communication_docs',
        'data/vector_db',
        'data/cache',
        'data/temp',
        'logs',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory}")

def setup_environment():
    """设置环境变量和配置文件"""
    # 复制 .env.example 到 .env
    if not Path('.env').exists():
        shutil.copy('.env.example', '.env')
        logger.info("创建 .env 文件")
    
    # 创建必要的空文件
    files = [
        'app/__init__.py',
        'app/core/__init__.py',
        'app/core/chains/__init__.py',
        'app/core/chains/mental_health/__init__.py',
        'app/core/chains/communication/__init__.py',
        'app/core/rag/__init__.py',
        'app/core/utils/__init__.py',
        'app/config/__init__.py',
        'app/modules/__init__.py',
        'tests/__init__.py'
    ]
    
    for file in files:
        Path(file).touch()
        logger.info(f"创建文件: {file}")

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import streamlit
        import langchain
        import chromadb
        import sentence_transformers
        logger.info("核心依赖检查通过")
    except ImportError as e:
        logger.error(f"缺少必要的依赖: {str(e)}")
        logger.info("请运行: pip install -r requirements.txt")
        sys.exit(1)

def main():
    """主函数"""
    logger.info("开始设置 LumiMind 环境...")
    
    # 检查 Python 版本
    if sys.version_info < (3, 8):
        logger.error("需要 Python 3.8 或更高版本")
        sys.exit(1)
    
    # 创建目录结构
    create_directory_structure()
    
    # 设置环境
    setup_environment()
    
    # 检查依赖
    check_dependencies()
    
    logger.info("环境设置完成！")
    logger.info("请确保：")
    logger.info("1. 已编辑 .env 文件并填入必要的 API Keys")
    logger.info("2. 已安装所有依赖: pip install -r requirements.txt")
    logger.info("3. 已初始化知识库: python scripts/initialize_kb.py")
    logger.info("\n现在可以运行: streamlit run app.py")

if __name__ == "__main__":
    main() 