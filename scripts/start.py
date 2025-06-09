#!/usr/bin/env python
"""
LumiMind 启动脚本
用于启动应用程序并检查环境配置
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """检查环境配置"""
    try:
        # 检查 Python 版本
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            logger.error("需要 Python 3.8 或更高版本")
            return False
        
        # 检查环境变量
        required_env_vars = [
            'OPENAI_API_KEY',
            'GOOGLE_API_KEY',
            'DEEPSEEK_API_KEY',
            'SILICONFLOW_API_KEY',
            'INTERNLM_API_KEY'
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"缺少必要的环境变量: {', '.join(missing_vars)}")
            logger.info("请检查 .env 文件配置")
            return False
        
        # 检查依赖包
        try:
            import streamlit
            import langchain
            import chromadb
            import sentence_transformers
        except ImportError as e:
            logger.error(f"缺少必要的依赖包: {str(e)}")
            logger.info("请运行: pip install -r requirements.txt")
            return False
        
        # 检查目录结构
        required_dirs = [
            'app/core/chains/mental_health',
            'app/core/chains/communication',
            'app/config',
            'data/cache',
            'data/vector_db',
            'knowledge_base/mental_health_docs',
            'knowledge_base/communication_docs'
        ]
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            if not full_path.exists():
                logger.warning(f"目录不存在: {dir_path}")
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"已创建目录: {dir_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"环境检查失败: {str(e)}")
        return False

def start_application():
    """启动应用程序"""
    try:
        # 检查环境
        if not check_environment():
            logger.error("环境检查未通过，请解决上述问题后重试")
            sys.exit(1)
        
        # 启动 Streamlit 应用
        logger.info("正在启动 LumiMind...")
        app_path = project_root / "app.py"
        
        # 使用 subprocess 启动应用
        process = subprocess.Popen(
            ["streamlit", "run", str(app_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 等待应用启动
        while True:
            output = process.stdout.readline()
            if output:
                print(output.strip())
            if "You can now view your Streamlit app in your browser" in output:
                break
        
        logger.info("LumiMind 已启动")
        
        # 保持进程运行
        process.wait()
        
    except KeyboardInterrupt:
        logger.info("正在关闭 LumiMind...")
        process.terminate()
        logger.info("LumiMind 已关闭")
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        sys.exit(1)

def main():
    """主函数"""
    logger.info("开始启动 LumiMind...")
    start_application()

if __name__ == "__main__":
    main() 