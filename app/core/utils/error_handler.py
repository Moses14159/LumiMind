from typing import Dict, Any, Optional
import logging
import traceback
from datetime import datetime
import json
from pathlib import Path
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ErrorHandler:
    """统一的错误处理类"""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.error_log_file = self.log_dir / "error.log"
        
        # 配置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志记录"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(self.error_log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理错误并返回用户友好的错误信息
        
        Args:
            error: 异常对象
            context: 错误发生的上下文信息
            
        Returns:
            错误处理结果
        """
        error_id = self._generate_error_id()
        error_time = datetime.now().isoformat()
        
        # 记录错误详情
        error_details = {
            'error_id': error_id,
            'timestamp': error_time,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        # 记录到日志
        logger.error(
            f"Error ID: {error_id}\n"
            f"Type: {error_details['error_type']}\n"
            f"Message: {error_details['error_message']}\n"
            f"Context: {json.dumps(context, ensure_ascii=False) if context else 'None'}\n"
            f"Traceback:\n{error_details['traceback']}"
        )
        
        # 保存错误详情到文件
        self._save_error_details(error_details)
        
        # 返回用户友好的错误信息
        return self._get_user_friendly_error(error, error_id)
    
    def _generate_error_id(self) -> str:
        """生成唯一的错误ID"""
        return f"ERR-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{hash(str(datetime.now()))}"
    
    def _save_error_details(self, error_details: Dict[str, Any]):
        """保存错误详情到文件"""
        error_file = self.log_dir / f"error_{error_details['error_id']}.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_details, f, ensure_ascii=False, indent=2)
    
    def _get_user_friendly_error(self, error: Exception, error_id: str) -> Dict[str, Any]:
        """生成用户友好的错误信息"""
        error_type = type(error).__name__
        
        # 根据错误类型返回不同的用户友好信息
        if isinstance(error, ValueError):
            return {
                'success': False,
                'error_id': error_id,
                'message': str(error),
                'suggestion': '请检查输入是否正确'
            }
        elif isinstance(error, FileNotFoundError):
            return {
                'success': False,
                'error_id': error_id,
                'message': '找不到指定的文件',
                'suggestion': '请确认文件路径是否正确'
            }
        elif isinstance(error, PermissionError):
            return {
                'success': False,
                'error_id': error_id,
                'message': '没有足够的权限执行操作',
                'suggestion': '请检查文件权限或联系管理员'
            }
        else:
            return {
                'success': False,
                'error_id': error_id,
                'message': '发生未知错误',
                'suggestion': '请稍后重试或联系管理员'
            }
    
    def get_error_details(self, error_id: str) -> Optional[Dict[str, Any]]:
        """获取指定错误ID的详细信息"""
        error_file = self.log_dir / f"error_{error_id}.json"
        if error_file.exists():
            with open(error_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def handle_llm_error(self, error: Exception) -> Dict[str, Any]:
        """处理LLM相关错误"""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "suggestion": "请检查API密钥和网络连接，或尝试使用其他模型。"
        }
        
        logger.error(f"LLM Error: {error_info}")
        return error_info

    def handle_rag_error(self, error: Exception) -> Dict[str, Any]:
        """处理RAG相关错误"""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "suggestion": "请检查知识库文件是否存在，或尝试重新初始化向量存储。"
        }
        
        logger.error(f"RAG Error: {error_info}")
        return error_info

    def handle_general_error(self, error: Exception) -> Dict[str, Any]:
        """处理一般错误"""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "suggestion": "请稍后重试，或联系技术支持。"
        }
        
        logger.error(f"General Error: {error_info}")
        return error_info

    async def analyze_error(self, error: Exception) -> Dict[str, Any]:
        """使用LLM分析错误（如果可用）"""
        if not self.llm:
            return self.handle_general_error(error)
            
        try:
            chain = self.error_prompt | self.llm
            analysis = await chain.ainvoke({
                "error_message": str(error)
            })
            
            return {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "llm_analysis": analysis
            }
        except Exception as e:
            logger.error(f"Error analysis failed: {str(e)}")
            return self.handle_general_error(error)

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """记录错误信息"""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        logger.error(f"Error occurred: {error_info}")
        return error_info 