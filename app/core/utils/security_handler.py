from typing import Dict, Any, List
import re
import logging
from pathlib import Path
import magic
import hashlib

logger = logging.getLogger(__name__)

class SecurityHandler:
    """处理文件上传和内容的安全检查"""
    
    def __init__(self):
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.sensitive_patterns = [
            r'\b\d{17}[\dXx]\b',  # 身份证号
            r'\b\d{11}\b',        # 手机号
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 邮箱
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # 银行卡号
        ]
        self.compiled_patterns = [re.compile(pattern) for pattern in self.sensitive_patterns]
        
    def check_file(self, file) -> Dict[str, Any]:
        """
        检查上传文件的安全性
        
        Args:
            file: 上传的文件对象
            
        Returns:
            检查结果
        """
        try:
            # 检查文件大小
            file_size = len(file.getvalue())
            if file_size > self.max_file_size:
                return {
                    'safe': False,
                    'message': f'文件大小超过限制 ({file_size/1024/1024:.1f}MB > 10MB)'
                }
            
            # 检查文件类型
            mime = magic.Magic(mime=True)
            file_type = mime.from_buffer(file.getvalue())
            if not self._is_safe_file_type(file_type):
                return {
                    'safe': False,
                    'message': f'不支持的文件类型: {file_type}'
                }
            
            # 检查敏感信息
            content = file.getvalue().decode('utf-8', errors='ignore')
            sensitive_info = self._check_sensitive_info(content)
            if sensitive_info:
                return {
                    'safe': False,
                    'message': f'文件包含敏感信息: {", ".join(sensitive_info)}'
                }
            
            return {
                'safe': True,
                'message': '文件安全检查通过'
            }
            
        except Exception as e:
            logger.error(f"安全检查时出错: {str(e)}")
            return {
                'safe': False,
                'message': f'安全检查时出错: {str(e)}'
            }
    
    def _is_safe_file_type(self, mime_type: str) -> bool:
        """检查文件类型是否安全"""
        safe_types = {
            'text/plain',
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/markdown'
        }
        return mime_type in safe_types
    
    def _check_sensitive_info(self, content: str) -> List[str]:
        """检查内容中的敏感信息"""
        found_info = []
        for pattern in self.compiled_patterns:
            matches = pattern.findall(content)
            if matches:
                found_info.append(f"发现 {len(matches)} 处敏感信息")
        return found_info
    
    def sanitize_content(self, content: str) -> str:
        """清理内容中的敏感信息"""
        for pattern in self.compiled_patterns:
            content = pattern.sub('[已脱敏]', content)
        return content 