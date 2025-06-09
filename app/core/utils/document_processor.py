from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import shutil
import logging
import hashlib
from datetime import datetime
import json

from langchain_community.document_loaders import (
    TextLoader,
    PDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """处理上传的文档，支持文本、PDF、Word和Markdown格式"""
    
    def __init__(self, temp_dir: Optional[str] = None, cache_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "lumi_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.supported_extensions = {
            '.txt': TextLoader,
            '.pdf': PDFLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader
        }
        
        # 根据文件类型设置不同的分块大小
        self.text_splitters = {
            '.txt': RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            ),
            '.pdf': RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                length_function=len,
            ),
            '.docx': RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=250,
                length_function=len,
            ),
            '.md': RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                length_function=len,
            )
        }

    def process_uploaded_file(self, file, kb_type: str) -> Dict[str, Any]:
        """
        处理上传的文件
        
        Args:
            file: 上传的文件对象
            kb_type: 知识库类型 ('mental_health' 或 'communication')
            
        Returns:
            处理结果信息
        """
        try:
            # 计算文件哈希值用于缓存
            file_hash = self._calculate_file_hash(file)
            cache_file = self.cache_dir / f"{file_hash}.json"
            
            # 检查缓存
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_result = json.load(f)
                logger.info(f"使用缓存的处理结果: {file.name}")
                return cached_result
            
            # 创建临时文件
            temp_file = Path(self.temp_dir) / file.name
            with open(temp_file, 'wb') as f:
                f.write(file.getvalue())
            
            # 检查文件类型
            if not self._is_supported_file_type(temp_file):
                return {
                    'success': False,
                    'message': f'不支持的文件类型: {temp_file.suffix}'
                }
            
            # 加载文档
            loader = self._get_loader(temp_file)
            if not loader:
                return {
                    'success': False,
                    'message': '无法加载文档'
                }
            
            # 分割文档
            documents = loader.load()
            chunks = self._split_documents(documents, temp_file.suffix)
            
            # 清理临时文件
            temp_file.unlink()
            
            result = {
                'success': True,
                'chunks': chunks,
                'metadata': {
                    'filename': file.name,
                    'kb_type': kb_type,
                    'upload_time': datetime.now().isoformat(),
                    'chunk_count': len(chunks),
                    'file_hash': file_hash
                }
            }
            
            # 缓存结果
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            return result
            
        except Exception as e:
            logger.error(f"处理文件时出错: {str(e)}")
            return {
                'success': False,
                'message': f'处理文件时出错: {str(e)}'
            }

    def process_batch_files(self, files: List[Any], kb_type: str) -> List[Dict[str, Any]]:
        """
        批量处理多个文件
        
        Args:
            files: 文件对象列表
            kb_type: 知识库类型
            
        Returns:
            处理结果列表
        """
        results = []
        for file in files:
            result = self.process_uploaded_file(file, kb_type)
            results.append(result)
        return results

    def _calculate_file_hash(self, file) -> str:
        """计算文件内容的哈希值"""
        file_content = file.getvalue()
        return hashlib.sha256(file_content).hexdigest()

    def _is_supported_file_type(self, file_path: Path) -> bool:
        """检查文件类型是否支持"""
        return file_path.suffix.lower() in self.supported_extensions

    def _get_loader(self, file_path: Path):
        """获取适合的文档加载器"""
        extension = file_path.suffix.lower()
        loader_class = self.supported_extensions.get(extension)
        if loader_class:
            return loader_class(str(file_path))
        return None

    def _split_documents(self, documents: List[Any], file_extension: str) -> List[Any]:
        """根据文件类型选择合适的分块器"""
        splitter = self.text_splitters.get(file_extension.lower(), self.text_splitters['.txt'])
        return splitter.split_documents(documents)

    def cleanup(self):
        """清理临时目录和缓存"""
        try:
            shutil.rmtree(self.temp_dir)
            # 可选：清理过期的缓存文件
            # self._cleanup_old_cache()
        except Exception as e:
            logger.error(f"清理临时目录时出错: {str(e)}")

    def _cleanup_old_cache(self, max_age_days: int = 7):
        """清理过期的缓存文件"""
        try:
            current_time = datetime.now()
            for cache_file in self.cache_dir.glob("*.json"):
                file_age = current_time - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age.days > max_age_days:
                    cache_file.unlink()
        except Exception as e:
            logger.error(f"清理缓存时出错: {str(e)}") 