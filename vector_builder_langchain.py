#!/usr/bin/env python3
"""
📦 主入口檔案 vector_builder_langchain.py - 修正版 (支援4A+4B結構)

🎯 設計原則：
   - 適應4A+4B拆分方案（vector_operations.py + management_api.py）
   - 完全向後兼容，所有現有程式無需修改
   - 智能回退機制，支援逐步檔案創建
   - 統一的重導出介面

🔒 向後兼容性保證：
   ✅ from vector_builder_langchain import OptimizedVectorSystem
   ✅ from vector_builder_langchain import SmartTextAnalyzer, OptimizedTextSplitter
   ✅ from vector_builder_langchain import AdaptiveBatchProcessor
   ✅ from vector_builder_langchain import SYSTEM_CONFIG, TOKEN_LIMITS
   ✅ from vector_builder_langchain import main

🔄 修正：支援4A+4B結構 vs 原5檔案結構的差異
"""

import os
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set, Tuple

# =============================================================================
# 🔄 智能導入策略 - 支援4A+4B結構 + 5檔案結構 + 完整回退
# =============================================================================

# 🎯 策略1: 嘗試從4A+4B結構導入 (當前使用的結構)
try:
    print("🔄 嘗試4A+4B結構導入...")
    
    # 4A: 向量操作核心層
    from vector_operations import VectorOperationsCore
    
    # 4B: 管理API層 (檢查文件名)
    try:
        from management_api import ManagementAPI, OptimizedVectorSystem
        USING_4AB_STRUCTURE = True
        print("✅ 4A+4B結構導入成功 (management_api.py)")
    except ImportError:
        # 嘗試備選文件名
        from management_interface import OptimizedVectorSystem
        ManagementAPI = OptimizedVectorSystem  # 建立別名
        USING_4AB_STRUCTURE = True
        print("✅ 4A+4B結構導入成功 (management_interface.py)")
    
    # 從4A導入基礎配置和類別
    SYSTEM_CONFIG = getattr(vector_operations, 'SYSTEM_CONFIG', {})
    TOKEN_LIMITS = getattr(vector_operations, 'TOKEN_LIMITS', {})
    SUPPORTED_EXTENSIONS = getattr(vector_operations, 'SUPPORTED_EXTENSIONS', set())
    
    # 從4A導入依賴檢查變數
    CHROMA_AVAILABLE = getattr(vector_operations, 'CHROMA_AVAILABLE', False)
    PGVECTOR_AVAILABLE = getattr(vector_operations, 'PGVECTOR_AVAILABLE', False)
    OPENAI_EMBEDDINGS_AVAILABLE = getattr(vector_operations, 'OPENAI_EMBEDDINGS_AVAILABLE', False)
    PSYCOPG2_AVAILABLE = getattr(vector_operations, 'PSYCOPG2_AVAILABLE', False)
    JIEBA_AVAILABLE = getattr(vector_operations, 'JIEBA_AVAILABLE', False)
    OPENCC_AVAILABLE = getattr(vector_operations, 'OPENCC_AVAILABLE', False)
    DOCX2TXT_AVAILABLE = getattr(vector_operations, 'DOCX2TXT_AVAILABLE', False)
    EPUB_AVAILABLE = getattr(vector_operations, 'EPUB_AVAILABLE', False)
    
    # 從4A導入資料類別
    FileInfo = getattr(vector_operations, 'FileInfo', None)
    TextAnalysis = getattr(vector_operations, 'TextAnalysis', None)
    ChunkInfo = getattr(vector_operations, 'ChunkInfo', None)
    SearchResult = getattr(vector_operations, 'SearchResult', None)
    
    # 從4A導入文本處理類別
    SmartTextAnalyzer = getattr(vector_operations, 'SmartTextAnalyzer', None)
    OptimizedTextSplitter = getattr(vector_operations, 'OptimizedTextSplitter', None)
    ChineseTextNormalizer = getattr(vector_operations, 'ChineseTextNormalizer', None)
    AdaptiveBatchProcessor = getattr(vector_operations, 'AdaptiveBatchProcessor', None)
    
    # 從4A導入工具函數
    detect_railway_environment = getattr(vector_operations, 'detect_railway_environment', None)
    check_openai_api_key = getattr(vector_operations, 'check_openai_api_key', None)
    
    print("✅ 4A+4B結構完全導入成功")

except ImportError as e:
    print(f"⚠️ 4A+4B結構導入失敗: {e}")
    USING_4AB_STRUCTURE = False
    
    # 🎯 策略2: 嘗試從5檔案結構導入
    try:
        print("🔄 嘗試5檔案結構導入...")
        from core_config import *
        from text_processing import *
        from vector_builder import *
        from management_interface import *
        print("✅ 5檔案結構導入成功")
        
    except ImportError:
        print("⚠️ 5檔案結構也不可用，使用回退定義")
        
        # 🎯 策略3: 完整回退定義 - 確保系統可以獨立運行
        SYSTEM_CONFIG = {
            "persist_dir": "./chroma_langchain_db",
            "data_dir": "./data",
            "device": "cpu",
            "log_level": "INFO",
            "max_workers": 4,
            "cache_embeddings": True,
            "backup_enabled": True
        }
        
        SMART_TEXT_CONFIG = {
            "micro_text_threshold": 100,
            "short_text_threshold": 500,
            "medium_text_threshold": 2000,
            "long_text_threshold": 8000,
            "ultra_long_threshold": 20000,
            "micro_text": {
                "min_length": 20,
                "process_whole": True,
                "merge_with_next": True,
                "chunk_size": 0,
                "description": "微文本整體處理"
            },
            "short_text": {
                "min_length": 50,
                "process_whole": True,
                "preserve_structure": True,
                "chunk_size": 0,
                "allow_merge": True,
                "description": "短文本整體保存"
            }
        }
        
        TOKEN_LIMITS = {
            "max_tokens_per_request": 150000,
            "max_batch_size": 8,
            "min_batch_size": 1,
            "batch_delay": 5.0,
            "retry_delay": 10,
            "max_retries": 3,
            "token_safety_margin": 0.2,
            "adaptive_batching": True
        }
        
        PERFORMANCE_CONFIG = {
            "embedding_batch_size": 12,
            "parallel_processing": True,
            "memory_limit_mb": 1024,
            "chunk_cache_size": 1000,
            "preload_models": True,
            "gc_frequency": 50
        }
        
        SUPPORTED_EXTENSIONS = {
            '.txt', '.md', '.pdf', '.csv', '.json', '.py', '.js',
            '.docx', '.doc', '.epub', '.rst', '.org'
        }
        
        # 依賴檢查變數（回退版本）
        CHROMA_AVAILABLE = False
        PGVECTOR_AVAILABLE = False
        OPENAI_EMBEDDINGS_AVAILABLE = False
        PSYCOPG2_AVAILABLE = False
        JIEBA_AVAILABLE = False
        OPENCC_AVAILABLE = False
        DOCX2TXT_AVAILABLE = False
        EPUB_AVAILABLE = False
        
        @dataclass
        class FileInfo:
            path: str
            size: int
            mtime: float
            hash: str
            encoding: str = "utf-8"
            file_type: str = ""
        
        @dataclass
        class TextAnalysis:
            length: int
            text_type: str
            language: str
            encoding: str
            structure_info: Dict
            quality_score: float
            processing_strategy: str
        
        @dataclass
        class ChunkInfo:
            chunk_id: str
            content: str
            metadata: Dict
            token_count: int
            quality_score: float
            relationships: List[str]
        
        @dataclass
        class SearchResult:
            content: str
            score: float
            metadata: Dict
            collection: str
            chunk_info: Optional[ChunkInfo] = None
        
        def detect_railway_environment():
            railway_indicators = [
                os.getenv("RAILWAY_PROJECT_ID"),
                os.getenv("RAILWAY_SERVICE_ID"),
                os.getenv("DATABASE_URL"),
                "railway.internal" in os.getenv("POSTGRES_HOST", "")
            ]
            return any(railway_indicators)
        
        def check_openai_api_key():
            api_key = os.getenv("OPENAI_API_KEY")
            return bool(api_key and api_key.startswith("sk-") and len(api_key) > 20)
        
        # 回退版本的處理類別
        class SmartTextAnalyzer:
            def __init__(self):
                pass
                
            def analyze_text(self, text: str, source_info: Dict = None):
                length = len(text)
                if length < 100:
                    text_type = "micro_text"
                elif length < 500:
                    text_type = "short_text"
                elif length < 2000:
                    text_type = "medium_text"
                elif length < 8000:
                    text_type = "long_text"
                else:
                    text_type = "ultra_long"
                
                return TextAnalysis(
                    length=length,
                    text_type=text_type,
                    language="chinese",
                    encoding="utf-8",
                    structure_info={},
                    quality_score=0.7,
                    processing_strategy="simple_split"
                )
        
        class OptimizedTextSplitter:
            def __init__(self):
                pass
            
            def smart_split_documents(self, text: str, doc_id: str, source_info: Dict = None):
                try:
                    from langchain_core.documents import Document
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
                    )
                    
                    chunks = splitter.split_text(text)
                    documents = []
                    
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            metadata = {
                                'doc_id': doc_id,
                                'chunk_id': f"{doc_id}_{i+1:03d}",
                                'chunk_index': i,
                                'text_type': 'medium_text',
                                'token_count': len(chunk) // 4,  # 簡單估計
                                'source': source_info.get('file_path', '') if source_info else ''
                            }
                            documents.append(Document(page_content=chunk, metadata=metadata))
                    
                    return documents
                except ImportError:
                    return []
        
        class ChineseTextNormalizer:
            def __init__(self):
                pass
            
            def normalize_text(self, text: str):
                return text.strip(), {}
            
            def create_search_variants(self, query: str):
                return [query]
        
        class AdaptiveBatchProcessor:
            def __init__(self):
                self.max_batch_size = TOKEN_LIMITS["max_batch_size"]
            
            def create_smart_batches(self, documents):
                if not documents:
                    return []
                
                batches = []
                batch_size = min(self.max_batch_size, len(documents))
                
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i+batch_size]
                    batch_info = {
                        'documents': len(batch_docs),
                        'tokens': sum(len(doc.page_content) // 4 for doc in batch_docs),
                        'types': {'medium_text': len(batch_docs)}
                    }
                    batches.append((batch_docs, batch_info))
                
                return batches
            
            def record_batch_result(self, success: bool, processing_time: float = 0):
                pass
            
            def get_performance_stats(self):
                return {
                    'total_batches': 0,
                    'success_rate': 1.0,
                    'avg_batch_time': 0
                }
        
        # 回退版本的主要系統類別
        class OptimizedVectorSystem:
            def __init__(self, data_dir: str = None, model_type: str = None):
                raise ImportError(
                    "OptimizedVectorSystem 需要完整的模塊架構。\n"
                    "請確保以下文件存在：\n"
                    "  - 4A+4B結構: vector_operations.py + management_api.py\n"
                    "  - 或5檔案結構: core_config.py + text_processing.py + vector_builder.py + management_interface.py"
                )

# =============================================================================
# 🎯 主程式函數 - 保持完全向後兼容
# =============================================================================

def main():
    """主程式 - 支援4A+4B結構的完整版本"""
    print("🇨🇳 === 完整優化版 LangChain 中文向量系統 === 🇨🇳")
    
    # 顯示當前架構
    if globals().get('USING_4AB_STRUCTURE', False):
        print("📦 當前架構: 4A+4B模塊化結構")
        print("   📁 4A: vector_operations.py (向量操作核心層)")
        print("   📁 4B: management_api.py (管理API層)")
    else:
        print("📦 當前架構: 5檔案模塊化結構或回退模式")
    
    # 系統狀態檢查
    print(f"🤖 OpenAI Embeddings: {'✅ 已啟用' if OPENAI_EMBEDDINGS_AVAILABLE else '❌ 未啟用'}")
    print(f"🐘 PostgreSQL: {'✅ 已啟用' if detect_railway_environment() else '❌ 未啟用'}")  
    print(f"📊 PGVector: {'✅ 已啟用' if PGVECTOR_AVAILABLE else '❌ 未啟用'}")
    print(f"📄 DOCX 支援: {'✅ 已啟用' if DOCX2TXT_AVAILABLE else '❌ 未啟用'}")
    print(f"📚 EPUB 支援: {'✅ 已啟用' if EPUB_AVAILABLE else '❌ 未啟用'}")
    print(f"🔤 繁簡轉換: {'✅ 已啟用' if OPENCC_AVAILABLE else '❌ 未啟用'}")
    print(f"🧠 智能處理: ✅ 已啟用")
    print(f"🚀 性能優化: ✅ 已啟用")
    
    # 初始化系統
    try:
        system = OptimizedVectorSystem()
        
        # 系統診斷
        print("\n" + "="*60)
        if hasattr(system, 'diagnose_system'):
            system.diagnose_system()
        else:
            print("📊 系統診斷: 基礎版本，完整診斷需要完整模塊")
        print("="*60 + "\n")
        
        # 同步集合（如果支持）
        if hasattr(system, 'sync_collections'):
            changes = system.sync_collections()
            
            # 顯示統計
            if hasattr(system, 'get_stats'):
                stats = system.get_stats()
                print(f"\n📊 集合統計:")
                for folder, count in stats.items():
                    print(f"   📁 {folder}: {count:,} 個文檔分塊")
            
            if changes > 0:
                print(f"\n✅ 處理完成，共 {changes} 個變更")
            else:
                print(f"\n✅ 所有文件都是最新的")
        else:
            print("ℹ️ 完整的同步功能需要完整模塊架構")
        
        return system
        
    except ImportError as e:
        print(f"\n❌ 系統初始化失敗: {e}")
        print(f"📋 建議解決方案:")
        print(f"   1. 確保 vector_operations.py 和 management_api.py 文件存在")
        print(f"   2. 或者創建完整的5檔案結構")
        print(f"   3. 檢查所有必要的依賴是否正確安裝")
        return None
    except Exception as e:
        print(f"\n❌ 執行時錯誤: {e}")
        return None

# =============================================================================
# 🎯 程式入口點 - 只在直接執行時運行
# =============================================================================

if __name__ == "__main__":
    system = main()
    if system and hasattr(system, 'test_knowledge_management'):
        system.test_knowledge_management()
    elif system:
        print("✅ 系統初始化成功，但缺少完整測試功能")
    else:
        print("❌ 系統初始化失敗")

# =============================================================================
# 🔒 向後兼容性導出 - 確保所有現有import語句正常工作
# =============================================================================

__all__ = [
    # 主要系統類別
    'OptimizedVectorSystem',
    
    # 核心配置
    'SYSTEM_CONFIG', 'TOKEN_LIMITS', 'SMART_TEXT_CONFIG', 'PERFORMANCE_CONFIG',
    'SUPPORTED_EXTENSIONS',
    
    # 依賴檢查變數
    'CHROMA_AVAILABLE', 'PGVECTOR_AVAILABLE', 'OPENAI_EMBEDDINGS_AVAILABLE',
    'PSYCOPG2_AVAILABLE', 'JIEBA_AVAILABLE', 'OPENCC_AVAILABLE', 
    'DOCX2TXT_AVAILABLE', 'EPUB_AVAILABLE',
    
    # 資料類別
    'FileInfo', 'TextAnalysis', 'ChunkInfo', 'SearchResult',
    
    # 處理類別
    'SmartTextAnalyzer', 'OptimizedTextSplitter', 'ChineseTextNormalizer', 
    'AdaptiveBatchProcessor',
    
    # 工具函數
    'detect_railway_environment', 'check_openai_api_key',
    
    # 主程式
    'main'
]