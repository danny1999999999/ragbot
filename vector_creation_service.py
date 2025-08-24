import time
import json
import shutil
import time
import hashlib
import re
import logging
import os
import gc
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, asdict
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
# 🔧 強制載入環境變數
from dotenv import load_dotenv
from database_adapter import DatabaseFactory, PostgreSQLAdapter
load_dotenv('.env', override=True)
logger = logging.getLogger(__name__)

import urllib.parse
from urllib.parse import quote_plus

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from config import app_config  # ⭐ 統一導入



# 🆕 1. 添加 Railway 環境檢測函數（放在文件頂部，導入語句之後）
def detect_railway_environment():
    """檢測是否在 Railway 環境中運行"""
    railway_indicators = [
        os.getenv("RAILWAY_PROJECT_ID"),
        os.getenv("RAILWAY_SERVICE_ID"), 
        os.getenv("DATABASE_URL"),
        "railway.internal" in os.getenv("POSTGRES_HOST", "")
    ]
    
    is_railway = any(railway_indicators)
    if is_railway:
        print("🚂 檢測到 Railway 部署環境")
        # 顯示 Railway 特定信息
        project_id = os.getenv("RAILWAY_PROJECT_ID", "unknown")
        service_id = os.getenv("RAILWAY_SERVICE_ID", "unknown")  
        print(f"   項目ID: {project_id[:8]}***")
        print(f"   服務ID: {service_id[:8]}***")
    
    return is_railway


# 🔧 優化版系統配置
SYSTEM_CONFIG = {
    "persist_dir": "./chroma_langchain_db",
    "data_dir": "./data", 
    "device": "cpu",  
    "log_level": "INFO",
    "max_workers": 4,  # 並發處理數
    "cache_embeddings": True,  # 快取嵌入向量
    "backup_enabled": True  # 自動備份
}

# 🧠 智能文本分類配置
SMART_TEXT_CONFIG = {
    # 文本長度閾值（字符數）
    "micro_text_threshold": 100,      # 微文本：標題、摘要等
    "short_text_threshold": 500,      # 短文本：短文章、段落
    "medium_text_threshold": 2000,    # 中等文本：中篇文章
    "long_text_threshold": 8000,      # 長文本：長篇文章
    "ultra_long_threshold": 20000,    # 超長文本：書籍章節
    
    # 各類文本的處理策略
    "micro_text": {
        "min_length": 20,
        "process_whole": True,
        "merge_with_next": True,  # 嘗試與下一個片段合併
        "chunk_size": 0,
        "description": "微文本整體處理"
    },
    
    "short_text": {
        "min_length": 50,
        "process_whole": True,
        "preserve_structure": True,
        "chunk_size": 0,
        "allow_merge": True,  # 允許合併相鄰短文本
        "description": "短文本整體保存"
    },
    
    "medium_text": {
        "chunk_size": 800,
        "chunk_overlap": 120,
        "preserve_paragraphs": True,
        "smart_boundaries": True,  # 智能邊界檢測
        "min_chunk_ratio": 0.3,   # 最小片段比例
        "description": "中等文本段落感知分割"
    },
    
    "long_text": {
        "chunk_size": 500,
        "chunk_overlap": 80,
        "hierarchical_split": True,
        "section_aware": True,    # 章節感知
        "quality_check": True,    # 質量檢查
        "description": "長文本精細化分割"
    },
    
    "ultra_long": {
        "chunk_size": 600,
        "chunk_overlap": 100,
        "multi_level_split": True,  # 多層級分割
        "chapter_detection": True,  # 章節檢測
        "summary_chunks": True,     # 生成摘要片段
        "description": "超長文本階層式處理"
    },
    
    "mega_text": {
        "chunk_size": 500,
        "chunk_overlap": 80,
        "multi_level_split": True,  # 多層級分割
        "chapter_detection": True,  # 章節檢測
        "aggressive_split": True,   # 積極分割
        "quality_filter": True,     # 質量過濾
        "description": "超大文本積極分割處理"
    }
}

# 🔧 優化版 Token 限制配置
TOKEN_LIMITS = {
    "max_tokens_per_request": 150000,  # 更保守的限制
    "max_batch_size": 8,               # 減小批次大小
    "min_batch_size": 1,
    "batch_delay": 5.0,                # 增加批次間延遲
    "retry_delay": 10,
    "max_retries": 3,
    "token_safety_margin": 0.2,        # 20% 安全邊際
    "adaptive_batching": True           # 自適應批次大小
}

# 🔧 性能優化配置
PERFORMANCE_CONFIG = {
    "embedding_batch_size": 12,
    "parallel_processing": True,
    "memory_limit_mb": 1024,      # 記憶體限制
    "chunk_cache_size": 1000,     # 分塊快取大小
    "preload_models": True,       # 預載入模型
    "gc_frequency": 50            # 垃圾回收頻率
}

# 支援的文件格式擴展
SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.pdf', '.csv', '.json', '.py', '.js', 
    '.docx', '.doc', '.epub', '.rst', '.org'
}

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    # 創建一個虛擬的 Chroma 類型用於類型註解
    class Chroma:
        pass




# 🔧 檢查環境變數
def check_openai_api_key():
    """檢查 OpenAI API Key"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  未設置 OPENAI_API_KEY 環境變數")
        print("   請在 .env 文件中設置: OPENAI_API_KEY=sk-your-api-key")
        return False
    
    if api_key.startswith("sk-") and len(api_key) > 20:
        print("✅ OpenAI API Key 格式正確")
        print(f"   API Key: {api_key[:8]}***{api_key[-4:]}")
        return True
    else:
        print("⚠️  OpenAI API Key 格式可能不正確")
        return False

# LangChain 核心組件導入
try:
    from langchain_postgres import PGVector
    PGVECTOR_AVAILABLE = True
    print("✅ PGVector 可用")
except ImportError:
    try:
        from langchain_community.vectorstores import PGVector
        PGVECTOR_AVAILABLE = True
        print("✅ PGVector (community) 可用")
    except ImportError:
        PGVECTOR_AVAILABLE = False
        print("❌ PGVector 不可用")
        # 可以回退到 Chroma
        from langchain_community.vectorstores import Chroma

# OpenAI embeddings 導入
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
    print("✅ OpenAI Embeddings 可用")
    
    if check_openai_api_key():
        print("✅ OpenAI 環境檢查通過")
    else:
        print("⚠️ OpenAI 環境配置可能有問題")
        
except ImportError as e:
    OPENAI_EMBEDDINGS_AVAILABLE = False
    print(f"⚠️ OpenAI Embeddings 不可用: {e}")

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, 
    CSVLoader, JSONLoader
)

# 🔧 PostgreSQL 依賴檢查
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("⚠️ 警告: psycopg2 未安裝，請執行: pip install psycopg2-binary")
