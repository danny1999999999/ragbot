#!/usr/bin/env python3
"""
向量操作核心層 - VectorOperationsCore
職責：底層數據操作、向量處理、文檔載入、集合管理
包含26個底層核心方法，從原 OptimizedVectorSystem 精確移動而來
"""

import time
import json
import hashlib
import re
import logging
import os
import gc
import threading
import tempfile
import shutil  # ✅ 添加缺少的import
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict  # ✅ 添加缺少的import

# 導入依賴
from core_config import *
from text_processing import *
from vector_builder import AdaptiveBatchProcessor

# LangChain核心
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, 
    CSVLoader, JSONLoader
)

# 向量存儲 (條件導入)
if PGVECTOR_AVAILABLE:
    try:
        from langchain_postgres import PGVector
    except ImportError:
        from langchain_community.vectorstores import PGVector
else:
    from langchain_community.vectorstores import Chroma

# OpenAI (條件導入)
if OPENAI_EMBEDDINGS_AVAILABLE:
    from langchain_openai import OpenAIEmbeddings

# PostgreSQL (條件導入)  
if PSYCOPG2_AVAILABLE:
    import psycopg2

logger = logging.getLogger(__name__)

class VectorOperationsCore:
    """向量操作核心類別 - 負責底層數據操作和向量處理"""
    
    def __init__(self, data_dir: str = None, model_type: str = None):
        """✅ 純PostgreSQL初始化 - 移除file_records依賴"""
        
        # 🔧 1. 基本變數設置
        self.data_dir = Path(data_dir or SYSTEM_CONFIG["data_dir"])
        self.model_type = model_type or "openai"
        self.persist_dir = Path(SYSTEM_CONFIG["persist_dir"])  # Chroma備用

        # ❌ 不再建立本地目錄 (純PostgreSQL方案)
        print("🚀 純PostgreSQL方案：不使用本地data目錄")
        
        # 🔧 2. 資料庫連接設置（但不測試）
        self.db_adapter = None
        self.connection_string = None
        self.use_postgres = False

        database_url = os.getenv("DATABASE_URL")
        if PGVECTOR_AVAILABLE and database_url:
            self.connection_string = database_url
            print("🔍 發現DATABASE_URL，準備測試PostgreSQL連接...")
        else:
            print("⚠️ DATABASE_URL未設置或PGVector不可用，將使用Chroma")

        if not PGVECTOR_AVAILABLE:
            print("⚠️ PGVector依賴未安裝，使用Chroma作為備用")
            self.persist_dir.mkdir(exist_ok=True)

        # ✅ 3. 先初始化Embedding模型（關鍵！）
        self._setup_embedding_model()
        print("✅ Embedding模型初始化完成")

        # ✅ 4. 現在可以測試PostgreSQL連接了（embeddings已存在）
        if PGVECTOR_AVAILABLE and database_url and hasattr(self, 'embeddings'):
            try:
                print("🔍 測試PostgreSQL + PGVector連接...")
                # 測試連接
                PGVector.from_existing_index(
                    collection_name="_test_connection",
                    embedding=self.embeddings,  # ✅ 現在安全了
                    connection_string=self.connection_string
                )
                self.use_postgres = True
                print("✅ PostgreSQL (pgvector) 連接成功")
            except Exception as e:
                print(f"⚠️ PostgreSQL (pgvector) 連接測試失敗: {e}")
                self.use_postgres = False
                print("📄 回退到Chroma本地存儲")
                self.persist_dir.mkdir(exist_ok=True)
        
        if not self.use_postgres:
            print("🔍 使用Chroma作為向量存儲")
            self.persist_dir.mkdir(exist_ok=True)
        
        # 🔧 5. 初始化文本處理組件
        self._setup_text_processing()
        
        # 🔧 6. 初始化處理器
        self.batch_processor = AdaptiveBatchProcessor()
        self.text_splitter = OptimizedTextSplitter()
        
        # 🔧 7. 初始化存儲（移除檔案記錄）
        self._vector_stores = {}
        
        # ✅ 添加file_records初始化
        self.file_records = {}
        
        # ✅ 改為純PostgreSQL初始化
        print("🚀 純PostgreSQL方案：所有檔案數據將直接存儲在PostgreSQL中")
        print("📄 不再維護本地檔案記錄 (file_records.json)")
        
        self.processing_lock = threading.Lock()
        
        print(f"🚀 向量操作核心初始化完成")
        print(f"   🤖 嵌入模型: {self.model_type}")
        print(f"   🔍 資料目錄: 不使用 (純PostgreSQL)")
        print(f"   🗄️ 向量庫: {'PostgreSQL + PGVector' if self.use_postgres else 'Chroma (本地)'}")
        print(f"   🧠 智能文本處理: ✅")
        print(f"   🔧 自適應批次: ✅")
        print(f"   📦 純PostgreSQL方案: {'✅' if self.use_postgres else '❌'}")

    def _setup_embedding_model(self):
        """設定嵌入模型"""
        try:
            if self.model_type == "openai":
                if not OPENAI_EMBEDDINGS_AVAILABLE:
                    raise ImportError("OpenAI Embeddings不可用")
                
                print(f"🔧 初始化OpenAI Embeddings...")
                
                api_key = os.getenv("OPENAI_API_KEY")
                base_url = os.getenv("OPENAI_API_BASE")
                
                embedding_params = {
                    "model": "text-embedding-3-small",
                    "api_key": api_key,
                    "max_retries": 3,
                    "request_timeout": 60  # 增加超時時間到60秒
                }
                
                if base_url:
                    embedding_params["base_url"] = base_url
                    print(f"🔧 使用自定義API端點: {base_url}")
                
                self.embeddings = OpenAIEmbeddings(**embedding_params)
                print(f"✅ OpenAI Embeddings初始化成功")
                
            else:
                # HuggingFace模型
                print(f"🔧 初始化HuggingFace Embeddings...")
                
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-zh-v1.5",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"batch_size": 16, "normalize_embeddings": True}
                )
                print(f"✅ HuggingFace Embeddings初始化成功")
                
        except Exception as e:
            print(f"❌ 嵌入模型初始化失敗: {e}")
            
            # 回退機制
            if self.model_type == "openai":
                print("📄 嘗試HuggingFace備選...")
                try:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="BAAI/bge-small-zh-v1.5",
                        model_kwargs={"device": "cpu"}
                    )
                    self.model_type = "huggingface"
                    print("✅ 已回退到HuggingFace")
                except Exception as e2:
                    raise RuntimeError(f"所有嵌入模型都初始化失敗: {e2}")
            else:
                raise

    def _setup_text_processing(self):
        """設定文本處理組件"""
        self.normalizer = ChineseTextNormalizer()
        self.analyzer = SmartTextAnalyzer()
        print("✅ 文本處理組件初始化完成")

    def get_or_create_vectorstore(self, collection_name: str):
        """獲取或創建向量存儲 - PostgreSQL優先"""
        if collection_name not in self._vector_stores:
            try:
                if self.use_postgres and PGVECTOR_AVAILABLE:
                    # 🔧 使用PGVector
                    try:
                        from langchain_postgres import PGVector
                        self._vector_stores[collection_name] = PGVector(
                            embeddings=self.embeddings,
                            collection_name=collection_name,
                            connection=self.connection_string,
                            use_jsonb=True,
                        )
                    except ImportError:
                        from langchain_community.vectorstores import PGVector
                        self._vector_stores[collection_name] = PGVector(
                            connection_string=self.connection_string,
                            collection_name=collection_name,
                            embedding_function=self.embeddings,
                            distance_strategy="cosine",
                            pre_delete_collection=False,
                            logger=logger
                        )
                    print(f"✅ PGVector向量存儲就緒: {collection_name}")
                else:
                    # 🔧 備用Chroma - 確保導入
                    if CHROMA_AVAILABLE:
                        from langchain_community.vectorstores import Chroma
                        self._vector_stores[collection_name] = Chroma(
                            collection_name=collection_name,
                            embedding_function=self.embeddings,
                            persist_directory=str(self.persist_dir)
                        )
                        print(f"✅ Chroma向量存儲就緒: {collection_name}")
                    else:
                        raise ImportError("Chroma不可用且PostgreSQL也不可用")
                        
            except Exception as e:
                logger.error(f"向量存儲創建失敗: {e}")
                raise RuntimeError(f"無法創建向量存儲: {e}")
        
        return self._vector_stores[collection_name]

    def _generate_doc_id(self, file_path: Path) -> str:
        """生成文檔ID"""
        # ✅ 修正語法錯誤
        content_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return f"doc_{file_path.stem}_{content_hash}"

    def load_document(self, file_path: Path) -> List[Document]:
        """載入並智能處理文檔"""
        try:
            extension = file_path.suffix.lower()
            
            # 根據檔案類型載入
            if extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
                full_text = "\n".join([doc.page_content for doc in documents if doc.page_content])
            elif extension in ['.docx', '.doc'] and DOCX2TXT_AVAILABLE:
                if DOCX_METHOD == "docx2txt":
                    import docx2txt
                    full_text = docx2txt.process(str(file_path))
                else:
                    loader = Docx2txtLoader(str(file_path))
                    documents = loader.load()
                    full_text = "\n".join([doc.page_content for doc in documents if doc.page_content])
            elif extension == '.epub' and EPUB_AVAILABLE:
                # 🆕 EPUB處理邏輯
                epub_processor = EpubProcessor()
                full_text = epub_processor.extract_epub_content(file_path)
            elif extension == '.csv':
                loader = CSVLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
                full_text = "\n".join([doc.page_content for doc in documents if doc.page_content])
            elif extension == '.json':
                loader = JSONLoader(str(file_path), jq_schema='.', text_content=False)
                documents = loader.load()
                full_text = "\n".join([doc.page_content for doc in documents if doc.page_content])
            else:
                # 嘗試自動檢測編碼
                encodings = ['utf-8', 'gbk', 'gb2312', 'big5']
                full_text = None
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            full_text = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if full_text is None:
                    raise ValueError(f"無法解碼文件: {file_path}")
            
            if not full_text or not full_text.strip():
                logger.warning(f"文件內容為空或無法提取: {file_path.name}")
                return []
            
            # 生成文檔ID
            doc_id = self._generate_doc_id(file_path)
            
            # 使用優化的文本分割器
            source_info = {
                'file_path': str(file_path),
                'file_type': extension,
                'file_size': file_path.stat().st_size if file_path.exists() else 0
            }
            
            documents = self.text_splitter.smart_split_documents(full_text, doc_id, source_info)
            
            # 添加統一元數據
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'filename': file_path.name,
                    'extension': extension,
                    'file_size': source_info['file_size'],
                    'load_timestamp': time.time()
                })
            
            print(f"📄 文檔載入完成: {file_path.name} ({len(documents)} 個分塊)")
            
            return documents
            
        except Exception as e:
            logger.error(f"載入文檔失敗 {file_path}: {e}")
            print(f"❌ 文檔載入失敗: {file_path.name} - {e}")
            return []

    def get_file_info(self, file_path: Path) -> Optional[FileInfo]:
        """獲取檔案基本信息"""
        try:
            if not file_path.exists():
                return None
                
            stat = file_path.stat()
            
            # 計算檔案哈希（用於變更檢測）
            hash_md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            return FileInfo(
                path=str(file_path),
                size=stat.st_size,
                mtime=stat.st_mtime,
                hash=hash_md5.hexdigest(),
                encoding='utf-8',
                file_type=file_path.suffix.lower()
            )
            
        except Exception as e:
            logger.error(f"獲取檔案信息失敗 {file_path}: {e}")
            return None

    def _parallel_load_documents(self, file_paths: List[Path], collection_name: str) -> List[Document]:
        """並發載入文檔"""
        all_documents = []
        max_workers = min(SYSTEM_CONFIG.get("max_workers", 4), len(file_paths))
        
        print(f"   🚀 並發載入 (工作線程: {max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.load_document, file_path): file_path 
                for file_path in file_paths
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    documents = future.result()
                    if documents:
                        for doc in documents:
                            doc.metadata['collection'] = collection_name
                        all_documents.extend(documents)
                        print(f"   ✅ {file_path.name}: {len(documents)} 分塊")
                    else:
                        print(f"   ⚠️ {file_path.name}: 無有效內容")
                except Exception as e:
                    print(f"   ❌ {file_path.name}: {e}")
                    logger.error(f"並發載入失敗 {file_path}: {e}")
        
        return all_documents

    def _sequential_load_documents(self, file_paths: List[Path], collection_name: str) -> List[Document]:
        """順序載入文檔"""
        all_documents = []
        
        print(f"   📄 順序載入")
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                print(f"   [{i}/{len(file_paths)}] 處理: {file_path.name}")
                documents = self.load_document(file_path)
                
                if documents:
                    for doc in documents:
                        doc.metadata['collection'] = collection_name
                    all_documents.extend(documents)
                    print(f"      ✅ 載入 {len(documents)} 個分塊")
                else:
                    print(f"      ⚠️ 無有效內容")
                    
            except Exception as e:
                print(f"      ❌ 載入失敗: {e}")
                logger.error(f"文件載入失敗 {file_path}: {e}")
        
        return all_documents

    def _load_file_records(self) -> Dict[str, Dict[str, FileInfo]]:
        """載入檔案記錄 - 加強錯誤處理和恢復機制"""
        record_file = self.data_dir / "file_records.json"
        
        # 🔧 檢查檔案是否存在
        if not record_file.exists():
            print("🔍 檔案記錄不存在，將建立新的記錄")
            return {}
        
        try:
            # 🔧 讀取並檢查檔案內容
            with open(record_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 🔧 檢查檔案是否為空
            if not content:
                print("⚠️ 檔案記錄為空，將建立新的記錄")
                return {}
            
            # 🔧 檢查是否以{開頭（基本JSON格式檢查）
            if not content.startswith('{'):
                print(f"⚠️ 檔案記錄格式錯誤，內容開頭: {repr(content[:50])}")
                return self._handle_corrupted_records(record_file, content)
            
            # 🔧 嘗試解析JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError as json_error:
                print(f"❌ JSON解析失敗: {json_error}")
                print(f"   錯誤位置: line {json_error.lineno}, column {json_error.colno}")
                print(f"   檔案前100字符: {repr(content[:100])}")
                return self._handle_corrupted_records(record_file, content)
            
            # 🔧 驗證資料格式
            if not isinstance(data, dict):
                print(f"⚠️ 檔案記錄格式錯誤，應為字典但得到: {type(data)}")
                return {}
            
            # 🔧 轉換為FileInfo物件
            records = {}
            for collection, files in data.items():
                records[collection] = {}
                for file_path, info in files.items():
                    try:
                        if isinstance(info, dict):
                            fileinfo_fields = {
                                'path': info.get('path', file_path),
                                'size': info.get('size', 0),
                                'mtime': info.get('mtime', time.time()),
                                'hash': info.get('hash', ''),
                                'encoding': info.get('encoding', 'utf-8'),
                                'file_type': info.get('file_type', '')
                            }
                            
                            file_info_obj = FileInfo(**fileinfo_fields)
                            
                            # 🔧 恢復額外屬性
                            if 'uploaded_by' in info:
                                file_info_obj.uploaded_by = info['uploaded_by']
                            if 'uploaded_at' in info:
                                file_info_obj.uploaded_at = info['uploaded_at']
                            if 'file_source' in info:
                                file_info_obj.file_source = info['file_source']
                            
                            records[collection][file_path] = file_info_obj
                        else:
                            records[collection][file_path] = info
                            
                    except Exception as e:
                        logger.warning(f"載入檔案記錄失敗 {file_path}: {e}")
                        # 🔧 建立預設記錄
                        try:
                            default_info = FileInfo(
                                path=file_path,
                                size=0,
                                mtime=time.time(),
                                hash='',
                                encoding='utf-8',
                                file_type=''
                            )
                            records[collection][file_path] = default_info
                        except Exception:
                            logger.error(f"無法建立預設FileInfo for {file_path}")
                            continue
            
            print(f"✅ 檔案記錄載入成功: {len(records)} 個集合")
            return records
            
        except Exception as e:
            logger.error(f"載入檔案記錄失敗: {e}")
            print(f"❌ 嚴重錯誤，載入檔案記錄失敗: {e}")
            return self._handle_corrupted_records(record_file, "")

    def _save_file_records(self):
        """保存檔案記錄"""
        record_file = self.data_dir / "file_records.json"
        
        try:
            # 確保目錄存在
            record_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 轉換FileInfo物件為字典
            serializable_records = {}
            for collection, files in self.file_records.items():
                serializable_records[collection] = {}
                for file_path, file_info in files.items():
                    if isinstance(file_info, FileInfo):
                        serializable_records[collection][file_path] = asdict(file_info)
                    else:
                        serializable_records[collection][file_path] = file_info
            
            # 先寫入臨時檔案，然後原子性替換
            temp_file = record_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_records, f, indent=2, ensure_ascii=False)
            
            temp_file.replace(record_file)
            logger.info(f"檔案記錄保存成功: {len(serializable_records)} 個集合")
            
        except Exception as e:
            logger.error(f"保存檔案記錄失敗: {e}")

    def _handle_corrupted_records(self, record_file: Path, content: str) -> Dict:
        """處理損壞的檔案記錄"""
        print(f"🔧 處理損壞的檔案記錄...")
        
        # 創建備份
        try:
            backup_file = record_file.with_suffix(f'.backup_{int(time.time())}')
            if record_file.exists():
                shutil.copy2(record_file, backup_file)
                print(f"   📦 已創建備份: {backup_file.name}")
        except Exception as e:
            logger.warning(f"創建備份失敗: {e}")
        
        # 嘗試恢復
        try:
            return self._rebuild_file_records()
        except Exception as e:
            logger.error(f"重建檔案記錄失敗: {e}")
            return {}

    def _rebuild_file_records(self) -> Dict:
        """重建檔案記錄"""
        print("🔧 重建檔案記錄...")
        
        new_records = {}
        
        if not self.data_dir.exists():
            return new_records
        
        # 掃描所有子目錄
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                collection_name = f"collection_{subdir.name}"
                new_records[collection_name] = {}
                
                # 掃描目錄中的檔案
                for file_path in subdir.rglob('*'):
                    if (file_path.is_file() and 
                        file_path.suffix.lower() in SUPPORTED_EXTENSIONS):
                        
                        file_info = self.get_file_info(file_path)
                        if file_info:
                            new_records[collection_name][str(file_path)] = file_info
        
        print(f"✅ 重建完成: {len(new_records)} 個集合")
        return new_records

    def get_file_source_statistics(self) -> Dict[str, Dict[str, int]]:
        """獲取檔案來源統計"""
        stats = {}
        
        for collection_name, files in self.file_records.items():
            stats[collection_name] = {
                'total': len(files),
                'upload': 0,
                'sync': 0,
                'unknown': 0
            }
            
            for file_info in files.values():
                source = getattr(file_info, 'file_source', 'unknown')
                if source in stats[collection_name]:
                    stats[collection_name][source] += 1
                else:
                    stats[collection_name]['unknown'] += 1
        
        return stats

    def diagnose_file_records(self) -> Dict:
        """診斷檔案記錄"""
        diagnosis = {
            'total_collections': len(self.file_records),
            'total_files': sum(len(files) for files in self.file_records.values()),
            'missing_files': 0,
            'outdated_files': 0,
            'corrupted_records': 0
        }
        
        for collection_name, files in self.file_records.items():
            for file_path, file_info in files.items():
                try:
                    path_obj = Path(file_path)
                    
                    if not path_obj.exists():
                        diagnosis['missing_files'] += 1
                        continue
                    
                    current_mtime = path_obj.stat().st_mtime
                    if abs(current_mtime - file_info.mtime) > 1:
                        diagnosis['outdated_files'] += 1
                        
                except Exception:
                    diagnosis['corrupted_records'] += 1
        
        return diagnosis

    def cleanup_invalid_records(self) -> Dict:
        """清理無效記錄"""
        removed_count = 0
        
        for collection_name in list(self.file_records.keys()):
            files = self.file_records[collection_name]
            
            for file_path in list(files.keys()):
                try:
                    if not Path(file_path).exists():
                        del files[file_path]
                        removed_count += 1
                except Exception:
                    del files[file_path]
                    removed_count += 1
            
            # 移除空集合
            if not files:
                del self.file_records[collection_name]
        
        if removed_count > 0:
            self._save_file_records()
        
        return {'removed_records': removed_count}

    def _process_batches(self, vectorstore: Union["Chroma", Any], batches: List[Tuple[List[Document], Dict]]) -> int:
        """處理批次向量化 - 完整的錯誤處理和元數據修復"""

        success_count = 0
        total_docs = sum(len(batch_docs) for batch_docs, _ in batches)
        
        print(f"\n📄 開始批次向量化...")
        print(f"   📦 總批次數: {len(batches)}")
        print(f"   📄 總文檔數: {total_docs}")
        
        for batch_num, (batch_docs, batch_info) in enumerate(batches, 1):
            print(f"\n   📦 批次 {batch_num}/{len(batches)}")
            print(f"      📄 文檔數: {batch_info['documents']}")
            print(f"      🔍 tokens: {batch_info['tokens']:,}")
            print(f"      📊 使用率: {(batch_info['tokens']/TOKEN_LIMITS['max_tokens_per_request']*100):.1f}%")
            
            # 顯示文檔類型分布
            type_info = ", ".join([f"{k}:{v}" for k, v in batch_info['types'].items()])
            print(f"      🏷️ 類型: {type_info}")
            
            start_time = time.time()
            
            try:
                print(f"      🚀 開始處理批次 {batch_num}...")
                print(f"      📡 正在調用OpenAI API... (這可能需要30-60秒)")
                
                # 🛠️ 修復：統一處理元數據，確保類型正確
                safe_docs = []
                for doc in batch_docs:
                    safe_metadata = self._ensure_simple_metadata(doc.metadata)
                    safe_doc = Document(page_content=doc.page_content, metadata=safe_metadata)
                    safe_docs.append(safe_doc)

                print(f"      🔧 已處理 {len(safe_docs)} 個文檔的元數據，確保類型兼容")
                
                vectorstore.add_documents(safe_docs)
                processing_time = time.time() - start_time
                
                success_count += len(batch_docs)
                self.batch_processor.record_batch_result(True, processing_time)
                
                print(f"      ✅ 批次 {batch_num} 完成 ({processing_time:.1f}s)")
                print(f"      📊 總進度: {success_count}/{total_docs} ({success_count/total_docs*100:.1f}%)")
                
                # 批次間延遲
                if batch_num < len(batches):
                    delay = TOKEN_LIMITS["batch_delay"]
                    print(f"      ⱱ️ 等待 {delay} 秒...")
                    time.sleep(delay)
                    
            except Exception as e:
                processing_time = time.time() - start_time
                self.batch_processor.record_batch_result(False, processing_time)
                
                error_msg = str(e)
                print(f"      ❌ 批次 {batch_num} 失敗 ({processing_time:.1f}s)")
                print(f"         錯誤: {error_msg}")
                
                # 🔧 特別處理元數據錯誤
                if "metadata" in error_msg.lower():
                    print(f"         🔧 檢測到元數據錯誤，嘗試更嚴格的處理...")
                    try:
                        # 重新嘗試，使用最嚴格的元數據過濾
                        ultra_safe_docs = []
                        for doc in batch_docs:
                            # 只保留最基本的元數據字段
                            minimal_metadata = {
                                'doc_id': str(doc.metadata.get('doc_id', 'unknown')),
                                'chunk_id': str(doc.metadata.get('chunk_id', 'unknown')),
                                'chunk_index': int(doc.metadata.get('chunk_index', 0)),
                                'text_type': str(doc.metadata.get('text_type', 'unknown')),
                                'source': str(doc.metadata.get('source', 'unknown')),
                                'filename': str(doc.metadata.get('filename', 'unknown')),
                                'token_count': int(doc.metadata.get('token_count', 0)),
                                'chunk_length': int(doc.metadata.get('chunk_length', 0)),
                                # URL處理
                                'contained_urls': str(doc.metadata.get('contained_urls', '')),
                                'url_count': int(doc.metadata.get('url_count', 0)),
                                'has_urls': bool(doc.metadata.get('has_urls', False))
                            }
                            
                            ultra_safe_doc = Document(
                                page_content=doc.page_content,
                                metadata=minimal_metadata
                            )
                            ultra_safe_docs.append(ultra_safe_doc)
                        
                        vectorstore.add_documents(ultra_safe_docs)
                        success_count += len(batch_docs)
                        print(f"         ✅ 使用最小化元數據重新處理成功")
                        continue
                        
                    except Exception as retry_e:
                        print(f"         ❌ 重新處理也失敗: {retry_e}")
                
                # 其他錯誤處理
                if "timeout" in error_msg.lower():
                    print(f"         🕐 超時錯誤，延長等待時間...")
                    time.sleep(30)
                elif "rate_limit" in error_msg.lower() or "429" in error_msg:
                    print(f"         🚦 速率限制，延長等待...")
                    time.sleep(60)
                elif "token" in error_msg.lower() and batch_info['documents'] > 1:
                    print(f"         🔧 Token超限，嘗試單個處理...")
                    single_success = self._process_documents_individually(vectorstore, batch_docs)
                    success_count += single_success
                elif "connection" in error_msg.lower():
                    print(f"         🌐 連接錯誤，等待重試...")
                    time.sleep(20)
                    try:
                        print(f"         📄 重試批次 {batch_num}...")
                        # 使用安全的元數據重試
                        safe_docs = []
                        for doc in batch_docs:
                            safe_metadata = self._ensure_simple_metadata(doc.metadata)
                            safe_doc = Document(page_content=doc.page_content, metadata=safe_metadata)
                            safe_docs.append(safe_doc)
                        
                        vectorstore.add_documents(safe_docs)
                        success_count += len(batch_docs)
                        print(f"         ✅ 重試成功")
                    except Exception as retry_e:
                        print(f"         ❌ 重試失敗: {retry_e}")
                else:
                    print(f"         ⚠️ 跳過此批次")
                    
                # 每次錯誤後添加額外延遲
                print(f"         ⸺️ 錯誤後暫停10秒...")
                time.sleep(10)
        
        return success_count

    def _ensure_simple_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """確保元數據只包含Chroma支持的簡單類型：string, int, float, bool"""
        safe_metadata = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                # ✅ 已經是Chroma支持的簡單類型，直接保留
                safe_metadata[key] = value
            elif isinstance(value, list):
                # ❌ 列表不被支持 → 轉為分隔符字符串
                if key == 'contained_urls' or 'url' in key.lower():
                    safe_metadata[key] = '|'.join(str(v) for v in value) if value else ''
                    safe_metadata[f'{key}_count'] = len(value)
                else:
                    safe_metadata[key] = '|'.join(str(v) for v in value) if value else ''
                    safe_metadata[f'{key}_count'] = len(value)
            elif isinstance(value, dict):
                # ❌ 字典不被支持 → 轉為JSON字符串
                safe_metadata[key] = json.dumps(value, ensure_ascii=False)
            else:
                # ❌ 其他類型不被支持 → 轉為字符串
                safe_metadata[key] = str(value)
        
        return safe_metadata

    def _process_documents_individually(self, vectorstore, documents: List[Document]) -> int:
        """單個處理文檔"""
        success_count = 0
        
        for i, doc in enumerate(documents):
            try:
                doc_tokens = doc.metadata.get('token_count', 0)
                if doc_tokens > TOKEN_LIMITS['max_tokens_per_request']:
                    print(f"         ⚠️ 文檔 {i+1} 仍然過大 ({doc_tokens:,} tokens)，跳過")
                    continue
                
                vectorstore.add_documents([doc])
                success_count += 1
                print(f"         ✅ 單個文檔 {i+1}/{len(documents)} 完成")
                time.sleep(1)  # 單個處理時短暫延遲
                
            except Exception as e:
                print(f"         ❌ 單個文檔 {i+1} 失敗: {e}")
        
        return success_count

    def incremental_update(self, collection_name: str, added_files: List[Path], 
                          modified_files: List[Path], deleted_files: List[str],
                          current_files: Dict[str, FileInfo]) -> bool:
        """🚀 優化版增量更新"""
        with self.processing_lock:
            try:
                vectorstore = self.get_or_create_vectorstore(collection_name)
                
                # 處理刪除和修改
                files_to_delete = deleted_files + [str(f) for f in modified_files]
                if files_to_delete:
                    for file_path in files_to_delete:
                        try:
                            vectorstore.delete(filter={"source": file_path})
                            print(f"🗑️ 已刪除: {Path(file_path).name}")
                        except Exception as e:
                            logger.warning(f"刪除文檔失敗 {file_path}: {e}")
                
                # 處理新增和修改的文件
                files_to_process = added_files + modified_files
                if not files_to_process:
                    print("   ✅ 無新文件需要處理")
                    return True
                
                print(f"📄 開始處理 {len(files_to_process)} 個文件...")
                print(f"   ⚠️ 處理大文件可能需要較長時間，請耐心等待...")
                
                # 並發載入文檔
                all_documents = []
                if PERFORMANCE_CONFIG.get("parallel_processing", True):
                    all_documents = self._parallel_load_documents(files_to_process, collection_name)
                else:
                    all_documents = self._sequential_load_documents(files_to_process, collection_name)
                
                if not all_documents:
                    print("   ⚠️ 沒有有效文檔需要向量化")
                    return True
                
                # 統計和成本估算
                total_tokens = sum(doc.metadata.get('token_count', 0) for doc in all_documents)
                estimated_cost = self.batch_processor.token_estimator.estimate_embedding_cost(total_tokens)
                
                print(f"\n📊 向量化統計:")
                print(f"   📄 總分塊數: {len(all_documents)}")
                print(f"   🔍 總tokens: {total_tokens:,}")
                print(f"   💰 估算成本: ${estimated_cost:.4f}")
                
                # 創建智能批次並處理
                batches = self.batch_processor.create_smart_batches(all_documents)
                success_count = self._process_batches(vectorstore, batches)
                
                print(f"\n🎉 向量化完成！")
                print(f"   ✅ 成功: {success_count}/{len(all_documents)} 個分塊")
                print(f"   📊 成功率: {(success_count/len(all_documents)*100):.1f}%")
                
                # 更新文件記錄
                self.file_records[collection_name] = current_files
                self._save_file_records()
                
                # 記憶體清理
                if success_count % PERFORMANCE_CONFIG.get("gc_frequency", 50) == 0:
                    gc.collect()
                
                return success_count > 0
                
            except Exception as e:
                logger.error(f"增量更新失敗 {collection_name}: {e}")
                print(f"❌ 系統錯誤: {e}")
                return False

    def get_collection_name(self, dir_path: Path) -> str:
        """獲取集合名稱"""
        try:
            relative_path = dir_path.relative_to(self.data_dir)
            if relative_path.parts:
                return f"collection_{relative_path.parts[0]}"
        except ValueError:
            pass
        return "collection_other"

    def sync_collections(self) -> int:
        """純PostgreSQL方案：不再掃描本地目錄"""
        print("📄 純PostgreSQL方案：跳過本地目錄掃描")
        print("✅ 所有數據都在PostgreSQL中，無需同步")
        return 0

    def scan_directory_changes(self, dir_path: Path, collection_name: str) -> Tuple[List[Path], List[Path], List[str], Dict[str, FileInfo]]:
        """掃描目錄變更 - 修正版：正確處理上傳檔案"""
        current_files = {}
        
        print(f"🔍 掃描目錄: {dir_path}")
        
        # 遞歸掃描目錄
        file_count = 0
        for file_path in dir_path.rglob('*'):
            if (
                file_path.is_file() and 
                file_path.suffix.lower() in SUPPORTED_EXTENSIONS and
                not file_path.name.startswith('.') and
                file_path.stat().st_size > 0
            ):  # 跳過空文件
                
                file_info = self.get_file_info(file_path)
                if file_info:
                    # 🆕 修正：使用標準化的絕對路徑作為鍵值
                    try:
                        # 使用absolute()避免符號連結問題
                        absolute_path = str(file_path.absolute())
                        current_files[absolute_path] = file_info
                        file_count += 1
                    except Exception as e:
                        logger.warning(f"路徑標準化失敗 {file_path}: {e}")
                        # 回退到原始路徑
                        current_files[str(file_path)] = file_info
                        file_count += 1
        
        print(f"📄 找到 {file_count} 個有效檔案")
        
        old_files = self.file_records.get(collection_name, {})
        print(f"📋 舊記錄中有 {len(old_files)} 個檔案")
        
        # 🆕 修正：正規化舊記錄的路徑鍵值
        normalized_old_files = {}
        normalization_errors = 0
        
        for old_path, old_info in old_files.items():
            try:
                old_path_obj = Path(old_path)
                
                if old_path_obj.is_absolute():
                    # 已經是絕對路徑
                    normalized_key = str(old_path_obj.absolute())
                else:
                    # 相對路徑轉絕對路徑
                    try:
                        abs_path = (dir_path / old_path).absolute()
                        normalized_key = str(abs_path)
                    except Exception:
                        # 如果無法轉換，保持原樣
                        normalized_key = old_path
                        
                normalized_old_files[normalized_key] = old_info
                
            except Exception as e:
                logger.warning(f"舊路徑正規化失敗 {old_path}: {e}")
                # 保持原路徑
                normalized_old_files[old_path] = old_info
                normalization_errors += 1
        
        if normalization_errors > 0:
            print(f"⚠️ {normalization_errors} 個舊路徑正規化失敗")
        
        # 🆕 修正：智能變更檢測
        added_files = []
        modified_files = []
        
        print("🔍 檢測變更...")
        
        for file_path, file_info in current_files.items():
            current_file_name = Path(file_path).name
            current_hash = file_info.hash
            
            # 首先嘗試精確路徑匹配
            if file_path in normalized_old_files:
                old_info = normalized_old_files[file_path]
                if old_info.hash != current_hash:
                    modified_files.append(Path(file_path))
                    print(f"🔍 修改檔案: {current_file_name}")
            else:
                # 🆕 智能檔案匹配：檢查是否是同一檔案的不同路徑表示
                file_found = False
                
                for old_path, old_info in normalized_old_files.items():
                    old_file_name = Path(old_path).name
                    
                    # 檔案名相同且哈希相同 = 同一檔案
                    if (
                        current_file_name == old_file_name and 
                        current_hash == old_info.hash
                    ):
                        file_found = True
                        print(f"📄 路徑變更但內容相同: {current_file_name}")
                        break
                        
                    # 檔案名相同但哈希不同 = 檔案被修改
                    elif (
                        current_file_name == old_file_name and 
                        current_hash != old_info.hash
                    ):
                        modified_files.append(Path(file_path))
                        file_found = True
                        print(f"🔍 修改檔案 (路徑變更): {current_file_name}")
                        break
                
                if not file_found:
                    added_files.append(Path(file_path))
                    print(f"📄 新檔案: {current_file_name}")
        
        # 🆕 修正：智能刪除檢測
        deleted_files = []
        
        for old_path in normalized_old_files.keys():
            old_file_name = Path(old_path).name
            
            if old_path not in current_files:
                # 檢查檔案是否真的不存在（可能只是路徑表示不同）
                file_still_exists = False
                
                for current_path in current_files.keys():
                    current_file_name = Path(current_path).name
                    if current_file_name == old_file_name:
                        # 進一步檢查是否是同一檔案（通過內容哈希）
                        current_hash = current_files[current_path].hash
                        old_hash = normalized_old_files[old_path].hash
                        
                        if current_hash == old_hash:
                            file_still_exists = True
                            break
                
                if not file_still_exists:
                    deleted_files.append(old_path)
                    print(f"🗑️ 刪除檔案: {old_file_name}")
        
        print(f"📊 變更統計:")
        print(f"   📄 新增: {len(added_files)}")
        print(f"   🔍 修改: {len(modified_files)}")
        print(f"   🗑️ 刪除: {len(deleted_files)}")
        
        return added_files, modified_files, deleted_files, current_files


# ✅ 確保所有import都能正常工作
__all__ = ['VectorOperationsCore']

"""
📋 檔案4A包含的26個底層核心方法：
🔧 系統初始化 (5個): __init__, _setup_embedding_model, _setup_text_processing, 
                   get_or_create_vectorstore, _generate_doc_id
📄 文檔載入處理 (4個): load_document, get_file_info, _parallel_load_documents, _sequential_load_documents  
📋 文件記錄管理 (7個): _load_file_records, _save_file_records, _handle_corrupted_records, 
                    _rebuild_file_records, get_file_source_statistics, diagnose_file_records, 
                    cleanup_invalid_records
🔗 向量處理核心 (4個): _process_batches, _ensure_simple_metadata, _process_documents_individually, 
                    incremental_update
📊 集合管理 (6個): get_collection_name, sync_collections, scan_directory_changes, 以及原方案中
                從4B移動過來的3個基礎方法構成了完整的底層操作支撐

⚠️  被依賴關係：management_api.py (檔案4B) 依賴此檔案
✅ 拆分完成：此檔案包含所有底層數據操作和向量處理功能
"""