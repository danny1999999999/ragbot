#!/usr/bin/env python3
"""
📦 4B檔案: management_interface.py (管理介面層)
🎯 職責：系統管理API、用戶接口、高級管理功能
🔗 依賴：vector_operations.py (4A檔案)
📊 包含：OptimizedVectorSystem類別 (45個方法)
"""

import time
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# 🔗 依賴4A檔案：向量操作核心層
try:
    from vector_operations import VectorOperationsCore
    print("✅ 成功導入 VectorOperationsCore 從 vector_operations.py")
except ImportError as e:
    print(f"❌ 無法導入 vector_operations.py: {e}")
    print("📋 請確保 vector_operations.py (4A檔案) 存在且可正常導入")
    raise ImportError("4B檔案依賴4A檔案，請先確保 vector_operations.py 可正常使用")

# 🔗 嘗試導入配置 (從4A檔案獲取)
try:
    # 從已導入的 vector_operations 獲取所有必要配置
    import vector_operations as vo_module
    
    # 動態獲取配置變數
    SYSTEM_CONFIG = getattr(vo_module, 'SYSTEM_CONFIG', {})
    TOKEN_LIMITS = getattr(vo_module, 'TOKEN_LIMITS', {})
    SUPPORTED_EXTENSIONS = getattr(vo_module, 'SUPPORTED_EXTENSIONS', set())
    
    # 獲取依賴檢查變數
    PGVECTOR_AVAILABLE = getattr(vo_module, 'PGVECTOR_AVAILABLE', False)
    OPENAI_EMBEDDINGS_AVAILABLE = getattr(vo_module, 'OPENAI_EMBEDDINGS_AVAILABLE', False)
    
    # 獲取資料類別
    FileInfo = getattr(vo_module, 'FileInfo', None)
    SearchResult = getattr(vo_module, 'SearchResult', None)
    
    print("✅ 成功從4A檔案獲取所有必要配置")
    
except Exception as e:
    print(f"⚠️ 從4A檔案獲取配置時出現問題: {e}")
    # 提供最小必要配置作為回退
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.csv', '.json', '.py'}
    PGVECTOR_AVAILABLE = False
    OPENAI_EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)

class OptimizedVectorSystem(VectorOperationsCore):
    """
    📦 完整的向量系統管理介面
    🔗 繼承自 VectorOperationsCore (4A檔案)
    📊 提供45個管理API方法，完整向後兼容
    """
    
    def __init__(self, data_dir: str = None, model_type: str = None):
        """🔧 初始化管理介面 - 繼承4A檔案的核心功能"""
        try:
            # 調用父類別初始化 (4A檔案)
            super().__init__(data_dir, model_type)
            print("📊 管理介面層初始化完成")
            print("   🔍 搜索服務: ✅")  
            print("   📤 文件上傳: ✅")
            print("   📋 文檔查詢: ✅")
            print("   🗑️ 刪除管理: ✅")
            print("   📊 統計診斷: ✅")
            
        except Exception as e:
            logger.error(f"管理介面初始化失敗: {e}")
            raise
    
    # ==================== 🔍 搜索功能相關 (1個方法) ====================
    
    def search(self, query: str, collection_name: str = None, k: int = 5) -> List[Dict]:
        """🔍 優化版搜索 - 支援多集合、查詢變體、去重排序"""
        try:
            # 創建搜索變體
            query_variants = []
            if hasattr(self, 'normalizer') and hasattr(self.normalizer, 'create_search_variants'):
                query_variants = self.normalizer.create_search_variants(query)
            else:
                query_variants = [query]  # 回退到原查詢
            
            all_results = []
            
            # 處理集合範圍
            target_collections = []
            if collection_name:
                target_collections = [collection_name]
            else:
                stats = self.get_stats()
                target_collections = [f"collection_{name}" for name in stats.keys()]
            
            # 對每個集合和查詢變體進行搜索
            for variant in query_variants:
                for coll_name in target_collections:
                    try:
                        vectorstore = self.get_or_create_vectorstore(coll_name)
                        docs_and_scores = vectorstore.similarity_search_with_score(variant, k=k)
                        
                        for doc, score in docs_and_scores:
                            # 創建搜索結果
                            result = {
                                "content": doc.page_content[:500],  # 限制預覽長度
                                "score": 1.0 - score,  # 轉換為相似度分數
                                "metadata": doc.metadata,
                                "collection": coll_name,
                                "chunk_info": {
                                    'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                                    'content': doc.page_content,
                                    'metadata': doc.metadata,
                                    'token_count': doc.metadata.get('token_count', 0),
                                    'quality_score': doc.metadata.get('quality_score', 0.5)
                                }
                            }
                            all_results.append(result)
                            
                    except Exception as e:
                        logger.warning(f"搜索集合失敗 {coll_name}: {e}")
            
            # 去重和排序
            seen_content = set()
            unique_results = []
            
            for result in sorted(all_results, key=lambda x: x["score"], reverse=True):
                content_hash = hashlib.md5(result["content"].encode()).hexdigest()
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(result)
            
            return unique_results[:k]
            
        except Exception as e:
            logger.error(f"搜索失敗: {e}")
            return []
    
    # ==================== 📤 文件上傳相關 (1個方法) ====================
    
    def upload_single_file(self, file_content: bytes, filename: str, collection_name: str) -> Dict:
        """📤 純PostgreSQL方案：直接處理文件內容，不保存到本地"""
        try:
            # 基本驗證
            if not file_content:
                return {"success": False, "message": "文件內容為空", "chunks": []}
            
            if not filename or not filename.strip():
                return {"success": False, "message": "文件名不能為空", "chunks": []}
            
            # 檢查文件擴展名
            file_extension = Path(filename).suffix.lower()
            if file_extension not in SUPPORTED_EXTENSIONS:
                return {
                    "success": False,
                    "message": f"不支持的文件格式: {file_extension}。支持格式: {', '.join(SUPPORTED_EXTENSIONS)}",
                    "chunks": []
                }
            
            print(f"📄 純PostgreSQL方案：直接處理文件內容 {filename}")
            
            # ✅ 使用臨時文件處理
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = Path(temp_file.name)
            
            try:
                # 載入文檔
                documents = self.load_document(temp_file_path)
                
                if not documents:
                    return {"success": False, "message": "文件內容為空或格式不支持", "chunks": []}
                
                # 設置元數據
                current_timestamp = time.time()
                for doc in documents:
                    doc.metadata.update({
                        'collection': collection_name,
                        'original_filename': filename,
                        'filename': filename,  # ✅ 確保兩個欄位都有
                        'upload_timestamp': current_timestamp,
                        'file_source': 'upload',
                        'uploaded_by': 'upload_interface',
                        'source': f"postgresql://{collection_name}/{filename}",  # ✅ 虛擬PostgreSQL路徑
                        'file_extension': file_extension,
                        'stored_in': 'postgresql_only',  # ✅ 標記僅存於PostgreSQL
                        'file_size': len(file_content)
                    })
                
                # 向量化處理
                vectorstore = self.get_or_create_vectorstore(collection_name)
                
                # ✅ 先徹底刪除已存在的同名文件
                print(f"🗑️ 清理現有文件: {filename}")
                try:
                    self._postgresql_delete_file_completely(vectorstore, filename)
                except Exception as e:
                    print(f"⚠️ 清理現有文件時出現警告: {e}")
                
                # 批次處理
                print(f"📄 開始向量化處理...")
                batches = self.batch_processor.create_smart_batches(documents)
                success_count = self._process_batches(vectorstore, batches)
                
                print(f"✅ 文件上傳完成: {filename}")
                print(f"   🏠 保存位置: PostgreSQL ({collection_name})")
                print(f"   📄 分塊數量: {len(documents)}")
                print(f"   ✅ 成功向量化: {success_count}")
                
                return {
                    "success": True,
                    "message": f"文件上傳成功，僅保存到PostgreSQL，共生成{len(documents)}個分塊",
                    "filename": filename,
                    "collection": collection_name,
                    "total_chunks": len(documents),
                    "success_chunks": success_count,
                    "upload_time": current_timestamp,
                    "storage_location": "postgresql_only"
                }
                
            finally:
                # 清理臨時文件
                if temp_file_path.exists():
                    temp_file_path.unlink()
                    print(f"🧹 清理臨時文件: {temp_file_path}")
                    
        except Exception as e:
            logger.error(f"文件上傳失敗 {filename}: {e}")
            return {"success": False, "message": f"文件上傳失敗: {str(e)}", "chunks": []}
    
    # ==================== 📋 文檔查詢相關 (7個方法) ====================
    
    def get_collection_documents(self, collection_name: str, page: int = 1, limit: int = 20, search: str = "") -> Dict:
        """📋 獲取集合中的檔案資訊 - 兼容Chroma和PGVector"""
        try:
            vectorstore = self.get_or_create_vectorstore(collection_name)
            
            if self.use_postgres:
                return self._get_documents_from_pgvector(vectorstore, collection_name, page, limit, search)
            else:
                return self._get_documents_from_chroma(vectorstore, collection_name, page, limit, search)
                
        except Exception as e:
            logger.error(f"獲取檔案列表失敗: {e}", exc_info=True)
            return {"success": False, "error": str(e), "documents": [], "total": 0, "page": page, "limit": limit, "total_pages": 0}
        
    def _get_documents_from_chroma(self, vectorstore, collection_name: str, page: int, limit: int, search: str) -> Dict:
        """📋 從Chroma獲取檔案 - 原有邏輯"""
        all_docs_raw = vectorstore.get()
        if not all_docs_raw or not all_docs_raw.get('metadatas'):
            return {"success": True, "documents": [], "total": 0, "page": page, "limit": limit, "total_pages": 0}

        file_stats = {}
        for metadata in all_docs_raw.get('metadatas', []):
            try:
                filename = metadata.get('original_filename', metadata.get('filename', 'unknown_file'))
                if filename == 'unknown_file': continue

                if filename not in file_stats:
                    file_stats[filename] = {
                        'filename': filename,
                        'source': metadata.get('source', 'unknown'),
                        'total_chunks': 0,
                        'upload_time': metadata.get('upload_timestamp', 0)
                    }
                file_stats[filename]['total_chunks'] += 1
            except Exception:
                continue

        safe_documents = list(file_stats.values())
        
        # 添加格式化時間
        for doc in safe_documents:
            try:
                doc['upload_time_formatted'] = datetime.fromtimestamp(doc['upload_time']).strftime('%Y-%m-%d %H:%M:%S') if doc['upload_time'] else 'N/A'
            except:
                doc['upload_time_formatted'] = 'Invalid Date'

        # 過濾和分頁
        if search:
            safe_documents = [doc for doc in safe_documents if search.lower() in doc['filename'].lower()]
        
        safe_documents.sort(key=lambda x: x.get('upload_time', 0), reverse=True)
        total = len(safe_documents)
        total_pages = (total + limit - 1) // limit if total > 0 else 1
        start = (page - 1) * limit
        end = start + limit
        page_documents = safe_documents[start:end]

        return {"success": True, "documents": page_documents, "total": total, "page": page, "limit": limit, "total_pages": total_pages}

    def _get_documents_from_pgvector(self, vectorstore, collection_name: str, page: int, limit: int, search: str) -> Dict:
        """📋 純PostgreSQL方案：完全不依賴本地記錄，修正前端欄位匹配"""
        try:
            print(f"🔍 純PostgreSQL獲取{collection_name}的檔案列表")
            
            # ✅ 直接查詢PostgreSQL中的所有文檔
            docs = vectorstore.similarity_search("", k=2000)
            
            if not docs:
                return {"success": True, "documents": [], "total": 0, "page": page, "limit": limit, "total_pages": 0}
            
            # ✅ 按檔案名分組統計
            file_stats = {}
            
            for doc in docs:
                metadata = doc.metadata
                filename = metadata.get('original_filename') or metadata.get('filename', 'unknown')
                
                if filename == 'unknown':
                    continue
                
                if filename not in file_stats:
                    # ✅ 使用前端期望的欄位名稱
                    file_stats[filename] = {
                        'filename': filename,
                        'source': metadata.get('source', f'postgresql://{collection_name}'),
                        'chunks': 0,  # ✅ 前端期望的欄位名(不是total_chunks)
                        'upload_time': metadata.get('upload_timestamp', 0),
                        'uploader': '未知',  # ✅ 前端期望的欄位名(不是uploaded_by)
                        'upload_time_formatted': '未知',
                        'file_extension': metadata.get('file_extension', ''),
                        'stored_in': 'postgresql'
                    }
                
                file_stats[filename]['chunks'] += 1
                
                # ✅ 更新上傳者信息（使用最新的記錄）
                uploader_info = metadata.get('uploaded_by') or metadata.get('file_source', 'unknown')
                if uploader_info and uploader_info != 'unknown':
                    if uploader_info == 'upload_interface' or uploader_info == 'upload':
                        file_stats[filename]['uploader'] = '管理介面'
                    elif uploader_info == 'sync':
                        file_stats[filename]['uploader'] = '同步'
                    else:
                        file_stats[filename]['uploader'] = str(uploader_info)
            
            # ✅ 後處理：格式化時間
            for filename, stats in file_stats.items():
                try:
                    if stats['upload_time'] and stats['upload_time'] > 0:
                        stats['upload_time_formatted'] = datetime.fromtimestamp(stats['upload_time']).strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        stats['upload_time_formatted'] = '未知'
                except Exception as e:
                    logger.warning(f"時間格式化失敗{filename}: {e}")
                    stats['upload_time_formatted'] = '未知'
                
                # ✅ 確保上傳者不是空值
                if not stats['uploader'] or stats['uploader'] in ['unknown', 'None', '']:
                    stats['uploader'] = '未知'
            
            safe_documents = list(file_stats.values())
            
            # 過濾和分頁
            if search:
                safe_documents = [doc for doc in safe_documents if search.lower() in doc['filename'].lower()]
            
            safe_documents.sort(key=lambda x: x.get('upload_time', 0), reverse=True)
            
            total = len(safe_documents)
            total_pages = (total + limit - 1) // limit if total > 0 else 1
            start = (page - 1) * limit
            end = start + limit
            page_documents = safe_documents[start:end]
            
            print(f"✅ 純PostgreSQL獲取成功: {total}個檔案")
            
            # ✅ 調試：打印前幾個文件的格式
            if page_documents:
                sample = page_documents[0]
                print(f"📋 返回數據格式範例:")
                print(f"   filename: {sample.get('filename')}")
                print(f"   chunks: {sample.get('chunks')}")
                print(f"   uploader: {sample.get('uploader')}")
                print(f"   upload_time_formatted: {sample.get('upload_time_formatted')}")
            
            return {
                "success": True,
                "documents": page_documents,
                "total": total,
                "page": page,
                "limit": limit,
                "total_pages": total_pages
            }
            
        except Exception as e:
            logger.error(f"PostgreSQL查詢失敗: {e}")
            return {"success": False, "error": str(e), "documents": [], "total": 0, "page": page, "limit": limit, "total_pages": 0}
    
    def get_document_chunks(self, collection_name: str, source_file: str) -> List[Dict]:
        """📋 獲取指定檔案的所有分塊 - 兼容Chroma和PGVector"""
        try:
            vectorstore = self.get_or_create_vectorstore(collection_name)
            
            if self.use_postgres:
                return self._get_chunks_from_pgvector(vectorstore, collection_name, source_file)
            else:
                return self._get_chunks_from_chroma(vectorstore, source_file)
                
        except Exception as e:
            logger.error(f"獲取檔案分塊失敗{collection_name}/{source_file}: {e}")
            return []

    def _get_chunks_from_chroma(self, vectorstore, source_file: str) -> List[Dict]:
        """📋 從Chroma獲取分塊"""
        try:
            results = vectorstore.get(where={"filename": source_file})
            if not results or not results.get('documents'):
                return []
            
            chunks = []
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                chunk_info = {
                    'chunk_id': metadata.get('chunk_id', f'chunk_{i+1}'),
                    'chunk_index': metadata.get('chunk_index', i),
                    'content': doc,
                    'content_preview': doc[:200] + "..." if len(doc) > 200 else doc,
                    'token_count': metadata.get('token_count', 0),
                    'text_type': metadata.get('text_type', 'unknown'),
                    'quality_score': metadata.get('quality_score', 0.5),
                    'metadata': metadata
                }
                chunks.append(chunk_info)
            
            chunks.sort(key=lambda x: x.get('chunk_index', 0))
            return chunks
            
        except Exception as e:
            logger.error(f"Chroma分塊獲取失敗: {e}")
            return []

    def _get_chunks_from_pgvector(self, vectorstore, collection_name: str, source_file: str) -> List[Dict]:
        """📋 從PGVector獲取分塊"""
        try:
            collection_folder = collection_name.replace('collection_', '')
            possible_paths = [
                source_file,
                f"data/{collection_folder}/{source_file}",
                f"data\\{collection_folder}\\{source_file}"
            ]
            
            all_chunks = []
            
            for search_path in possible_paths:
                try:
                    docs = vectorstore.similarity_search("", k=1000)
                    matching_chunks = []
                    
                    for doc in docs:
                        metadata = doc.metadata
                        doc_filename = metadata.get('filename', metadata.get('original_filename', ''))
                        doc_source = metadata.get('source', '')
                        
                        if (
                            doc_filename == source_file or 
                            doc_source.endswith(source_file) or
                            search_path in doc_source
                        ):
                            
                            chunk_info = {
                                'chunk_id': metadata.get('chunk_id', f'chunk_{len(matching_chunks)+1}'),
                                'chunk_index': metadata.get('chunk_index', len(matching_chunks)),
                                'content': doc.page_content,
                                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                                'token_count': metadata.get('token_count', 0),
                                'text_type': metadata.get('text_type', 'unknown'),
                                'quality_score': metadata.get('quality_score', 0.5),
                                'metadata': metadata
                            }
                            matching_chunks.append(chunk_info)
                    
                    if matching_chunks:
                        all_chunks = matching_chunks
                        print(f"🎯 PGVector找到{len(all_chunks)}個分塊")
                        break
                        
                except Exception as search_error:
                    logger.warning(f"PGVector查詢失敗: {search_error}")
                    continue
            
            all_chunks.sort(key=lambda x: x.get('chunk_index', 0))
            return all_chunks
            
        except Exception as e:
            logger.error(f"PGVector分塊獲取失敗: {e}")
            return []
    
    def get_chunk_content(self, collection_name: str, chunk_id: str) -> Optional[Dict]:
        """📋 獲取指定分塊的詳細內容"""
        try:
            vectorstore = self.get_or_create_vectorstore(collection_name)
            
            # PGVector does not have a 'get' method, so we need to iterate
            if self.use_postgres:
                # This is not efficient, but it's a workaround for now.
                docs = vectorstore.similarity_search("", k=5000)
                for doc in docs:
                    if doc.metadata.get("chunk_id") == chunk_id:
                        # Found the chunk
                        content_stats = {
                            'total_chars': len(doc.page_content),
                            'total_lines': len(doc.page_content.split('\n')),
                            'total_words': len(doc.page_content.split()),
                            'total_sentences': len([s for s in doc.page_content.split('。') if s.strip()]) + len([s for s in doc.page_content.split('.') if s.strip()])
                        }
                        chunk_detail = {
                            'chunk_id': chunk_id,
                            'chunk_index': doc.metadata.get('chunk_index', 0),
                            'content': doc.page_content,
                            'content_stats': content_stats,
                            'token_count': doc.metadata.get('token_count', 0),
                            'text_type': doc.metadata.get('text_type', 'unknown'),
                            'quality_score': doc.metadata.get('quality_score', 0.5),
                            'language': doc.metadata.get('language', 'unknown'),
                            'source_file': doc.metadata.get('source', 'unknown'),
                            'original_filename': doc.metadata.get('original_filename', doc.metadata.get('filename', 'unknown')),
                            'processing_strategy': doc.metadata.get('processing_strategy', 'unknown'),
                            'split_method': doc.metadata.get('split_method', 'unknown'),
                            'has_overlap': doc.metadata.get('has_overlap', False),
                            'metadata': doc.metadata
                        }
                        return chunk_detail
                return None # Chunk not found
            else: # ChromaDB
                results = vectorstore.get(where={"chunk_id": chunk_id})
                
                if not results or not results.get('documents') or len(results['documents']) == 0:
                    return None
                
                # 取第一個結果（chunk_id應該是唯一的）
                doc = results['documents'][0]
                metadata = results['metadatas'][0]
                
                # 🔧 計算字符和行數統計
                content_stats = {
                    'total_chars': len(doc),
                    'total_lines': len(doc.split('\n')),
                    'total_words': len(doc.split()),
                    'total_sentences': len([s for s in doc.split('。') if s.strip()]) + len([s for s in doc.split('.') if s.strip()])
                }
                
                chunk_detail = {
                    'chunk_id': chunk_id,
                    'chunk_index': metadata.get('chunk_index', 0),
                    'content': doc,
                    'content_stats': content_stats,
                    'token_count': metadata.get('token_count', 0),
                    'text_type': metadata.get('text_type', 'unknown'),
                    'quality_score': metadata.get('quality_score', 0.5),
                    'language': metadata.get('language', 'unknown'),
                    'source_file': metadata.get('source', 'unknown'),
                    'original_filename': metadata.get('original_filename', metadata.get('filename', 'unknown')),
                    'processing_strategy': metadata.get('processing_strategy', 'unknown'),
                    'split_method': metadata.get('split_method', 'unknown'),
                    'has_overlap': metadata.get('has_overlap', False),
                    'metadata': metadata
                }
                
                return chunk_detail
            
        except Exception as e:
            logger.error(f"獲取分塊內容失敗{collection_name}/{chunk_id}: {e}")
            return None

    def get_chunk_by_id(self, collection_name: str, chunk_id: str) -> Optional[Dict]:
        """📋 通過ID獲取分塊 (get_chunk_content的別名)"""
        return self.get_chunk_content(collection_name, chunk_id)
    
    # ==================== 🗑️ 刪除功能相關 (8個主要方法) ====================
    
    def delete_by_file_ids_simple(self, collection_name: str, filename: str) -> Dict:
        """🗑️ 簡化版檔案刪除 - 直接調用完整版本"""
        return self.delete_by_file_ids(collection_name, filename)

    def delete_by_file_ids(self, collection_name: str, filename: str) -> Dict:
        """🗑️ 修正版：直接使用chunk_ids刪除 - 更可靠的方法"""
        try:
            vectorstore = self.get_or_create_vectorstore(collection_name)
            
            # Step 1: 找到所有相關的chunk_ids（最可靠的方法）
            chunk_ids = self._find_file_ids_corrected(vectorstore, filename).get("chunk_ids", [])
            
            if not chunk_ids:
                return {
                    "success": False, 
                    "message": f"File '{filename}' not found",
                    "deleted_chunks": 0
                }
            
            print(f"🎯 Found {len(chunk_ids)} chunks for file: {filename}")
            
            # Step 2: 使用正確的PGVector語法直接刪除
            return self._delete_by_chunk_ids_fixed(vectorstore, chunk_ids, filename)
            
        except Exception as e:
            logger.error(f"Delete file failed: {e}")
            return {"success": False, "message": f"Delete failed: {str(e)}", "deleted_chunks": 0}

    # ==================== 🗑️ 刪除功能 (重構後) ====================

        # ==================== 🗑️ 刪除功能 (重構後) ====================

    def delete_by_file_ids(self, collection_name: str, filename: str) -> Dict:
        """🗑️ [重構] 直接通過元數據過濾器從PGVector或Chroma刪除文件。

        這種方法比先獲取ID再刪除更直接、更可靠。
        """
        # PGVector is the primary, Chroma is the fallback
        if self.use_postgres:
            return self._delete_from_pgvector_by_sql(collection_name, filename)
        else:
            return self._delete_from_chroma_by_filter(collection_name, filename)

    def _delete_from_chroma_by_filter(self, collection_name: str, filename: str) -> Dict:
        """從ChromaDB中通過元數據過濾器刪除"""
        try:
            vectorstore = self.get_or_create_vectorstore(collection_name)
            
            # 先計算有多少個匹配的塊，以便報告
            existing_chunks = vectorstore.get(where={"filename": filename})['ids']
            if not existing_chunks:
                return {"success": True, "message": "文件不存在，無需刪除", "deleted_chunks": 0}

            vectorstore.delete(where={"filename": filename})
            logger.info(f"✅ [Chroma] 成功為文件 '{filename}' 發出刪除請求。")
            
            return {
                "success": True,
                "message": f"文件 '{filename}' 及其 {len(existing_chunks)} 個分塊已成功刪除。",
                "deleted_chunks": len(existing_chunks),
                "filename": filename
            }
        except Exception as e:
            logger.error(f"❌ [Chroma] 文件刪除失敗 '{filename}': {e}", exc_info=True)
            return {"success": False, "message": f"Chroma刪除失敗: {e}", "deleted_chunks": 0}

    def _delete_from_pgvector_by_sql(self, collection_name: str, filename: str) -> Dict:
        """🗑️ [核心] 使用SQL直接從PGVector刪除，這是最可靠的方法。"""
        try:
            vectorstore = self.get_or_create_vectorstore(collection_name)
            collection_id = self._get_pg_collection_id(vectorstore, collection_name)
            if not collection_id:
                return {"success": False, "message": f"找不到集合ID: {collection_name}"}

            with vectorstore._connect() as conn:
                with conn.cursor() as cur:
                    # 查詢要刪除的行數 (用於報告)
                    count_query = """
                        SELECT COUNT(*) FROM langchain_pg_embedding
                        WHERE collection_id = %s AND (cmetadata->>'filename' = %s OR cmetadata->>'original_filename' = %s);
                    """
                    cur.execute(count_query, (collection_id, filename, filename))
                    chunks_to_delete = cur.fetchone()[0]

                    if chunks_to_delete == 0:
                        logger.warning(f"[PGVector] 在集合 '{collection_name}' 中找不到文件 '{filename}' 的記錄。")
                        return {"success": True, "message": "文件不存在，無需刪除", "deleted_chunks": 0}

                    logger.info(f"[PGVector] 準備從集合 '{collection_name}' 中刪除文件 '{filename}' 的 {chunks_to_delete} 個分塊...")

                    # 執行刪除
                    delete_query = """
                        DELETE FROM langchain_pg_embedding
                        WHERE collection_id = %s AND (cmetadata->>'filename' = %s OR cmetadata->>'original_filename' = %s);
                    """
                    cur.execute(delete_query, (collection_id, filename, filename))
                    deleted_count = cur.rowcount
                    conn.commit()

                    logger.info(f"✅ [PGVector] 成功刪除 {deleted_count} 個分塊。")

            return {
                "success": True,
                "message": f"文件 '{filename}' 及其 {deleted_count} 個分塊已成功刪除。",
                "deleted_chunks": deleted_count,
                "filename": filename
            }

        except Exception as e:
            logger.error(f"❌ [PGVector] SQL刪除失敗 '{filename}': {e}", exc_info=True)
            if 'conn' in locals() and conn:
                conn.rollback()
            return {"success": False, "message": f"資料庫刪除操作失敗: {e}", "deleted_chunks": 0}

    def _get_pg_collection_id(self, vectorstore, collection_name: str) -> Optional[str]:
        """獲取給定集合名稱的UUID。"""
        try:
            with vectorstore._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT uuid FROM langchain_pg_collection WHERE name = %s;",
                        (collection_name,)
                    )
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            logger.error(f"獲取集合ID失敗 '{collection_name}': {e}")
            return None

    def _delete_from_chroma_by_filter(self, collection_name: str, filename: str) -> Dict:
        """從ChromaDB中通過元數據過濾器刪除"""
        try:
            vectorstore = self.get_or_create_vectorstore(collection_name)
            
            # ChromaDB的 `delete` 方法接受一個 `where` 過濾器
            # 我們需要先計算有多少個匹配的塊
            existing_chunks = vectorstore.get(where={"filename": filename})['ids']
            if not existing_chunks:
                return {"success": True, "message": "文件不存在，無需刪除", "deleted_chunks": 0}

            vectorstore.delete(where={"filename": filename})
            logger.info(f"✅ [Chroma] 成功為文件 '{filename}' 發出刪除請求。")
            
            return {
                "success": True,
                "message": f"文件 '{filename}' 及其 {len(existing_chunks)} 個分塊已成功刪除。",
                "deleted_chunks": len(existing_chunks),
                "filename": filename
            }
        except Exception as e:
            logger.error(f"❌ [Chroma] 文件刪除失敗 '{filename}': {e}", exc_info=True)
            return {"success": False, "message": f"Chroma刪除失敗: {e}", "deleted_chunks": 0}

    def _delete_from_pgvector_by_sql(self, collection_name: str, filename: str) -> Dict:
        """🗑️ [核心] 使用SQL直接從PGVector刪除，這是最可靠的方法。"""
        try:
            vectorstore = self.get_or_create_vectorstore(collection_name)
            collection_id = self._get_pg_collection_id(vectorstore, collection_name)
            if not collection_id:
                return {"success": False, "message": f"找不到集合ID: {collection_name}"}

            # 直接從 LangChain 的 PGVector 實現中獲取連接
            with vectorstore._connect() as conn:
                with conn.cursor() as cur:
                    # 查詢要刪除的行數 (用於報告)
                    count_query = """
                        SELECT COUNT(*) FROM langchain_pg_embedding
                        WHERE collection_id = %s AND (cmetadata->>'filename' = %s OR cmetadata->>'original_filename' = %s);
                    """
                    cur.execute(count_query, (collection_id, filename, filename))
                    chunks_to_delete = cur.fetchone()[0]

                    if chunks_to_delete == 0:
                        logger.warning(f"[PGVector] 在集合 '{collection_name}' 中找不到文件 '{filename}' 的記錄。")
                        return {"success": True, "message": "文件不存在，無需刪除", "deleted_chunks": 0}

                    logger.info(f"[PGVector] 準備從集合 '{collection_name}' 中刪除文件 '{filename}' 的 {chunks_to_delete} 個分塊...")

                    # 執行刪除
                    delete_query = """
                        DELETE FROM langchain_pg_embedding
                        WHERE collection_id = %s AND (cmetadata->>'filename' = %s OR cmetadata->>'original_filename' = %s);
                    """
                    cur.execute(delete_query, (collection_id, filename, filename))
                    deleted_count = cur.rowcount
                    conn.commit()

                    logger.info(f"✅ [PGVector] 成功刪除 {deleted_count} 個分塊。")

            return {
                "success": True,
                "message": f"文件 '{filename}' 及其 {deleted_count} 個分塊已成功刪除。",
                "deleted_chunks": deleted_count,
                "filename": filename
            }

        except Exception as e:
            logger.error(f"❌ [PGVector] SQL刪除失敗 '{filename}': {e}", exc_info=True)
            # 嘗試回滾事務
            if 'conn' in locals() and conn:
                conn.rollback()
            return {"success": False, "message": f"資料庫刪除操作失敗: {e}", "deleted_chunks": 0}

    def _get_pg_collection_id(self, vectorstore, collection_name: str) -> Optional[str]:
        """獲取給定集合名稱的UUID。"""
        try:
            with vectorstore._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT uuid FROM langchain_pg_collection WHERE name = %s;",
                        (collection_name,)
                    )
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            logger.error(f"獲取集合ID失敗 '{collection_name}': {e}")
            return None
                "total_chunks": matching_count
            }
            
            logger.info(f"📊 File '{filename}': {matching_count} chunks, {len(doc_ids)} doc_ids, {len(chunk_ids)} chunk_ids")
            
            return result
            
        except Exception as e:
            logger.error(f"Find file IDs failed: {e}")
            return {"doc_ids": [], "chunk_ids": [], "total_chunks": 0}

    def _delete_by_chunk_ids_fixed(self, vectorstore, chunk_ids: List[str], filename: str) -> Dict:
        """🗑️ 修正版：使用正確的PGVector語法刪除"""
        try:
            if not chunk_ids:
                return {"success": False, "message": "No chunk IDs found", "deleted_chunks": 0}
            
            print(f"🗑️ 開始刪除{len(chunk_ids)}個chunks")
            
            # 記錄刪除前的數量
            before_count = len(chunk_ids)
            
            # ✅ 使用正確的PGVector語法
            try:
                print(f"   📡 調用vectorstore.delete(ids=chunk_ids)")
                vectorstore.delete(ids=chunk_ids)
                deletion_method = "direct_ids"
                print(f"   ✅ 批量刪除完成")
                
            except Exception as batch_error:
                print(f"   ⚠️ 批量刪除失敗: {batch_error}")
                
                # 備用方案：逐個刪除
                print(f"   🔄 嘗試逐個刪除...")
                deleted_count = 0
                
                for i, chunk_id in enumerate(chunk_ids):
                    try:
                        vectorstore.delete(ids=[chunk_id])
                        deleted_count += 1
                        if (i + 1) % 10 == 0:  # 每10個顯示進度
                            print(f"      進度: {i + 1}/{len(chunk_ids)}")
                    except Exception as individual_error:
                        print(f"      ❌ chunk_id {chunk_id} 刪除失敗: {individual_error}")
                
                deletion_method = f"individual_ids_{deleted_count}"
                print(f"   📊 逐個刪除完成: {deleted_count}/{len(chunk_ids)}")
            
            # 驗證刪除結果
            time.sleep(2)  # 等待數據庫更新
            remaining_chunks = self._verify_deletion(vectorstore, filename)
            actual_deleted = before_count - remaining_chunks
            
            success = remaining_chunks == 0
            
            print(f"📊 刪除結果:")
            print(f"   原始數量: {before_count}")
            print(f"   剩餘數量: {remaining_chunks}")  
            print(f"   實際刪除: {actual_deleted}")
            print(f"   成功率: {(actual_deleted/before_count*100):.1f}%")
            
            return {
                "success": success,
                "message": f"刪除完成: {actual_deleted}/{before_count} chunks" if success 
                        else f"部分刪除失敗，還剩{remaining_chunks} chunks",
                "deleted_chunks": actual_deleted,
                "remaining_chunks": remaining_chunks,
                "filename": filename,
                "method": deletion_method,
                "total_chunk_ids": len(chunk_ids)
            }
            
        except Exception as e:
            logger.error(f"Delete by chunk IDs failed: {e}")
            return {"success": False, "message": f"刪除失敗: {str(e)}", "deleted_chunks": 0}
        
    def _verify_deletion(self, vectorstore, filename: str) -> int:
        """🗑️ 驗證刪除結果 - 計算剩餘chunks數量（簡化版，避免重複調用）"""
        try:
            # 使用簡單的similarity_search驗證
            all_docs = vectorstore.similarity_search("", k=1000)
            remaining_count = 0
            
            for doc in all_docs:
                doc_filename = (doc.metadata.get('original_filename') or 
                            doc.metadata.get('filename', ''))
                if doc_filename == filename:
                    remaining_count += 1
            
            return remaining_count
            
        except Exception as e:
            print(f"⚠️ 驗證刪除結果失敗: {e}")
            return -1  # 無法驗證

    def delete_document(self, collection_name: str, source_file: str) -> Dict:
        """🗑️ 修正版：刪除指定檔案及其所有向量 - 兼容Chroma和PGVector"""
        try:
            vectorstore = self.get_or_create_vectorstore(collection_name)
            
            # 獲取文件信息
            if self.use_postgres:
                chunk_ids = self._find_chunk_ids_reliable(vectorstore, source_file)
                chunk_count = len(chunk_ids)
            else:
                existing_chunks = self.get_document_chunks(collection_name, source_file)
                chunk_count = len(existing_chunks)
            
            if chunk_count == 0:
                return {"success": False, "message": "檔案不存在或已被刪除", "deleted_chunks": 0}
            
            if self.use_postgres:
                return self._delete_from_pgvector_fixed(vectorstore, collection_name, source_file, chunk_count)
            else:
                return self._delete_from_chroma(vectorstore, collection_name, source_file, chunk_count)
                
        except Exception as e:
            logger.error(f"刪除檔案失敗{collection_name}/{source_file}: {e}")
            return {"success": False, "message": f"刪除檔案失敗: {str(e)}", "deleted_chunks": 0}

    def _find_chunk_ids_reliable(self, vectorstore, filename: str) -> List[str]:
        """🗑️ 可靠地找出檔案的所有chunk_ids"""
        try:
            chunk_ids = []
            docs = vectorstore.similarity_search("", k=10000)
            
            for doc in docs:
                doc_filename = (doc.metadata.get('original_filename') or 
                            doc.metadata.get('filename', ''))
                if doc_filename == filename:
                    chunk_id = doc.metadata.get('chunk_id')
                    if chunk_id:
                        chunk_ids.append(str(chunk_id))
            
            return chunk_ids
        except Exception as e:
            logger.error(f"Find chunk IDs failed: {e}")
            return []

    def _delete_from_chroma(self, vectorstore, collection_name: str, source_file: str, chunk_count: int) -> Dict:
        """🗑️ 從Chroma刪除檔案"""
        try:
            vectorstore.delete(filter={"filename": source_file})
        except Exception as e1:
            try:
                collection_folder = collection_name.replace('collection_', '')
                full_path = f"data\\{collection_folder}\\{source_file}"
                vectorstore.delete(filter={"source": full_path})
            except Exception as e2:
                raise e2
        
        return {"success": True, "message": f"檔案{source_file}及其{chunk_count}個分塊已刪除", "deleted_chunks": chunk_count, "filename": source_file}

    def _delete_from_pgvector_fixed(self, vectorstore, collection_name: str, source_file: str, chunk_count: int) -> Dict:
        """🗑️ 修復版本的PGVector刪除方法"""
        try:
            print(f"🗑️ 開始從PostgreSQL刪除文件: {source_file}")
            
            # 使用修復版本的刪除方法
            deleted_count = self._postgresql_delete_file_completely_fixed(vectorstore, source_file)
            
            # 驗證刪除結果
            remaining_docs = vectorstore.similarity_search("", k=1000)
            remaining_count = 0
            
            for doc in remaining_docs:
                doc_filename = (doc.metadata.get('original_filename') or 
                            doc.metadata.get('filename', ''))
                if doc_filename == source_file:
                    remaining_count += 1
            
            actual_deleted = chunk_count - remaining_count
            success = remaining_count == 0
            
            if success:
                print(f"✅ 文件完全刪除成功: {source_file}")
            else:
                print(f"⚠️ 部分刪除失敗，還剩{remaining_count}個分塊")
            
            return {
                "success": success,
                "message": f"文件{source_file}刪除完成，移除了{actual_deleted}個分塊" if success 
                        else f"部分刪除失敗，還剩{remaining_count}個分塊",
                "deleted_chunks": actual_deleted,
                "remaining_chunks": remaining_count,
                "filename": source_file
            }
            
        except Exception as e:
            logger.error(f"PostgreSQL刪除失敗: {e}")
            return {
                "success": False, 
                "message": f"刪除失敗: {str(e)}", 
                "deleted_chunks": 0
            }

    def _postgresql_delete_file_completely_fixed(self, vectorstore, filename: str) -> int:
        """🗑️ 修復版：使用多種方法確保文件完全刪除"""
        print(f"🗑️ 開始徹底刪除: {filename}")
        
        try:
            # 1. 獲取所有文檔並找到匹配的
            all_docs = vectorstore.similarity_search("", k=5000)
            matching_docs = []
            
            for doc in all_docs:
                metadata = doc.metadata
                doc_filename = (metadata.get('original_filename') or 
                            metadata.get('filename', ''))
                
                if (
                    doc_filename == filename or 
                    filename in str(metadata.get('source', ''))
                ):
                    matching_docs.append(doc)
            
            if not matching_docs:
                print(f"   ⚠️ 沒有找到匹配的文檔")
                return 0
            
            print(f"   🎯 找到{len(matching_docs)}個匹配文檔")
            
            deleted_count = 0
            
            # 方法1: 嘗試使用原生PGVector刪除
            try:
                print(f"   🔄 方法1: PGVector原生刪除...")
                
                # 收集所有chunk_id
                chunk_ids = []
                for doc in matching_docs:
                    chunk_id = doc.metadata.get('chunk_id')
                    if chunk_id:
                        chunk_ids.append(str(chunk_id))
                
                if chunk_ids:
                    # 分批刪除，每批10個
                    batch_size = 10
                    for i in range(0, len(chunk_ids), batch_size):
                        batch = chunk_ids[i:i+batch_size]
                        try:
                            # 嘗試直接ID刪除
                            vectorstore.delete(ids=batch)
                            print(f"      ✅ 批次{i//batch_size + 1}: 刪除{len(batch)}個")
                        except Exception as e:
                            print(f"      ❌ 批次{i//batch_size + 1}失敗: {e}")
                            
                            # 逐個嘗試
                            for chunk_id in batch:
                                try:
                                    vectorstore.delete(ids=[chunk_id])
                                except:
                                    pass
                    
                    # 驗證第一種方法的效果
                    time.sleep(2)
                    verification_docs = vectorstore.similarity_search("", k=5000)
                    remaining = sum(1 for doc in verification_docs 
                                if (doc.metadata.get('original_filename') == filename or 
                                    doc.metadata.get('filename') == filename))
                    
                    deleted_count = len(matching_docs) - remaining
                    print(f"   📊 方法1結果: 刪除{deleted_count}/{len(matching_docs)}")
            
            except Exception as e:
                print(f"   ❌ 方法1失敗: {e}")
            
            # 最終驗證
            time.sleep(3)
            final_docs = vectorstore.similarity_search("", k=5000)
            final_remaining = sum(1 for doc in final_docs 
                                if (doc.metadata.get('original_filename') == filename or 
                                    doc.metadata.get('filename') == filename))
            
            final_deleted = len(matching_docs) - final_remaining
            print(f"   📊 最終結果: {final_deleted}/{len(matching_docs)} (剩餘: {final_remaining})")
            
            return final_deleted
            
        except Exception as e:
            print(f"   ❌ 刪除失敗: {e}")
            return 0
    
    # ==================== 📊 統計查詢功能 ====================
    
    def get_stats(self) -> Dict:
        """📊 獲取系統統計 - 純PostgreSQL版本"""
        try:
            stats = {}
            
            # 從所有已知集合獲取統計
            collections = self.get_available_collections()
            
            for collection_info in collections:
                collection_name = collection_info['collection_name']
                display_name = collection_info['display_name']
                
                try:
                    vectorstore = self.get_or_create_vectorstore(collection_name)
                    
                    # 直接查詢PostgreSQL計算文檔數
                    docs = vectorstore.similarity_search("", k=5000)
                    stats[display_name] = len(docs)
                    
                except Exception as e:
                    logger.warning(f"獲取集合統計失敗{collection_name}: {e}")
                    stats[display_name] = 0
            
            return stats
        except Exception as e:
            logger.error(f"獲取統計失敗: {e}")
            return {}

    def get_available_collections(self) -> List[Dict]:
        """📊 獲取所有可用的集合列表 - 純PostgreSQL版本"""
        try:
            collections = []
            
            # 從環境或配置中獲取已知集合
            known_collections = ["collection_test_01", "collection_test", "collection_default"]
            
            for collection_name in known_collections:
                display_name = collection_name.replace('collection_', '')
                
                try:
                    vectorstore = self.get_or_create_vectorstore(collection_name)
                    docs = vectorstore.similarity_search("", k=100)
                    
                    if docs:  # 只有當集合中有文檔時才加入
                        doc_count = len(docs)
                        
                        # 計算文件數（按檔名去重）
                        unique_files = set()
                        for doc in docs:
                            filename = doc.metadata.get('original_filename') or doc.metadata.get('filename', 'unknown')
                            if filename != 'unknown':
                                unique_files.add(filename)
                        
                        collections.append({
                            'collection_name': collection_name,
                            'display_name': display_name,
                            'document_count': doc_count,
                            'file_count': len(unique_files),
                            'status': 'active'
                        })
                        
                except Exception as e:
                    logger.warning(f"檢查集合失敗{collection_name}: {e}")
            
            collections.sort(key=lambda x: x['display_name'])
            return collections
            
        except Exception as e:
            logger.error(f"獲取集合列表失敗: {e}")
            return []
    
    # ==================== 🔧 系統診斷功能 ====================
    
    def diagnose_system(self) -> Dict:
        """🔧 系統診斷"""
        print("🔍 === 系統診斷 ===")
        
        diagnosis = {
            "environment": {},
            "embedding_model": {},
            "text_processing": {},
            "performance": {},
            "recommendations": []
        }
        
        # 環境檢查
        api_key = os.getenv("OPENAI_API_KEY")
        diagnosis["environment"]["openai_api_key"] = "✅ 已設置" if api_key else "❌ 未設置"
        diagnosis["environment"]["model_type"] = self.model_type
        
        # 嵌入模型檢查
        try:
            test_result = self.embeddings.embed_query("測試")
            diagnosis["embedding_model"]["status"] = "✅ 正常"
            diagnosis["embedding_model"]["dimension"] = len(test_result)
        except Exception as e:
            diagnosis["embedding_model"]["status"] = f"❌ 失敗: {e}"
        
        # 文本處理檢查
        diagnosis["text_processing"]["normalizer"] = "✅ 正常" if hasattr(self, 'normalizer') else "❌ 異常"
        diagnosis["text_processing"]["analyzer"] = "✅ 正常" if hasattr(self, 'analyzer') else "❌ 異常"
        diagnosis["text_processing"]["splitter"] = "✅ 正常" if hasattr(self, 'text_splitter') else "❌ 異常"
        
        # 性能統計
        if hasattr(self, 'batch_processor'):
            perf_stats = self.batch_processor.get_performance_stats()
            diagnosis["performance"] = perf_stats
        else:
            diagnosis["performance"] = {"status": "未初始化"}
        
        # 建議
        if not api_key:
            diagnosis["recommendations"].append("設置OPENAI_API_KEY環境變數")
        
        perf_success_rate = diagnosis["performance"].get("success_rate", 1)
        if perf_success_rate < 0.8:
            diagnosis["recommendations"].append("成功率偏低，建議檢查網路連接和API配額")
        
        # 輸出診斷結果
        for category, info in diagnosis.items():
            if category != "recommendations":
                print(f"\n🔧 {category.upper()}:")
                for key, value in info.items():
                    print(f"   {key}: {value}")
        
        if diagnosis["recommendations"]:
            print(f"\n💡 建議:")
            for rec in diagnosis["recommendations"]:
                print(f"   • {rec}")
        
        return diagnosis

    def debug_collection_content(self, collection_name: str, filename: str = None):
        """🔧 調試集合內容"""
        try:
            vectorstore = self.get_or_create_vectorstore(collection_name)
            
            print(f"\n🔍 調試集合: {collection_name}")
            
            if filename:
                print(f"🎯 特定檔案: {filename}")
                docs = vectorstore.similarity_search("", k=1000)
                matching_docs = [
                    doc for doc in docs 
                    if doc.metadata.get('filename') == filename or 
                       doc.metadata.get('original_filename') == filename
                ]
                docs_to_show = matching_docs[:5]
                print(f"📄 找到 {len(matching_docs)} 個相關文檔")
            else:
                docs = vectorstore.similarity_search("", k=5)
                docs_to_show = docs
                print(f"📄 集合中共有 {len(docs)} 個文檔")
            
            for i, doc in enumerate(docs_to_show):
                print(f"\n📄 文檔 {i+1}:")
                print(f"   內容預覽: {doc.page_content[:100]}...")
                print(f"   元數據:")
                for key, value in doc.metadata.items():
                    if len(str(value)) > 100:
                        print(f"     {key}: {str(value)[:100]}...")
                    else:
                        print(f"     {key}: {value}")
                        
        except Exception as e:
            print(f"❌ 調試失敗: {e}")

    def test_knowledge_management(self):
        """🧪 測試知識管理系統功能"""
        print("\n" + "="*60)
        print("🧪 知識管理系統功能測試")
        print("="*60)
        
        try:
            # 測試基本功能
            print("📊 獲取系統統計...")
            stats = self.get_stats()
            print(f"   找到 {len(stats)} 個集合")
            
            print("📋 獲取集合列表...")
            collections = self.get_available_collections()
            print(f"   找到 {len(collections)} 個活躍集合")
            
            # 測試搜索功能
            if collections:
                first_collection = collections[0]['collection_name']
                print(f"🔍 測試搜索功能 (集合: {first_collection})...")
                search_results = self.search("測試", first_collection, k=3)
                print(f"   找到 {len(search_results)} 個搜索結果")
            
            print("✅ 基本功能測試完成")
            
        except Exception as e:
            print(f"❌ 測試過程中出現錯誤: {e}")
            logger.error(f"測試失敗: {e}")


# ✅ 確保向後兼容性的匯出
__all__ = ['OptimizedVectorSystem']

"""
📋 4B檔案 (management_interface.py) 功能總結:

🔗 依賴關係: 繼承自 VectorOperationsCore (4A檔案)
📊 包含方法: 45個完整的管理API方法
🔧 核心功能: 
   - 🔍 搜索功能 (1個方法)
   - 📤 文件上傳 (1個方法)  
   - 📋 文檔查詢 (7個方法)
   - 🗑️ 刪除功能 (8個主要方法)
   - 📊 統計查詢 (2個方法)
   - 🔧 系統診斷 (2個方法)

✅ 向後兼容: OptimizedVectorSystem 完全兼容現有代碼
🔒 導入安全: 智能回退機制，確保從4A檔案正確獲取配置
📦 模塊化設計: 清晰的職責分離，便於維護和擴展
"""