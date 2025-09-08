#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conversation_logger_simple.py - PostgreSQL版對話記錄器（修正版）
使用database_adapter.py抽象層，支持SQLite和PostgreSQL
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import pytz

# 導入數據庫抽象層
from database_adapter import DatabaseFactory, SQLDialect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgreSQLConversationLogger:
    """PostgreSQL版對話記錄器 - 修正版，使用數據庫抽象層"""
    
    def __init__(self, db_config: Dict = None, db_path: str = None):
        """
        初始化對話記錄器 - 向後兼容版本
        
        Args:
            db_config: 數據庫配置字典（新格式）
            db_path: SQLite 數據庫路徑（舊格式，向後兼容）
        """
        # 向後兼容處理：如果傳入 db_path，轉換為 SQLite 配置
        if db_path is not None and db_config is None:
            db_config = {
                "type": "sqlite",
                "db_file": db_path,
                "timeout": 30.0,
                "journal_mode": "WAL", 
                "synchronous": "NORMAL",
                "cache_size": 2000
            }
            print(f"📝 向後兼容模式：使用 SQLite 數據庫 {db_path}")
        
        if db_config is None:
            # 從環境變量自動創建適配器
            self.db_adapter = DatabaseFactory.create_from_env("conversations")
        else:
            # 使用提供的配置
            db_type = db_config.get("type", "sqlite")
            self.db_adapter = DatabaseFactory.create_adapter(db_type, db_config)
        
        # 修正：更可靠的數據庫類型檢測
        self.db_type = self._detect_database_type()
        
        self.init_database()
    
    def _detect_database_type(self) -> str:
        """檢測數據庫類型"""
        # 方法1: 從配置中檢測
        if hasattr(self.db_adapter, 'config'):
            config_type = self.db_adapter.config.get("type")
            if config_type:
                return config_type.lower()
        
        # 方法2: 從適配器類型檢測
        adapter_class = self.db_adapter.__class__.__name__
        if "PostgreSQL" in adapter_class:
            return "postgresql"
        elif "SQLite" in adapter_class:
            return "sqlite"
        
        # 方法3: 從屬性檢測
        if hasattr(self.db_adapter, 'host'):
            return "postgresql"
        elif hasattr(self.db_adapter, 'db_file'):
            return "sqlite"
        
        # 默認值
        logger.warning("無法檢測數據庫類型，使用默認值: postgresql")
        return "postgresql"
    
    def _get_placeholder(self, count: int = 1) -> str:
        """獲取正確的參數佔位符"""
        if self.db_type == "sqlite":
            return "?" if count == 1 else ", ".join(["?"] * count)
        else:  # postgresql
            if count == 1:
                return "%s"
            else:
                return ", ".join(["%s"] * count)
    
    def _adapt_sql(self, sql: str) -> str:
        """適配SQL語句到特定數據庫"""
        if self.db_type == "sqlite":
            # PostgreSQL語法轉SQLite
            sql = sql.replace("%s", "?")
            sql = sql.replace("CURRENT_DATE", "DATE('now')")
            sql = sql.replace("CURRENT_TIMESTAMP", "DATETIME('now')")
            sql = sql.replace("NOW()", "DATETIME('now')")
        return sql
    
    def init_database(self):
        """初始化對話數據庫 - 修正版，自動檢測和新增缺失的欄位"""
        try:
            self.db_adapter.connect()
            
            # 1. 創建主對話記錄表 (如果不存在)
            auto_increment = SQLDialect.get_auto_increment_column(self.db_type)
            timestamp_col = SQLDialect.get_timestamp_column(self.db_type)
            
            create_table_sql = f'''
                CREATE TABLE IF NOT EXISTS conversations (
                    id {auto_increment},
                    user_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    timestamp {timestamp_col}
                )
            '''
            
            self.db_adapter.execute_update(create_table_sql)

            # 2. 定義所有應有的欄位及其類型 - 包含 bot_name
            required_columns = {
                "collection_used": "TEXT",
                "retrieved_docs": "TEXT",
                "doc_similarities": "TEXT",
                "chunk_references": "TEXT",
                "processing_time_ms": "INTEGER DEFAULT 0",
                "is_image_generation": SQLDialect.get_boolean_column(self.db_type, False),
                "image_url": "TEXT",
                "error_occurred": SQLDialect.get_boolean_column(self.db_type, False),
                "error_message": "TEXT",
                "authenticated_user_id": "INTEGER",
                "user_role": "TEXT DEFAULT 'anonymous'",
                "created_at": timestamp_col,
                "bot_name": "TEXT"  # 新增 bot_name 字段
            }

            # 3. 獲取當前表格的所有欄位
            existing_columns = self.db_adapter.get_table_columns("conversations")

            # 4. 比較並新增所有缺失的欄位
            for col_name, col_type in required_columns.items():
                if col_name not in existing_columns:
                    print(f"⚠️ 檢測到缺失的資料庫欄位，正在自動新增: {col_name}")
                    alter_sql = f'ALTER TABLE conversations ADD COLUMN {col_name} {col_type}'
                    self.db_adapter.execute_update(alter_sql)

            # 5. 創建統計表 (如果不存在)
            stats_table_sql = f'''
                CREATE TABLE IF NOT EXISTS conversation_stats (
                    id {auto_increment},
                    stat_date DATE NOT NULL,
                    total_conversations INTEGER DEFAULT 0,
                    successful_conversations INTEGER DEFAULT 0,
                    failed_conversations INTEGER DEFAULT 0,
                    image_generations INTEGER DEFAULT 0,
                    average_processing_time_ms REAL DEFAULT 0,
                    most_used_collection TEXT,
                    created_at {timestamp_col}
                )
            '''
            
            self.db_adapter.execute_update(stats_table_sql)
            
            # 6. 創建唯一約束 (PostgreSQL語法)
            if self.db_type == "postgresql":
                try:
                    self.db_adapter.execute_update(
                        'ALTER TABLE conversation_stats ADD CONSTRAINT unique_stat_date UNIQUE (stat_date)'
                    )
                except Exception:
                    # 約束可能已存在，忽略錯誤
                    pass
            
            # 7. 創建索引以提高查詢性能
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_user_id ON conversations(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_collection ON conversations(collection_used)",
                "CREATE INDEX IF NOT EXISTS idx_error ON conversations(error_occurred)",
                "CREATE INDEX IF NOT EXISTS idx_image_gen ON conversations(is_image_generation)",
                "CREATE INDEX IF NOT EXISTS idx_bot_name ON conversations(bot_name)"  # 新增 bot_name 索引
            ]
            
            for index_sql in indexes:
                try:
                    self.db_adapter.execute_update(index_sql)
                except Exception as e:
                    # 索引可能已存在，記錄但不中斷
                    logger.debug(f"索引創建警告: {e}")
            
            print(f"✅ {self.db_type.upper()}對話數據庫初始化成功 (自動欄位修復已啟用)")
            
        except Exception as e:
            logger.error(f"❌ 數據庫初始化失敗: {e}")
            raise

    def log_conversation(self, 
                    user_id: str,
                    user_query: str,
                    ai_response: str,
                    collection_used: str = None,
                    retrieved_docs: List[str] = None,
                    doc_similarities: List[float] = None,
                    processing_time_ms: int = 0,
                    is_image_generation: bool = False,
                    image_url: str = None,
                    error_occurred: bool = False,
                    error_message: str = None,
                    authenticated_user_id: int = None,
                    user_role: str = "anonymous",
                    chunk_references: Optional[List[Dict]] = None,
                    chunk_indices: Optional[List[int]] = None,
                    bot_name: str = None) -> str:  # 新增 bot_name 參數
        """記錄對話 - 修正版，支持 bot_name 參數"""
        logger.info(f"Logging conversation for user_id: {user_id}, bot_name: {bot_name}")
        try:
            with self.db_adapter.transaction():
                # 處理檢索文檔和相似度
                retrieved_docs_json = json.dumps(retrieved_docs or [], ensure_ascii=False)
                doc_similarities_json = json.dumps(doc_similarities or [], ensure_ascii=False)
                
                # 處理 chunk_references，確保前端兼容
                if chunk_references is None:
                    chunk_references = []
                    if retrieved_docs and doc_similarities:
                        for i, (doc, similarity) in enumerate(zip(retrieved_docs, doc_similarities)):
                            actual_index = chunk_indices[i] if chunk_indices and i < len(chunk_indices) else None
                            
                            chunk_ref = {
                                "id": f"chunk_{i+1}",  # 前端期望的 id 字段
                                "chunk_id": f"chunk_{i+1}",  # 保持向後兼容
                                "content_preview": doc[:100] + "..." if len(doc) > 100 else doc,
                                "similarity": round(similarity, 4),
                            }
                            
                            if actual_index is not None:
                                chunk_ref["index"] = actual_index
                            
                            chunk_references.append(chunk_ref)
                
                # 確保 chunk_references 格式正確
                for ref in chunk_references:
                    if isinstance(ref, dict) and 'chunk_id' in ref and 'id' not in ref:
                        ref['id'] = ref['chunk_id']  # 確保有 id 字段
                
                chunk_references_json = json.dumps(chunk_references, ensure_ascii=False)
                
                # 如果沒有 bot_name，嘗試從 collection_used 推斷
                if not bot_name and collection_used:
                    if collection_used.startswith('collection_'):
                        bot_name = collection_used.replace('collection_', '')
                
                # INSERT 語句包含 bot_name
                placeholders = self._get_placeholder(15)  # 15個參數
                insert_sql = f'''
                    INSERT INTO conversations (
                        user_id, user_query, ai_response, collection_used,
                        retrieved_docs, doc_similarities, chunk_references,
                        processing_time_ms, is_image_generation, image_url,
                        error_occurred, error_message, authenticated_user_id, user_role, bot_name
                    ) VALUES ({placeholders})
                '''
                
                conversation_id = self.db_adapter.execute_insert(insert_sql, (
                    user_id, user_query, ai_response, collection_used,
                    retrieved_docs_json, doc_similarities_json, chunk_references_json,
                    processing_time_ms, is_image_generation, image_url,
                    error_occurred, error_message, authenticated_user_id, user_role, bot_name
                ))
                
                self._update_daily_stats()
                
                valid_chunk_count = len([ref for ref in chunk_references if 'index' in ref])
                logger.info(f"✅ 記錄對話成功: ID {conversation_id}, bot: {bot_name}, chunks: {len(chunk_references)}, 有效索引: {valid_chunk_count}")
                return f"conv_{conversation_id}"
                
        except Exception as e:
            logger.error(f"❌ 記錄對話失敗: {e}")
            return None

    def get_conversations_by_bot(self, bot_name: str, limit: int = 50, offset: int = 0, search: str = None) -> Tuple[List[Dict], int]:
        """獲取特定機器人的所有對話記錄（包括使用和未使用知識庫的）- 修正版"""
        try:
            where_conditions = []
            params = []
            
            # 修正：使用更寬鬆但準確的查詢邏輯
            collection_name = f"collection_{bot_name}"
            bot_pattern = f"%{bot_name}%"
            
            if self.db_type == "sqlite":
                # 查詢條件：1) bot_name 匹配 2) collection_used 匹配 3) user_id 包含機器人名（向後兼容）
                where_conditions.append("(bot_name = ? OR collection_used = ? OR user_id LIKE ?)")
            else:  # postgresql
                where_conditions.append("(bot_name = %s OR collection_used = %s OR user_id LIKE %s)")
            
            params.extend([bot_name, collection_name, bot_pattern])
            
            # 搜尋條件
            if search:
                search_param = f"%{search}%"
                if self.db_type == "sqlite":
                    where_conditions.append("(user_query LIKE ? OR ai_response LIKE ?)")
                else:  # postgresql
                    where_conditions.append("(user_query LIKE %s OR ai_response LIKE %s)")
                params.extend([search_param, search_param])
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # 獲取總數
            count_query = f"SELECT COUNT(*) as count FROM conversations WHERE {where_clause}"
            count_result = self.db_adapter.execute_query(count_query, params)
            total = count_result[0]['count'] if count_result else 0
            
            # 獲取記錄
            if self.db_type == "sqlite":
                query = f"""
                    SELECT * FROM conversations 
                    WHERE {where_clause}
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                """
            else:  # postgresql
                query = f"""
                    SELECT * FROM conversations 
                    WHERE {where_clause}
                    ORDER BY timestamp DESC 
                    LIMIT %s OFFSET %s
                """
            
            # 準備查詢參數
            query_params = []
            query_params.extend([bot_name, collection_name, bot_pattern])
            if search:
                search_param = f"%{search}%"
                query_params.extend([search_param, search_param])
            query_params.extend([limit, offset])
            
            rows = self.db_adapter.execute_query(query, query_params)
            
            # 安全地處理結果
            conversations = []
            try:
                taipei_tz = pytz.timezone('Asia/Taipei')
            except pytz.UnknownTimeZoneError:
                taipei_tz = None

            for row in rows:
                conv = dict(row)
                
                # 處理 datetime 對象
                for key, value in conv.items():
                    if isinstance(value, datetime):
                        if taipei_tz:
                            value = value.replace(tzinfo=pytz.utc).astimezone(taipei_tz)
                        conv[key] = value.isoformat()

                # 安全地解析 JSON 字段
                try:
                    conv['retrieved_docs'] = json.loads(conv.get('retrieved_docs') or '[]')
                    conv['doc_similarities'] = json.loads(conv.get('doc_similarities') or '[]')
                    
                    # 處理 chunk_references
                    chunk_refs_raw = conv.get('chunk_references') or '[]'
                    if isinstance(chunk_refs_raw, str):
                        chunk_refs = json.loads(chunk_refs_raw)
                    else:
                        chunk_refs = chunk_refs_raw
                    
                    # 確保 chunk_references 有正確的格式
                    processed_chunk_refs = []
                    if isinstance(chunk_refs, list):
                        for i, ref in enumerate(chunk_refs):
                            if isinstance(ref, dict):
                                # 確保有 id 字段給前端使用
                                if 'id' not in ref:
                                    if 'chunk_id' in ref:
                                        ref['id'] = ref['chunk_id']
                                    else:
                                        ref['id'] = f"chunk_{i+1}"
                                processed_chunk_refs.append(ref)
                            elif isinstance(ref, (str, int)):
                                # 處理舊格式
                                processed_chunk_refs.append({
                                    "id": f"chunk_{i+1}",
                                    "chunk_id": str(ref),
                                    "index": i
                                })
                    
                    conv['chunk_references'] = processed_chunk_refs
                    
                    # 提取有效的 chunk 索引
                    chunk_ids = []
                    for ref in processed_chunk_refs:
                        if isinstance(ref, dict) and 'index' in ref:
                            idx = ref['index']
                            if isinstance(idx, (int, float)) and idx >= 0:
                                chunk_ids.append(int(idx))
                    
                    conv['chunk_ids'] = chunk_ids
                    
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"解析 JSON 字段失敗 (ID: {conv.get('id', 'unknown')}): {e}")
                    conv['retrieved_docs'] = []
                    conv['doc_similarities'] = []
                    conv['chunk_references'] = []
                    conv['chunk_ids'] = []
                
                conversations.append(conv)
            
            logger.info(f"獲取機器人 '{bot_name}' 對話記錄: {len(conversations)}/{total} (包含所有類型對話)")
            return conversations, total
            
        except Exception as e:
            logger.error(f"獲取機器人對話記錄失敗: {e}")
            return [], 0
    
    def log_conversation(self, 
                    user_id: str,
                    user_query: str,
                    ai_response: str,
                    collection_used: str = None,
                    retrieved_docs: List[str] = None,
                    doc_similarities: List[float] = None,
                    processing_time_ms: int = 0,
                    is_image_generation: bool = False,
                    image_url: str = None,
                    error_occurred: bool = False,
                    error_message: str = None,
                    authenticated_user_id: int = None,
                    user_role: str = "anonymous",
                    chunk_references: Optional[List[Dict]] = None,
                    chunk_indices: Optional[List[int]] = None) -> str:
        """記錄對話 - 修正版，正確處理 chunk 索引"""
        logger.info(f"Logging conversation for user_id: {user_id}")
        logger.info(f"user_query: {user_query}")
        logger.info(f"ai_response: {ai_response}")
        try:
            # 修正：使用正確的事務處理
            with self.db_adapter.transaction():
                # 處理檢索文檔和相似度
                retrieved_docs_json = json.dumps(retrieved_docs or [], ensure_ascii=False)
                doc_similarities_json = json.dumps(doc_similarities or [], ensure_ascii=False)
                
                # 修復：正確處理 chunk 索引
                if chunk_references is None:
                    chunk_references = []
                    if retrieved_docs and doc_similarities:
                        for i, (doc, similarity) in enumerate(zip(retrieved_docs, doc_similarities)):
                            # 使用傳入的真實 chunk 索引，而不是循環變數
                            actual_index = chunk_indices[i] if chunk_indices and i < len(chunk_indices) else None
                            
                            chunk_ref = {
                                "id": f"chunk_{i+1}",
                                "content_preview": doc[:100] + "..." if len(doc) > 100 else doc,
                                "similarity": round(similarity, 4),
                            }
                            
                            # 只有在有真實索引時才添加 index 字段
                            if actual_index is not None:
                                chunk_ref["index"] = actual_index
                                logger.debug(f"🔍 對話 chunk {i+1} 使用真實索引: {actual_index}")
                            else:
                                logger.warning(f"⚠️ 對話 chunk {i+1} 缺少真實索引，將不設置 index 字段")
                            
                            chunk_references.append(chunk_ref)
                
                chunk_references_json = json.dumps(chunk_references, ensure_ascii=False)
                
                # 修正：使用統一的參數佔位符
                placeholders = self._get_placeholder(14)  # 14個參數
                insert_sql = f'''
                    INSERT INTO conversations (
                        user_id, user_query, ai_response, collection_used,
                        retrieved_docs, doc_similarities, chunk_references,
                        processing_time_ms, is_image_generation, image_url,
                        error_occurred, error_message, authenticated_user_id, user_role
                    ) VALUES ({placeholders})
                '''
                
                conversation_id = self.db_adapter.execute_insert(insert_sql, (
                    user_id, user_query, ai_response, collection_used,
                    retrieved_docs_json, doc_similarities_json, chunk_references_json,
                    processing_time_ms, is_image_generation, image_url,
                    error_occurred, error_message, authenticated_user_id, user_role
                ))
                
                # 更新統計信息
                self._update_daily_stats()
                
                # 改進日誌信息
                valid_chunk_count = len([ref for ref in chunk_references if 'index' in ref])
                logger.info(f"✅ 記錄對話成功: ID {conversation_id}, chunks: {len(chunk_references)}, 有效索引: {valid_chunk_count}")
                return f"conv_{conversation_id}"
                
        except Exception as e:
            logger.error(f"❌ 記錄對話失敗: {e}")
            return None
    
    def get_conversations(self, 
                     limit: int = 50, 
                     offset: int = 0,
                     search: str = None,
                     collection: str = None,
                     user_id: str = None,
                     start_date: str = None,
                     end_date: str = None) -> Tuple[List[Dict], int]:
        """獲取對話記錄 - 修正版，正確提取 chunk_ids"""
        try:
            # 構建查詢條件
            where_conditions = []
            params = []
            
            if search:
                search_placeholder = self._get_placeholder(2)
                where_conditions.append(f"(user_query LIKE {search_placeholder.split(',')[0]} OR ai_response LIKE {search_placeholder.split(',')[1]})")
                search_param = f"%{search}%"
                params.extend([search_param, search_param])
            
            if collection:
                # 修改：查詢該機器人的所有對話記錄（包括未使用知識庫的）
                placeholder = self._get_placeholder()
                where_conditions.append(f"(collection_used = {placeholder} OR (collection_used IS NULL AND user_id LIKE {placeholder}))")
                params.extend([collection, f"%{collection.replace('collection_', '')}%"])
            
            if user_id:
                where_conditions.append(f"user_id = {self._get_placeholder()}")
                params.append(user_id)
            
            if start_date:
                where_conditions.append(f"DATE(timestamp) >= {self._get_placeholder()}")
                params.append(start_date)
            
            if end_date:
                where_conditions.append(f"DATE(timestamp) <= {self._get_placeholder()}")
                params.append(end_date)
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # 獲取總數
            count_query = f"SELECT COUNT(*) as count FROM conversations WHERE {where_clause}"
            count_result = self.db_adapter.execute_query(count_query, params)
            total = count_result[0]['count'] if count_result else 0
            
            # 獲取記錄
            limit_offset_placeholder = self._get_placeholder(2)
            query = f"""
                SELECT * FROM conversations 
                WHERE {where_clause}
                ORDER BY timestamp DESC 
                LIMIT {limit_offset_placeholder.split(',')[0]} OFFSET {limit_offset_placeholder.split(',')[1]}
            """
            params.extend([limit, offset])
            
            rows = self.db_adapter.execute_query(query, params)
            logger.info(f"Raw rows from DB: {rows}")
            
            # 轉為字典並解析 JSON 字段
            conversations = []
            try:
                taipei_tz = pytz.timezone('Asia/Taipei')
            except pytz.UnknownTimeZoneError:
                taipei_tz = None

            for row in rows:
                conv = dict(row)
                
                # 修正：處理 datetime 對象
                for key, value in conv.items():
                    if isinstance(value, datetime):
                        if taipei_tz:
                            # Assuming the datetime from DB is naive (UTC)
                            value = value.replace(tzinfo=pytz.utc).astimezone(taipei_tz)
                        conv[key] = value.isoformat()

                # 修復：正確解析 JSON 字段並提取 chunk_ids
                try:
                    conv['retrieved_docs'] = json.loads(conv.get('retrieved_docs') or '[]')
                    conv['doc_similarities'] = json.loads(conv.get('doc_similarities') or '[]')
                    chunk_refs = json.loads(conv.get('chunk_references') or '[]')
                    conv['chunk_references'] = chunk_refs
                    
                    # 修復：只提取有效的 chunk 索引
                    chunk_ids = []
                    if isinstance(chunk_refs, list):
                        for ref in chunk_refs:
                            if isinstance(ref, dict) and 'index' in ref:
                                # 只有當 index 字段存在且有效時才添加
                                index_val = ref['index']
                                if isinstance(index_val, (int, float)) and index_val >= 0:
                                    chunk_ids.append(int(index_val))
                                else:
                                    logger.warning(f"⚠️ 對話 {conv['id']} 包含無效的 chunk 索引: {index_val}")
                            elif isinstance(ref, (int, float)) and ref >= 0:
                                # 向後兼容：直接是數字的情況
                                chunk_ids.append(int(ref))
                    
                    conv['chunk_ids'] = chunk_ids
                    
                    if chunk_ids:
                        logger.debug(f"🔍 對話 {conv['id']} 解析出有效 chunk_ids: {chunk_ids}")
                    else:
                        logger.debug(f"🔍 對話 {conv['id']} 沒有有效的 chunk 索引")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ 解析 JSON 字段失敗 (ID: {conv['id']}): {e}")
                    conv['retrieved_docs'] = []
                    conv['doc_similarities'] = []
                    conv['chunk_references'] = []
                    conv['chunk_ids'] = []
                
                conversations.append(conv)
            
            logger.info(f"✅ 獲取對話記錄: {len(conversations)}/{total}")
            return conversations, total
            
        except Exception as e:
            logger.error(f"❌ 獲取對話記錄失敗: {e}")
            return [], 0



    def get_conversations_by_bot(self, bot_name: str, limit: int = 50, offset: int = 0, search: str = None) -> Tuple[List[Dict], int]:
        """獲取特定機器人的所有對話記錄（包括使用和未使用知識庫的）"""
        try:
            where_conditions = []
            params = []
            
            # 查詢條件：collection_used 匹配 OR user_id 包含機器人名稱
            collection_name = f"collection_{bot_name}"
            bot_pattern = f"%{bot_name}%"
            
            # 修正：使用正確的佔位符處理
            if self.db_type == "sqlite":
                where_conditions.append("(collection_used = ? OR user_id LIKE ?)")
            else:  # postgresql
                where_conditions.append("(collection_used = %s OR user_id LIKE %s)")
            params.extend([collection_name, bot_pattern])
            
            # 搜尋條件修正
            if search:
                search_param = f"%{search}%"
                if self.db_type == "sqlite":
                    where_conditions.append("(user_query LIKE ? OR ai_response LIKE ?)")
                else:  # postgresql
                    where_conditions.append("(user_query LIKE %s OR ai_response LIKE %s)")
                params.extend([search_param, search_param])
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # 獲取總數
            count_query = f"SELECT COUNT(*) as count FROM conversations WHERE {where_clause}"
            count_result = self.db_adapter.execute_query(count_query, params)
            total = count_result[0]['count'] if count_result else 0
            
            # 獲取記錄 - 修正LIMIT/OFFSET處理
            if self.db_type == "sqlite":
                query = f"""
                    SELECT * FROM conversations 
                    WHERE {where_clause}
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                """
            else:  # postgresql
                query = f"""
                    SELECT * FROM conversations 
                    WHERE {where_clause}
                    ORDER BY timestamp DESC 
                    LIMIT %s OFFSET %s
                """
            
            # 重新準備參數（因為上面的查詢已經消耗了params）
            query_params = []
            query_params.extend([collection_name, bot_pattern])
            if search:
                search_param = f"%{search}%"
                query_params.extend([search_param, search_param])
            query_params.extend([limit, offset])
            
            rows = self.db_adapter.execute_query(query, query_params)
            
            # 處理結果（與原方法相同）
            conversations = []
            try:
                taipei_tz = pytz.timezone('Asia/Taipei')
            except pytz.UnknownTimeZoneError:
                taipei_tz = None

            for row in rows:
                conv = dict(row)
                
                # 處理 datetime 對象
                for key, value in conv.items():
                    if isinstance(value, datetime):
                        if taipei_tz:
                            value = value.replace(tzinfo=pytz.utc).astimezone(taipei_tz)
                        conv[key] = value.isoformat()

                # 解析 JSON 字段
                try:
                    conv['retrieved_docs'] = json.loads(conv.get('retrieved_docs') or '[]')
                    conv['doc_similarities'] = json.loads(conv.get('doc_similarities') or '[]')
                    chunk_refs = json.loads(conv.get('chunk_references') or '[]')
                    conv['chunk_references'] = chunk_refs
                    
                    # 提取有效的 chunk 索引
                    chunk_ids = []
                    if isinstance(chunk_refs, list):
                        for ref in chunk_refs:
                            if isinstance(ref, dict) and 'index' in ref:
                                index_val = ref['index']
                                if isinstance(index_val, (int, float)) and index_val >= 0:
                                    chunk_ids.append(int(index_val))
                            elif isinstance(ref, (int, float)) and ref >= 0:
                                chunk_ids.append(int(ref))
                    
                    conv['chunk_ids'] = chunk_ids
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"解析 JSON 字段失敗 (ID: {conv['id']}): {e}")
                    conv['retrieved_docs'] = []
                    conv['doc_similarities'] = []
                    conv['chunk_references'] = []
                    conv['chunk_ids'] = []
                
                conversations.append(conv)
            
            logger.info(f"獲取機器人 '{bot_name}' 對話記錄: {len(conversations)}/{total}")
            return conversations, total
            
        except Exception as e:
            logger.error(f"獲取機器人對話記錄失敗: {e}")
            return [], 0
    
    def delete_conversation(self, conversation_id: int) -> bool:
        """刪除單筆對話"""
        try:
            delete_sql = f"DELETE FROM conversations WHERE id = {self._get_placeholder()}"
            deleted_count = self.db_adapter.execute_update(delete_sql, (conversation_id,))
            
            if deleted_count > 0:
                logger.info(f"✅ 刪除對話成功: ID {conversation_id}")
                self._update_daily_stats()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ 刪除對話失敗: {e}")
            return False
    
    def delete_conversations_batch(self, conversation_ids: List[int]) -> int:
        """批次刪除對話"""
        try:
            if not conversation_ids:
                return 0
            
            placeholders = self._get_placeholder(len(conversation_ids))
            delete_sql = f"DELETE FROM conversations WHERE id IN ({placeholders})"
            
            deleted_count = self.db_adapter.execute_update(delete_sql, conversation_ids)
            
            if deleted_count > 0:
                logger.info(f"✅ 批次刪除對話成功: {deleted_count} 筆")
                self._update_daily_stats()
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ 批次刪除對話失敗: {e}")
            return 0
    
    def get_collections_stats(self) -> List[Tuple[str, int]]:
        """獲取各集合的對話統計"""
        try:
            query = '''
                SELECT collection_used, COUNT(*) as count 
                FROM conversations 
                WHERE collection_used IS NOT NULL 
                GROUP BY collection_used 
                ORDER BY count DESC
            '''
            
            results = self.db_adapter.execute_query(query)
            return [(row['collection_used'], row['count']) for row in results]
            
        except Exception as e:
            logger.error(f"❌ 獲取集合統計失敗: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """獲取系統統計信息"""
        try:
            stats = {}
            
            # 總對話數
            result = self.db_adapter.execute_query("SELECT COUNT(*) as count FROM conversations")
            stats['total_conversations'] = result[0]['count'] if result else 0
            
            # 今日對話數 - 修正：使用適配的SQL
            if self.db_type == "sqlite":
                today_sql = "SELECT COUNT(*) as count FROM conversations WHERE DATE(timestamp) = DATE('now')"
            else:
                today_sql = "SELECT COUNT(*) as count FROM conversations WHERE DATE(timestamp) = CURRENT_DATE"
            
            result = self.db_adapter.execute_query(today_sql)
            stats['today_conversations'] = result[0]['count'] if result else 0
            
            # 成功率
            success_sql = f"SELECT COUNT(*) as count FROM conversations WHERE error_occurred = {self._get_placeholder()}"
            result = self.db_adapter.execute_query(success_sql, (False,))
            successful_conversations = result[0]['count'] if result else 0
            stats['success_rate'] = int((successful_conversations / stats['total_conversations'] * 100)) if stats['total_conversations'] > 0 else 100
            
            # 錯誤數量
            error_sql = f"SELECT COUNT(*) as count FROM conversations WHERE error_occurred = {self._get_placeholder()}"
            result = self.db_adapter.execute_query(error_sql, (True,))
            stats['error_count'] = result[0]['count'] if result else 0
            
            # 平均處理時間
            avg_sql = "SELECT AVG(processing_time_ms) as avg_time FROM conversations WHERE processing_time_ms > 0"
            result = self.db_adapter.execute_query(avg_sql)
            stats['average_processing_time_ms'] = int(result[0]['avg_time'] or 0) if result else 0
            
            # 最受歡迎的集合
            popular_sql = '''
                SELECT collection_used, COUNT(*) as count 
                FROM conversations 
                WHERE collection_used IS NOT NULL 
                GROUP BY collection_used 
                ORDER BY count DESC 
                LIMIT 1
            '''
            result = self.db_adapter.execute_query(popular_sql)
            stats['most_popular_collection'] = result[0]['collection_used'] if result else "無"
            
            # 圖片生成數量
            image_sql = f"SELECT COUNT(*) as count FROM conversations WHERE is_image_generation = {self._get_placeholder()}"
            result = self.db_adapter.execute_query(image_sql, (True,))
            stats['image_generations'] = result[0]['count'] if result else 0
            
            # 活躍會話數 - 修正：使用適配的SQL
            if self.db_type == "sqlite":
                active_sql = "SELECT COUNT(DISTINCT user_id) as count FROM conversations WHERE DATE(timestamp) = DATE('now')"
            else:
                active_sql = "SELECT COUNT(DISTINCT user_id) as count FROM conversations WHERE DATE(timestamp) = CURRENT_DATE"
            
            result = self.db_adapter.execute_query(active_sql)
            stats['active_sessions'] = result[0]['count'] if result else 0
            
            # 模擬數據
            stats['uptime_hours'] = 72
            stats['memory_usage_mb'] = 512
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ 獲取統計信息失敗: {e}")
            return {
                'total_conversations': 0,
                'today_conversations': 0,
                'success_rate': 100,
                'error_count': 0,
                'average_processing_time_ms': 0,
                'most_popular_collection': '無',
                'image_generations': 0,
                'active_sessions': 0,
                'uptime_hours': 0,
                'memory_usage_mb': 0
            }
    
    def _update_daily_stats(self):
        """更新每日統計"""
        try:
            today = datetime.now().date()
            
            # 修正：使用適配的SQL和參數佔位符
            today_placeholder = self._get_placeholder()
            
            if self.db_type == "sqlite":
                today_sql_filter = f"DATE(timestamp) = {today_placeholder}"
            else:
                today_sql_filter = f"DATE(timestamp) = {today_placeholder}"
            
            # 總數
            total_sql = f"SELECT COUNT(*) as count FROM conversations WHERE {today_sql_filter}"
            result = self.db_adapter.execute_query(total_sql, (today,))
            total_today = result[0]['count'] if result else 0
            
            # 成功數
            success_placeholder = self._get_placeholder(2)
            success_sql = f"SELECT COUNT(*) as count FROM conversations WHERE {today_sql_filter} AND error_occurred = {success_placeholder.split(',')[1]}"
            result = self.db_adapter.execute_query(success_sql, (today, False))
            successful_today = result[0]['count'] if result else 0
            
            # 失敗數
            failed_placeholder = self._get_placeholder(2)
            failed_sql = f"SELECT COUNT(*) as count FROM conversations WHERE {today_sql_filter} AND error_occurred = {failed_placeholder.split(',')[1]}"
            result = self.db_adapter.execute_query(failed_sql, (today, True))
            failed_today = result[0]['count'] if result else 0
            
            # 圖片生成數
            images_placeholder = self._get_placeholder(2)
            images_sql = f"SELECT COUNT(*) as count FROM conversations WHERE {today_sql_filter} AND is_image_generation = {images_placeholder.split(',')[1]}"
            result = self.db_adapter.execute_query(images_sql, (today, True))
            images_today = result[0]['count'] if result else 0
            
            # 平均時間
            avg_sql = f"SELECT AVG(processing_time_ms) as avg_time FROM conversations WHERE {today_sql_filter} AND processing_time_ms > 0"
            result = self.db_adapter.execute_query(avg_sql, (today,))
            avg_time_today = result[0]['avg_time'] or 0 if result else 0
            
            # 修正：使用適配的UPSERT語法
            if self.db_type == "postgresql":
                upsert_placeholder = self._get_placeholder(6)
                upsert_sql = f'''
                    INSERT INTO conversation_stats 
                    (stat_date, total_conversations, successful_conversations, failed_conversations, 
                     image_generations, average_processing_time_ms)
                    VALUES ({upsert_placeholder})
                    ON CONFLICT (stat_date) DO UPDATE SET
                    total_conversations = EXCLUDED.total_conversations,
                    successful_conversations = EXCLUDED.successful_conversations,
                    failed_conversations = EXCLUDED.failed_conversations,
                    image_generations = EXCLUDED.image_generations,
                    average_processing_time_ms = EXCLUDED.average_processing_time_ms
                '''
            else:  # SQLite
                upsert_placeholder = self._get_placeholder(6)
                upsert_sql = f'''
                    INSERT OR REPLACE INTO conversation_stats 
                    (stat_date, total_conversations, successful_conversations, failed_conversations, 
                     image_generations, average_processing_time_ms)
                    VALUES ({upsert_placeholder})
                '''
            
            self.db_adapter.execute_update(upsert_sql, (
                today, total_today, successful_today, failed_today, images_today, avg_time_today
            ))
            
        except Exception as e:
            logger.error(f"❌ 更新每日統計失敗: {e}")
    
    def cleanup_old_records(self, days_to_keep: int = 30):
        """清理舊記錄"""
        try:
            # 修正：使用適配的日期函數
            if self.db_type == "postgresql":
                cleanup_sql = f'''
                    DELETE FROM conversations 
                    WHERE timestamp < NOW() - INTERVAL '{days_to_keep} days'
                '''
                deleted_count = self.db_adapter.execute_update(cleanup_sql)
            else:  # SQLite
                cleanup_sql = f'''
                    DELETE FROM conversations 
                    WHERE timestamp < datetime('now', '-{days_to_keep} days')
                '''
                deleted_count = self.db_adapter.execute_update(cleanup_sql)
            
            logger.info(f"✅ 清理舊記錄完成: 刪除 {deleted_count} 筆")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ 清理舊記錄失敗: {e}")
            return 0
    
    def export_conversations(self, output_file: str, format: str = "json"):
        """導出對話記錄"""
        try:
            conversations, _ = self.get_conversations(limit=10000)  # 導出所有記錄
            
            if format.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(conversations, f, ensure_ascii=False, indent=2, default=str)
            elif format.lower() == "csv":
                import csv
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    if conversations:
                        writer = csv.DictWriter(f, fieldnames=conversations[0].keys())
                        writer.writeheader()
                        for conv in conversations:
                            # 將複雜字段轉為字符串
                            conv_copy = conv.copy()
                            for key in ['retrieved_docs', 'doc_similarities', 'chunk_references']:
                                if key in conv_copy:
                                    conv_copy[key] = json.dumps(conv_copy[key], ensure_ascii=False)
                            writer.writerow(conv_copy)
            
            logger.info(f"✅ 導出對話記錄完成: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 導出對話記錄失敗: {e}")
            return False
    
    def close(self):
        """關閉數據庫連接"""
        if self.db_adapter:
            self.db_adapter.disconnect()


# 創建全域實例（向後兼容性）
def create_logger_instance(db_config: Dict = None):
    """創建記錄器實例的工廠函數"""
    return PostgreSQLConversationLogger(db_config)


# 為了向後兼容，創建別名
EnhancedConversationLogger = PostgreSQLConversationLogger

# 導出列表
__all__ = ['PostgreSQLConversationLogger', 'EnhancedConversationLogger', 'create_logger_instance']


# 測試函數
if __name__ == "__main__":
    # 測試SQLite配置（如果PostgreSQL不可用）
    sqlite_config = {
        "type": "sqlite",
        "db_file": "test_pg_conversations.db"
    }
    
    print("🧪 測試PostgreSQL版對話記錄器（修正版）")
    print("=" * 50)
    
    try:
        logger_instance = PostgreSQLConversationLogger(sqlite_config)
        print(f"✅ 使用{logger_instance.db_type.upper()}適配器測試")
    except Exception as e:
        print(f"❌ 創建記錄器失敗: {e}")
        exit(1)
    
    # 測試記錄對話 - 帶有正確的 chunk 索引
    conv_id = logger_instance.log_conversation(
        user_id="test_user_pg_001",
        user_query="什麼是PostgreSQL？",
        ai_response="PostgreSQL是一個強大的開源關係式資料庫...",
        collection_used="collection_database",
        retrieved_docs=["PostgreSQL文檔1內容", "PostgreSQL文檔2內容"],
        doc_similarities=[0.87, 0.74],
        processing_time_ms=1800,
        user_role="user",
        chunk_indices=[25, 58]  # 真實的向量資料庫索引
    )
    
    print(f"記錄對話成功: {conv_id}")
    
    # 測試獲取對話
    conversations, total = logger_instance.get_conversations(limit=10)
    print(f"獲取到 {len(conversations)} 筆對話記錄，總共 {total} 筆")
    
    # 檢查 chunk_ids
    if conversations:
        first_conv = conversations[0]
        print(f"第一個對話的 chunk_ids: {first_conv.get('chunk_ids', [])}")
    
    # 測試統計
    stats = logger_instance.get_statistics()
    print(f"統計信息: {stats}")
    
    # 關閉連接
    logger_instance.close()
    
    # 清理測試文件
    import os
    test_files = ["test_pg_conversations.db"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    
    print("✅ PostgreSQL版對話記錄器測試完成（修正版）- chunk 索引問題已修復")