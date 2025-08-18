#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
database_adapter.py - 統一數據庫抽象層
支援 SQLite 和 PostgreSQL，適用於用戶管理和對話記錄系統
版本：2.0 - 支援連接池和改進的錯誤處理
"""

import os
import json
import sqlite3
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# 嘗試導入 PostgreSQL 相關模組
try:
    import psycopg2
    from psycopg2 import pool
    import psycopg2.extras
    from psycopg2 import sql
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 抽象基類 ==========
class DatabaseAdapter(ABC):
    """數據庫抽象介面 - 定義統一的操作方法"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
        self.lock = threading.RLock()
        self._is_connected = False
    
    @abstractmethod
    def connect(self):
        """建立數據庫連接"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """關閉數據庫連接"""
        pass
    
    @abstractmethod
    def execute_query(self, sql: str, params: Union[tuple, list] = None) -> List[Dict[str, Any]]:
        """執行查詢"""
        with self.lock:
            try:
                conn = self.get_connection()
                
                # 轉換 SQL 語法
                sql, params = self._convert_sql_params(sql, params)
                
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(sql, params or ())
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                    
            except Exception as e:
                logger.error(f"❌ PostgreSQL 查詢失敗: {e}")
                logger.error(f"   SQL: {sql}")
                logger.error(f"   參數: {params}")
                raise
    def execute_update(self, sql: str, params: Union[tuple, list] = None) -> int:
        """執行更新"""
        with self.lock:
            try:
                conn = self.get_connection()
                
                # 轉換 SQL 語法
                sql, params = self._convert_sql_params(sql, params)
                
                with conn.cursor() as cursor:
                    cursor.execute(sql, params or ())
                    rowcount = cursor.rowcount
                    conn.commit()
                    return rowcount
                    
            except Exception as e:
                logger.error(f"❌ PostgreSQL 更新失敗: {e}")
                logger.error(f"   SQL: {sql}")
                logger.error(f"   參數: {params}")
                try:
                    conn = self.get_connection()
                    conn.rollback()
                except:
                    pass
                raise
    def execute_insert(self, sql: str, params: Union[tuple, list] = None) -> Optional[int]:
        """執行插入 - 修正版本"""
        with self.lock:
            conn = None
            try:
                conn = self.get_connection()
                original_sql = sql
                
                with conn.cursor() as cursor:
                    # 智能處理 RETURNING 子句
                    sql_upper = sql.upper().strip()
                    
                    # 只有在 INSERT 語句且沒有 RETURNING 時才添加
                    if (sql_upper.startswith('INSERT') and 
                        'RETURNING' not in sql_upper):
                        # 檢查是否有可能的 id 欄位
                        sql += " RETURNING id"
                    
                    # 轉換 SQL 語法
                    sql, params = self._convert_sql_params(sql, params)
                    
                    cursor.execute(sql, params or ())
                    
                    # 處理返回值
                    if 'RETURNING' in sql.upper():
                        result = cursor.fetchone()
                        conn.commit()
                        if result:
                            return result[0] if isinstance(result, (tuple, list)) else result
                        return None
                    else:
                        # 沒有 RETURNING 的情況，提交並返回影響的行數
                        rowcount = cursor.rowcount
                        conn.commit()
                        return rowcount if rowcount > 0 else None
            except Exception as e:
                logger.error(f"❌ PostgreSQL 插入失敗: {e}")
                logger.error(f"   原始SQL: {original_sql}")
                logger.error(f"   執行SQL: {sql}")
                logger.error(f"   參數: {params}")
                
                # 處理特定錯誤類型
                if "column" in str(e).lower() and "does not exist" in str(e).lower():
                    logger.error("   可能原因：表格沒有 'id' 欄位，請考慮在 SQL 中明確指定 RETURNING 欄位")
                
                # 安全的回滾處理
                if conn:
                    try:
                        conn.rollback()
                    except Exception as rollback_error:
                        logger.error(f"   回滾失敗: {rollback_error}")
                raise
    def begin_transaction(self):
        """開始事務"""
        pass
    
    @abstractmethod
    def commit_transaction(self):
        """提交事務"""
        pass
    
    @abstractmethod
    def rollback_transaction(self):
        """回滾事務"""
        pass
    
    @abstractmethod
    def get_table_columns(self, table_name: str) -> List[str]:
        """獲取表格的所有列名"""
        pass
    
    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """檢查表格是否存在"""
        pass
    
    @contextmanager
    def transaction(self):
        """事務上下文管理器"""
        try:
            self.begin_transaction()
            yield
            self.commit_transaction()
        except Exception as e:
            try:
                self.rollback_transaction()
            except Exception as rollback_error:
                logger.error(f"回滾事務失敗: {rollback_error}")
            raise e
    
    def close(self):
        """關閉連接的統一方法"""
        self.disconnect()

# ========== SQLite 適配器 ==========
class SQLiteAdapter(DatabaseAdapter):
    """SQLite 數據庫適配器 - 改進版本"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.db_file = config.get("db_file", "database.db")
        self.timeout = config.get("timeout", 30.0)
        self.journal_mode = config.get("journal_mode", "WAL")
        self.synchronous = config.get("synchronous", "NORMAL")
        self.cache_size = config.get("cache_size", 2000)
        
    def connect(self):
        """連接到 SQLite 數據庫"""
        try:
            # 確保數據庫目錄存在
            db_path = Path(self.db_file)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.connection = sqlite3.connect(
                self.db_file, 
                timeout=self.timeout,
                check_same_thread=False
            )
            self.connection.row_factory = sqlite3.Row
            
            # 優化 SQLite 設定
            self.connection.execute(f"PRAGMA journal_mode = {self.journal_mode}")
            self.connection.execute(f"PRAGMA synchronous = {self.synchronous}")
            self.connection.execute(f"PRAGMA cache_size = {self.cache_size}")
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            self._is_connected = True
            logger.info(f"✅ SQLite 連接成功: {self.db_file}")
            
        except Exception as e:
            logger.error(f"❌ SQLite 連接失敗: {e}")
            raise
    
    def disconnect(self):
        """關閉 SQLite 連接"""
        if self.connection:
            try:
                self.connection.close()
                self._is_connected = False
                logger.debug("SQLite 連接已關閉")
            except Exception as e:
                logger.error(f"關閉 SQLite 連接失敗: {e}")
            finally:
                self.connection = None
    
    def execute_query(self, sql: str, params: Union[tuple, list] = None) -> List[Dict[str, Any]]:
        """執行查詢"""
        with self.lock:
            try:
                conn = self.get_connection()
                
                # 轉換 SQL 語法
                sql, params = self._convert_sql_params(sql, params)
                
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(sql, params or ())
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                    
            except Exception as e:
                logger.error(f"❌ PostgreSQL 查詢失敗: {e}")
                logger.error(f"   SQL: {sql}")
                logger.error(f"   參數: {params}")
                raise
    def execute_update(self, sql: str, params: Union[tuple, list] = None) -> int:
        """執行更新"""
        with self.lock:
            try:
                conn = self.get_connection()
                
                # 轉換 SQL 語法
                sql, params = self._convert_sql_params(sql, params)
                
                with conn.cursor() as cursor:
                    cursor.execute(sql, params or ())
                    rowcount = cursor.rowcount
                    conn.commit()
                    return rowcount
                    
            except Exception as e:
                logger.error(f"❌ PostgreSQL 更新失敗: {e}")
                logger.error(f"   SQL: {sql}")
                logger.error(f"   參數: {params}")
                try:
                    conn = self.get_connection()
                    conn.rollback()
                except:
                    pass
                raise
    def execute_insert(self, sql: str, params: Union[tuple, list] = None) -> Optional[int]:
        """執行插入 - 修正版本"""
        with self.lock:
            conn = None
            try:
                conn = self.get_connection()
                original_sql = sql
                
                with conn.cursor() as cursor:
                    # 智能處理 RETURNING 子句
                    sql_upper = sql.upper().strip()
                    
                    # 只有在 INSERT 語句且沒有 RETURNING 時才添加
                    if (sql_upper.startswith('INSERT') and 
                        'RETURNING' not in sql_upper):
                        # 檢查是否有可能的 id 欄位
                        sql += " RETURNING id"
                    
                    # 轉換 SQL 語法
                    sql, params = self._convert_sql_params(sql, params)
                    
                    cursor.execute(sql, params or ())
                    
                    # 處理返回值
                    if 'RETURNING' in sql.upper():
                        result = cursor.fetchone()
                        conn.commit()
                        if result:
                            return result[0] if isinstance(result, (tuple, list)) else result
                        return None
                    else:
                        # 沒有 RETURNING 的情況，提交並返回影響的行數
                        rowcount = cursor.rowcount
                        conn.commit()
                        return rowcount if rowcount > 0 else None
            except Exception as e:
                logger.error(f"❌ PostgreSQL 插入失敗: {e}")
                logger.error(f"   原始SQL: {original_sql}")
                logger.error(f"   執行SQL: {sql}")
                logger.error(f"   參數: {params}")
                
                # 處理特定錯誤類型
                if "column" in str(e).lower() and "does not exist" in str(e).lower():
                    logger.error("   可能原因：表格沒有 'id' 欄位，請考慮在 SQL 中明確指定 RETURNING 欄位")
                
                # 安全的回滾處理
                if conn:
                    try:
                        conn.rollback()
                    except Exception as rollback_error:
                        logger.error(f"   回滾失敗: {rollback_error}")
                raise
    def begin_transaction(self):
        """開始事務"""
        if not self.connection or not self._is_connected:
            self.connect()
        # SQLite 自動開始事務
    
    def commit_transaction(self):
        """提交事務"""
        if self.connection:
            self.connection.commit()
    
    def rollback_transaction(self):
        """回滾事務"""
        if self.connection:
            self.connection.rollback()
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """獲取表格列名"""
        try:
            results = self.execute_query(f"PRAGMA table_info({table_name})")
            return [row['name'] for row in results]
        except Exception:
            return []
    
    def table_exists(self, table_name: str) -> bool:
        """檢查表格是否存在"""
        try:
            results = self.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            return len(results) > 0
        except Exception:
            return False

# ========== PostgreSQL 適配器 ==========
class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL 數據庫適配器 - 支援連接池"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("需要安裝 psycopg2: pip install psycopg2-binary")
        
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 5432)
        self.database = config.get("database", "chatbot_system")
        self.user = config.get("user", "postgres")
        self.password = config.get("password", "")
        self.schema = config.get("schema", "public")
        
        # 連接池配置
        self.min_connections = config.get("min_connections", 1)
        self.max_connections = config.get("max_connections", 10)
        self.connect_timeout = config.get("connect_timeout", 30)
        self.command_timeout = config.get("command_timeout", 30)
        
        # 連接池和線程本地存儲
        self.connection_pool = None
        self._local = threading.local()
        
    def _convert_sql_params(self, sql: str, params: Union[tuple, list] = None):
        """將 SQLite 風格的 SQL 轉換為 PostgreSQL 風格"""
        if params and '?' in sql:
            # 將所有 ? 參數轉換為 %s 格式（psycopg2 風格）
            sql = sql.replace('?', '%s')
        
        # 修復布爾值比較
        sql = sql.replace('= 1', '= TRUE')
        sql = sql.replace('= 0', '= FALSE')
        sql = sql.replace('CASE WHEN is_active = TRUE THEN 1', 'CASE WHEN is_active THEN 1')
        
        return sql, params

    def connect(self):
        """創建 PostgreSQL 連接池"""
        try:
            if not self.connection_pool:
                self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                    self.min_connections,
                    self.max_connections,
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    connect_timeout=self.connect_timeout
                )
                self._is_connected = True
                logger.info(f"✅ PostgreSQL 連接池創建成功: {self.host}:{self.port}/{self.database}")
                logger.info(f"   連接池大小: {self.min_connections}-{self.max_connections}")
                
        except Exception as e:
            logger.error(f"❌ PostgreSQL 連接池創建失敗: {e}")
            raise
    
    def get_connection(self):
        """從連接池獲取連接"""
        if not self.connection_pool:
            self.connect()
        
        # 為每個線程獲取獨立連接
        if (not hasattr(self._local, 'connection') or 
            self._local.connection is None or 
            self._local.connection.closed):
            try:
                self._local.connection = self.connection_pool.getconn()
                if self._local.connection is None:
                    raise ConnectionError("連接池返回 None 連接")
                
                self._local.connection.autocommit = False
                # 設置搜索路徑
                with self._local.connection.cursor() as cursor:
                    cursor.execute(f"SET search_path TO {self.schema}")
                    
                logger.debug(f"✅ 為線程獲取新連接: {threading.current_thread().name}")
                    
            except Exception as e:
                logger.error(f"從連接池獲取連接失敗: {e}")
                # 清理可能的無效連接
                if hasattr(self._local, 'connection'):
                    self._local.connection = None
                raise
        
        return self._local.connection
    
    def release_connection(self):
        """釋放連接回連接池"""
        if (hasattr(self._local, 'connection') and 
            self._local.connection is not None):
            try:
                if not self._local.connection.closed:
                    # 確保沒有未完成的事務
                    if self._local.connection.status != psycopg2.extensions.STATUS_READY:
                        self._local.connection.rollback()
                    self.connection_pool.putconn(self._local.connection)
                    logger.debug(f"✅ 連接已釋放回連接池: {threading.current_thread().name}")
            except Exception as e:
                logger.error(f"釋放連接失敗: {e}")
                # 如果釋放失敗，嘗試關閉連接
                try:
                    if (self._local.connection and 
                        not self._local.connection.closed):
                        self._local.connection.close()
                except:
                    pass
            finally:
                self._local.connection = None
    
    def disconnect(self):
        """關閉 PostgreSQL 連接池"""
        try:
            # 釋放當前線程的連接
            if hasattr(self._local, 'connection'):
                self.release_connection()
            
            # 關閉連接池
            if self.connection_pool:
                self.connection_pool.closeall()
                self.connection_pool = None
                self._is_connected = False
                logger.info("✅ PostgreSQL 連接池已關閉")
                
        except Exception as e:
            logger.error(f"❌ 關閉連接池失敗: {e}")
    
    def execute_query(self, sql: str, params: Union[tuple, list] = None) -> List[Dict[str, Any]]:
        """執行查詢"""
        with self.lock:
            try:
                conn = self.get_connection()
                
                # 轉換 SQL 語法
                sql, params = self._convert_sql_params(sql, params)
                
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(sql, params or ())
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                    
            except Exception as e:
                logger.error(f"❌ PostgreSQL 查詢失敗: {e}")
                logger.error(f"   SQL: {sql}")
                logger.error(f"   參數: {params}")
                raise
    def execute_update(self, sql: str, params: Union[tuple, list] = None) -> int:
        """執行更新"""
        with self.lock:
            try:
                conn = self.get_connection()
                
                # 轉換 SQL 語法
                sql, params = self._convert_sql_params(sql, params)
                
                with conn.cursor() as cursor:
                    cursor.execute(sql, params or ())
                    rowcount = cursor.rowcount
                    conn.commit()
                    return rowcount
                    
            except Exception as e:
                logger.error(f"❌ PostgreSQL 更新失敗: {e}")
                logger.error(f"   SQL: {sql}")
                logger.error(f"   參數: {params}")
                try:
                    conn = self.get_connection()
                    conn.rollback()
                except:
                    pass
                raise
    def execute_insert(self, sql: str, params: Union[tuple, list] = None) -> Optional[int]:
        """執行插入 - 修正版本"""
        with self.lock:
            conn = None
            try:
                conn = self.get_connection()
                original_sql = sql
                
                with conn.cursor() as cursor:
                    # 智能處理 RETURNING 子句
                    sql_upper = sql.upper().strip()
                    
                    # 只有在 INSERT 語句且沒有 RETURNING 時才添加
                    if (sql_upper.startswith('INSERT') and 
                        'RETURNING' not in sql_upper):
                        # 檢查是否有可能的 id 欄位
                        sql += " RETURNING id"
                    
                    # 轉換 SQL 語法
                    sql, params = self._convert_sql_params(sql, params)
                    
                    cursor.execute(sql, params or ())
                    
                    # 處理返回值
                    if 'RETURNING' in sql.upper():
                        result = cursor.fetchone()
                        conn.commit()
                        if result:
                            return result[0] if isinstance(result, (tuple, list)) else result
                        return None
                    else:
                        # 沒有 RETURNING 的情況，提交並返回影響的行數
                        rowcount = cursor.rowcount
                        conn.commit()
                        return rowcount if rowcount > 0 else None
            except Exception as e:
                logger.error(f"❌ PostgreSQL 插入失敗: {e}")
                logger.error(f"   原始SQL: {original_sql}")
                logger.error(f"   執行SQL: {sql}")
                logger.error(f"   參數: {params}")
                
                # 處理特定錯誤類型
                if "column" in str(e).lower() and "does not exist" in str(e).lower():
                    logger.error("   可能原因：表格沒有 'id' 欄位，請考慮在 SQL 中明確指定 RETURNING 欄位")
                
                # 安全的回滾處理
                if conn:
                    try:
                        conn.rollback()
                    except Exception as rollback_error:
                        logger.error(f"   回滾失敗: {rollback_error}")
                raise
    def begin_transaction(self):
        """開始事務"""
        conn = self.get_connection()
        # PostgreSQL 會自動開始事務
    
    def commit_transaction(self):
        """提交事務"""
        try:
            conn = self.get_connection()
            conn.commit()
        except Exception as e:
            logger.error(f"❌ 事務提交失敗: {e}")
            raise
    
    def rollback_transaction(self):
        """回滾事務"""
        try:
            conn = self.get_connection()
            conn.rollback()
        except Exception as e:
            logger.error(f"❌ 事務回滾失敗: {e}")
            raise
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """獲取表格列名"""
        try:
            results = self.execute_query("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s AND table_schema = %s
                ORDER BY ordinal_position
            """, (table_name, self.schema))
            return [row['column_name'] for row in results]
        except Exception:
            return []
    
    def table_exists(self, table_name: str) -> bool:
        """檢查表格是否存在"""
        try:
            results = self.execute_query("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = %s AND table_schema = %s
            """, (table_name, self.schema))
            return len(results) > 0
        except Exception:
            return False

# ========== 工廠模式 ==========
class DatabaseFactory:
    """數據庫適配器工廠"""
    
    @staticmethod
    def create_adapter(db_type: str, config: Dict[str, Any]) -> DatabaseAdapter:
        """創建數據庫適配器"""
        db_type = db_type.lower().strip()
        
        if db_type == "sqlite":
            return SQLiteAdapter(config)
        elif db_type in ["postgresql", "postgres"]:
            return PostgreSQLAdapter(config)
        else:
            raise ValueError(f"不支援的數據庫類型: {db_type}")
    
    @staticmethod
    def create_from_env(db_name: str = "default") -> DatabaseAdapter:
        """從環境變數創建適配器"""
        db_type = os.getenv(f"DB_TYPE_{db_name.upper()}", os.getenv("DB_TYPE", "sqlite"))
        
        if db_type.lower() == "sqlite":
            config = {
                "db_file": os.getenv(f"SQLITE_FILE_{db_name.upper()}", f"{db_name}.db"),
                "timeout": float(os.getenv("SQLITE_TIMEOUT", "30.0")),
                "journal_mode": os.getenv("SQLITE_JOURNAL_MODE", "WAL"),
                "synchronous": os.getenv("SQLITE_SYNCHRONOUS", "NORMAL"),
                "cache_size": int(os.getenv("SQLITE_CACHE_SIZE", "2000"))
            }
        elif db_type.lower() in ["postgresql", "postgres"]:
            config = {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "database": os.getenv("POSTGRES_DATABASE", "chatbot_system"),
                "user": os.getenv("POSTGRES_USER", "postgres"),
                "password": os.getenv("POSTGRES_PASSWORD", ""),
                "schema": os.getenv("POSTGRES_SCHEMA", "public"),
                "min_connections": int(os.getenv("POSTGRES_MIN_CONNECTIONS", "1")),
                "max_connections": int(os.getenv("POSTGRES_MAX_CONNECTIONS", "10")),
                "connect_timeout": int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "30")),
                "command_timeout": int(os.getenv("POSTGRES_COMMAND_TIMEOUT", "30"))
            }
        else:
            raise ValueError(f"不支援的數據庫類型: {db_type}")
        
        return DatabaseFactory.create_adapter(db_type, config)

# ========== SQL 語法適配器 ==========
class SQLDialect:
    """SQL 語法差異處理"""
    
    @staticmethod
    def get_auto_increment_column(db_type: str) -> str:
        """獲取自增列定義"""
        if db_type.lower() == "sqlite":
            return "INTEGER PRIMARY KEY AUTOINCREMENT"
        elif db_type.lower() in ["postgresql", "postgres"]:
            return "SERIAL PRIMARY KEY"
        else:
            raise ValueError(f"不支援的數據庫類型: {db_type}")
    
    @staticmethod
    def get_boolean_column(db_type: str, default_value: bool = True) -> str:
        """獲取布爾列定義"""
        if db_type.lower() == "sqlite":
            return f"BOOLEAN DEFAULT {1 if default_value else 0}"
        elif db_type.lower() in ["postgresql", "postgres"]:
            return f"BOOLEAN DEFAULT {'TRUE' if default_value else 'FALSE'}"
        else:
            raise ValueError(f"不支援的數據庫類型: {db_type}")
    
    @staticmethod
    def get_timestamp_column(db_type: str) -> str:
        """獲取時間戳列定義"""
        if db_type.lower() == "sqlite":
            return "TEXT DEFAULT CURRENT_TIMESTAMP"
        elif db_type.lower() in ["postgresql", "postgres"]:
            return "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        else:
            raise ValueError(f"不支援的數據庫類型: {db_type}")
    
    @staticmethod
    def get_json_column(db_type: str) -> str:
        """獲取 JSON 列定義"""
        if db_type.lower() == "sqlite":
            return "TEXT"  # SQLite 沒有原生 JSON 支援
        elif db_type.lower() in ["postgresql", "postgres"]:
            return "JSONB"  # PostgreSQL 使用 JSONB
        else:
            raise ValueError(f"不支援的數據庫類型: {db_type}")

# ========== 連接管理器 ==========
class ConnectionManager:
    """數據庫連接管理器 - 統一管理多個數據庫連接"""
    
    def __init__(self):
        self.adapters: Dict[str, DatabaseAdapter] = {}
        self.lock = threading.RLock()
    
    def get_adapter(self, name: str, config: Dict[str, Any] = None) -> DatabaseAdapter:
        """獲取或創建數據庫適配器"""
        with self.lock:
            if name not in self.adapters:
                if not config:
                    raise ValueError(f"首次創建適配器 '{name}' 需要提供配置")
                
                self.adapters[name] = DatabaseFactory.create_adapter(
                    config["type"], config
                )
                logger.info(f"✅ 創建數據庫適配器: {name} ({config['type']})")
            
            return self.adapters[name]
    
    def close_all(self):
        """關閉所有數據庫連接"""
        with self.lock:
            for name, adapter in self.adapters.items():
                try:
                    adapter.close()
                    logger.info(f"✅ 關閉數據庫連接: {name}")
                except Exception as e:
                    logger.error(f"❌ 關閉數據庫連接失敗 {name}: {e}")
            
            self.adapters.clear()

# 全局連接管理器實例
connection_manager = ConnectionManager()

# ========== 測試函數 ==========
def test_adapter():
    """測試數據庫適配器"""
    print("🧪 測試數據庫抽象層")
    print("=" * 50)
    
    # 測試 SQLite
    print("📝 測試 SQLite 適配器...")
    sqlite_config = {
        "db_file": "test_adapter.db",
        "journal_mode": "WAL",
        "cache_size": 1000
    }
    sqlite_adapter = DatabaseFactory.create_adapter("sqlite", sqlite_config)
    
    try:
        sqlite_adapter.connect()
        print("✅ SQLite 適配器連接成功")
        
        # 創建測試表
        sqlite_adapter.execute_update("""
            CREATE TABLE IF NOT EXISTS test_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 插入測試數據
        user_id = sqlite_adapter.execute_insert(
            "INSERT INTO test_users (username) VALUES (?)",
            ("test_user",)
        )
        print(f"✅ 插入成功，ID: {user_id}")
        
        # 查詢測試數據
        users = sqlite_adapter.execute_query("SELECT * FROM test_users")
        print(f"✅ 查詢成功，用戶數: {len(users)}")
        
    except Exception as e:
        print(f"❌ SQLite 測試失敗: {e}")
    finally:
        sqlite_adapter.disconnect()
    
    # 清理測試文件
    if os.path.exists("test_adapter.db"):
        os.remove("test_adapter.db")
    
    # 測試 PostgreSQL（如果可用）
    if PSYCOPG2_AVAILABLE:
        print("\n📝 測試 PostgreSQL 適配器...")
        pg_config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DATABASE", "test_db"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", ""),
            "min_connections": 1,
            "max_connections": 3
        }
        
        try:
            pg_adapter = DatabaseFactory.create_adapter("postgresql", pg_config)
            pg_adapter.connect()
            print("✅ PostgreSQL 適配器連接成功")
            
            # 簡單查詢測試
            result = pg_adapter.execute_query("SELECT 1 as test")
            print(f"✅ PostgreSQL 查詢測試成功: {result}")
            
        except Exception as e:
            print(f"⚠️ PostgreSQL 測試跳過（可能未配置）: {e}")
        finally:
            try:
                pg_adapter.disconnect()
            except:
                pass
    else:
        print("\n⚠️ PostgreSQL 測試跳過（psycopg2 未安裝）")
    
    print("=" * 50)
    print("✅ 數據庫抽象層測試完成")

if __name__ == "__main__":
    test_adapter()