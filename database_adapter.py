
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
from urllib.parse import quote_plus

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
    def get_connection(self):
        """獲取數據庫連接"""
        pass
    
    @abstractmethod
    def execute_query(self, sql: str, params: Union[tuple, list] = None) -> List[Dict[str, Any]]:
        """執行查詢"""
        pass
    
    @abstractmethod
    def execute_update(self, sql: str, params: Union[tuple, list] = None) -> int:
        """執行更新"""
        pass
    
    @abstractmethod
    def execute_insert(self, sql: str, params: Union[tuple, list] = None) -> Optional[int]:
        """執行插入"""
        pass
    
    def begin_transaction(self):
        """開始事務"""
        pass
    
    def commit_transaction(self):
        """提交事務"""
        if self.connection:
            self.connection.commit()
    
    def rollback_transaction(self):
        """回滾事務"""
        if self.connection:
            self.connection.rollback()
    
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

# ========== SQLite 適配器 (完全修復版) ==========
class SQLiteAdapter(DatabaseAdapter):
    """SQLite 數據庫適配器 - 完全修復版本"""
    
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
    
    def get_connection(self):
        """獲取 SQLite 連接"""
        if not self.connection or not self._is_connected:
            self.connect()
        return self.connection
    
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
        """執行查詢 - SQLite 正確實現"""
        with self.lock:
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                cursor.execute(sql, params or ())
                rows = cursor.fetchall()
                # 轉換 sqlite3.Row 為字典
                return [dict(row) for row in rows]
                    
            except Exception as e:
                logger.error(f"❌ SQLite 查詢失敗: {e}")
                logger.error(f"   SQL: {sql}")
                logger.error(f"   參數: {params}")
                raise
    
    def execute_update(self, sql: str, params: Union[tuple, list] = None) -> int:
        """執行更新 - SQLite 正確實現"""
        with self.lock:
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                cursor.execute(sql, params or ())
                rowcount = cursor.rowcount
                conn.commit()
                return rowcount
                    
            except Exception as e:
                logger.error(f"❌ SQLite 更新失敗: {e}")
                logger.error(f"   SQL: {sql}")
                logger.error(f"   參數: {params}")
                try:
                    conn.rollback()
                except:
                    pass
                raise
    
    def execute_insert(self, sql: str, params: Union[tuple, list] = None) -> Optional[int]:
        """執行插入 - SQLite 正確實現"""
        with self.lock:
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                cursor.execute(sql, params or ())
                
                # SQLite 使用 lastrowid 獲取插入的 ID
                insert_id = cursor.lastrowid
                conn.commit()
                return insert_id
                    
            except Exception as e:
                logger.error(f"❌ SQLite 插入失敗: {e}")
                logger.error(f"   SQL: {sql}")
                logger.error(f"   參數: {params}")
                try:
                    conn.rollback()
                except:
                    pass
                raise
    
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

# ========== PostgreSQL 適配器 (修復版) ==========
class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL 數據庫適配器 - 支持連接池"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("需要安裝 psycopg2: pip install psycopg2-binary")
        
        # 🔧 處理不同的配置方式
        if "connection_string" in config:
            self.connection_string = config["connection_string"]
        elif "DATABASE_URL" in config:
            self.connection_string = config["DATABASE_URL"]
        else:
            # 從個別參數構建
            self.host = config.get("host", "localhost")
            self.port = config.get("port", 5432)
            self.database = config.get("database", "chatbot_system")
            self.user = config.get("user", "postgres")
            self.password = config.get("password", "")
            
            # 🛠️ 密碼編碼處理
            if self.password:
                encoded_password = quote_plus(self.password)
                self.connection_string = f"postgresql://{self.user}:{encoded_password}@{self.host}:{self.port}/{self.database}"
            else:
                self.connection_string = f"postgresql://{self.user}@{self.host}:{self.port}/{self.database}"
        
        # 確保 SSL 配置
        if "sslmode=" not in self.connection_string:
            separator = "&" if "?" in self.connection_string else "?"
            self.connection_string += f"{separator}sslmode=prefer"
        
        self.schema = config.get("schema", "public")
        self.connect_timeout = config.get("connect_timeout", 30)
        
        # 簡化連接管理（不使用連接池）
        self._local = threading.local()
        
    def connect(self):
        """創建 PostgreSQL 連接"""
        try:
            self.connection = psycopg2.connect(
                self.connection_string,
                connect_timeout=self.connect_timeout
            )
            self.connection.autocommit = False
            
            # 設置搜索路徑
            with self.connection.cursor() as cursor:
                cursor.execute(f"SET search_path TO {self.schema}")
            
            self._is_connected = True
            logger.info(f"✅ PostgreSQL 連接成功")
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL 連接失敗: {e}")
            logger.error(f"   連接字符串: {self.connection_string[:50]}...")
            raise
    
    def get_connection(self):
        """獲取 PostgreSQL 連接"""
        if not self.connection or not self._is_connected or self.connection.closed:
            self.connect()
        return self.connection
    
    def disconnect(self):
        """關閉 PostgreSQL 連接"""
        if self.connection:
            try:
                if not self.connection.closed:
                    self.connection.close()
                self._is_connected = False
                logger.info("✅ PostgreSQL 連接已關閉")
            except Exception as e:
                logger.error(f"❌ 關閉連接失敗: {e}")
            finally:
                self.connection = None
    
    def _convert_sql_params(self, sql: str, params: Union[tuple, list] = None):
        """將 SQLite 風格的 SQL 轉換為 PostgreSQL 風格"""
        if params and '?' in sql:
            # 將所有 ? 參數轉換為 %s 格式（psycopg2 風格）
            sql = sql.replace('?', '%s')
        
        # 修復布爾值比較
        sql = sql.replace('= 1', '= TRUE')
        sql = sql.replace('= 0', '= FALSE')
        
        return sql, params
    
    def execute_query(self, sql: str, params: Union[tuple, list] = None) -> List[Dict[str, Any]]:
        """執行查詢 - PostgreSQL 正確實現"""
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
        """執行更新 - PostgreSQL 正確實現"""
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
                    conn.rollback()
                except:
                    pass
                raise
    
    def execute_insert(self, sql: str, params: Union[tuple, list] = None) -> Optional[int]:
        """執行插入 - PostgreSQL 正確實現"""
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
            raise ValueError(f"不支持的數據庫類型: {db_type}")
    
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
            # 🔧 優先使用 DATABASE_URL
            database_url = os.getenv("DATABASE_URL")
            if database_url:
                config = {
                    "connection_string": database_url,
                    "schema": os.getenv("POSTGRES_SCHEMA", "public"),
                    "connect_timeout": int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "30"))
                }
            else:
                config = {
                    "host": os.getenv("POSTGRES_HOST", "localhost"),
                    "port": int(os.getenv("POSTGRES_PORT", "5432")),
                    "database": os.getenv("POSTGRES_DATABASE", "chatbot_system"),
                    "user": os.getenv("POSTGRES_USER", "postgres"),
                    "password": os.getenv("POSTGRES_PASSWORD", ""),
                    "schema": os.getenv("POSTGRES_SCHEMA", "public"),
                    "connect_timeout": int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "30"))
                }
        else:
            raise ValueError(f"不支持的數據庫類型: {db_type}")
        
        return DatabaseFactory.create_adapter(db_type, config)

# ========== 測試函數 ==========
def test_corrected_adapter():
    """測試修復後的數據庫適配器"""
    print("🧪 測試修復後的數據庫適配器")
    print("=" * 60)
    
    # 設置環境變數
    os.environ["DATABASE_URL"] = "postgresql://postgres:hhpxxq6almxtdrzv1rvvz6cn37a0ec31@centerbeam.proxy.rlwy.net:42556/railway"
    
    # 1. 測試 PostgreSQL
    print("1️⃣ 測試 PostgreSQL 適配器...")
    try:
        pg_config = {
            "connection_string": os.environ["DATABASE_URL"]
        }
        pg_adapter = DatabaseFactory.create_adapter("postgresql", pg_config)
        pg_adapter.connect()
        
        # 基本查詢測試
        result = pg_adapter.execute_query("SELECT 1 as test, 'hello world' as message")
        print(f"✅ PostgreSQL 基本查詢: {result}")
        
        # pgvector 測試
        vector_result = pg_adapter.execute_query("SELECT '[1,2,3]'::vector as test_vector")
        print(f"✅ pgvector 測試: {vector_result}")
        
        pg_adapter.disconnect()
        print("✅ PostgreSQL 測試完成")
        
    except Exception as e:
        print(f"❌ PostgreSQL 測試失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 測試 SQLite
    print("\n2️⃣ 測試 SQLite 適配器...")
    try:
        sqlite_config = {
            "db_file": "test_corrected.db"
        }
        sqlite_adapter = DatabaseFactory.create_adapter("sqlite", sqlite_config)
        sqlite_adapter.connect()
        
        # 創建測試表
        sqlite_adapter.execute_update("""
            CREATE TABLE IF NOT EXISTS test_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 插入測試
        insert_id = sqlite_adapter.execute_insert(
            "INSERT INTO test_table (name) VALUES (?)",
            ("test_user",)
        )
        print(f"✅ SQLite 插入測試，ID: {insert_id}")
        
        # 查詢測試
        results = sqlite_adapter.execute_query("SELECT * FROM test_table")
        print(f"✅ SQLite 查詢測試: {len(results)} 條記錄")
        
        sqlite_adapter.disconnect()
        
        # 清理測試文件
        if os.path.exists("test_corrected.db"):
            os.remove("test_corrected.db")
        
        print("✅ SQLite 測試完成")
        
    except Exception as e:
        print(f"❌ SQLite 測試失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 測試從環境變數創建
    print("\n3️⃣ 測試從環境變數創建適配器...")
    try:
        os.environ["DB_TYPE"] = "postgresql"
        env_adapter = DatabaseFactory.create_from_env()
        env_adapter.connect()
        
        result = env_adapter.execute_query("SELECT 'from_env' as source")
        print(f"✅ 環境變數適配器: {result}")
        
        env_adapter.disconnect()
        print("✅ 環境變數測試完成")
        
    except Exception as e:
        print(f"❌ 環境變數測試失敗: {e}")
    
    print("\n📊 測試總結:")
    print("如果所有測試都通過，說明數據庫適配器已完全修復")

if __name__ == "__main__":
    test_corrected_adapter()
# ========== SQLDialect �ɤB ==========
from enum import Enum

class SQLDialect(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    
    @classmethod
    def from_string(cls, dialect_str: str):
        dialect_str = dialect_str.lower().strip()
        if dialect_str in ["sqlite", "sqlite3"]:
            return cls.SQLITE
        elif dialect_str in ["postgresql", "postgres"]:
            return cls.POSTGRESQL
        else:
            return cls.SQLITE
    
    @staticmethod
    def get_auto_increment_column(db_type: str) -> str:
        if db_type.lower() == "sqlite":
            return "INTEGER PRIMARY KEY AUTOINCREMENT"
        elif db_type.lower() in ["postgresql", "postgres"]:
            return "SERIAL PRIMARY KEY"
        else:
            return "INTEGER PRIMARY KEY AUTOINCREMENT"
    
    @staticmethod
    def get_timestamp_column(db_type: str) -> str:
        if db_type.lower() == "sqlite":
            return "TEXT DEFAULT CURRENT_TIMESTAMP"
        elif db_type.lower() in ["postgresql", "postgres"]:
            return "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        else:
            return "TEXT DEFAULT CURRENT_TIMESTAMP"
    
    @staticmethod
    def get_boolean_column(db_type: str, default_value: bool = False) -> str:
        if db_type.lower() == "sqlite":
            default_val = "1" if default_value else "0"
            return f"INTEGER DEFAULT {default_val}"
        elif db_type.lower() in ["postgresql", "postgres"]:
            default_val = "TRUE" if default_value else "FALSE"
            return f"BOOLEAN DEFAULT {default_val}"
        else:
            default_val = "1" if default_value else "0"
            return f"INTEGER DEFAULT {default_val}"
