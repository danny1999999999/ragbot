#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
user_manager.py - 增強版用戶管理器
支持數據庫抽象層，同時保持完全向後兼容
"""

import hashlib
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager
import threading

# 🔧 嘗試導入抽象層，如果失敗則使用傳統模式
ADAPTER_AVAILABLE = False
try:
    from database_adapter import DatabaseAdapter, DatabaseFactory, SQLDialect
    ADAPTER_AVAILABLE = True
    print("✅ 數據庫抽象層可用")
except ImportError:
    print("📦 數據庫抽象層不可用，使用傳統 SQLite 模式")

# 安全的 print 函數
def safe_print(*args, **kwargs):
    """安全的 print 函數，處理編碼問題"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        import sys
        try:
            for arg in args:
                sys.stdout.buffer.write(str(arg).encode('utf-8', errors='replace'))
                sys.stdout.buffer.write(b' ')
            sys.stdout.buffer.write(b'\n')
            sys.stdout.flush()
        except:
            safe_args = [str(arg).encode('ascii', errors='replace').decode('ascii') for arg in args]
            print(*safe_args, **kwargs)

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from config import app_config  # ⭐ 統一導入


@dataclass
class User:
    """用戶數據類 - 完全向後兼容"""
    username: str
    email: str
    password_hash: str
    role: str
    id: int = 0
    is_active: bool = True
    created_at: str = ""
    last_login: str = ""
    failed_attempts: int = 0  # 對應資料庫的 failed_attempts
    locked_until: str = ""    # 對應資料庫的 locked_until
    
    # 為了向下相容，提供 login_attempts 屬性
    @property
    def login_attempts(self):
        return self.failed_attempts
    
    @login_attempts.setter
    def login_attempts(self, value):
        self.failed_attempts = value
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """將用戶數據轉換為字典，以便在 API 中使用"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active
        }

@dataclass
class Session:
    """Session 數據類"""
    id: int = 0
    user_id: int = 0
    token: str = ""
    created_at: str = ""
    expires_at: str = ""
    is_active: bool = True
    ip_address: str = ""
    user_agent: str = ""

@dataclass
class AuditLog:
    """審計日誌數據類"""
    id: int = 0
    user_id: int = 0
    action: str = ""
    details: str = ""
    ip_address: str = ""
    timestamp: str = ""

class CompatibleUserManager:
    """增強版用戶管理器 - 支持抽象層，完全向後兼容"""
    
    def __init__(self, db_file: str = "user_management.db"):
        """
        初始化用戶管理器
        
        Args:
            db_file: 數據庫文件路徑（僅在 SQLite 模式下使用）
        """
        self.db_file = db_file
        self.lock = threading.RLock()
        
        # 🔧 決定使用哪種模式 (新版邏輯：偵測到Railway環境時強制使用Adapter)
        is_railway = bool(os.getenv('RAILWAY_PROJECT_ID'))
        use_adapter_env = os.getenv("USE_DATABASE_ADAPTER", "false").lower() == "true"
        
        if (use_adapter_env or is_railway) and ADAPTER_AVAILABLE:
            self._init_with_adapter()
        else:
            self._init_traditional_sqlite()
    
    def _init_with_adapter(self):
        """使用抽象層初始化"""
        try:
            safe_print("🔄 使用數據庫抽象層模式初始化...")
            self.mode = "adapter"
            
            # 從環境變數創建適配器
            try:
                self.db = DatabaseFactory.create_from_env("user_management")
            except Exception as e:
                safe_print(f"⚠️ 無法從環境變數創建適配器，使用默認 SQLite: {e}")
                # 回退到 SQLite
                self.db = DatabaseFactory.create_adapter("sqlite", {
                    "db_file": self.db_file
                })
            
            # 連接並初始化數據庫
            self.db.connect()
            self._create_database_schema_adapter()
            self._verify_and_migrate_schema_adapter()
            self._create_default_admin()
            safe_print("✅ 用戶管理器初始化成功（抽象層模式）")
            
        except Exception as e:
            safe_print(f"❌ 抽象層初始化失敗，回退到傳統模式: {e}")
            self._init_traditional_sqlite()
    
    def _init_traditional_sqlite(self):
        """使用傳統 SQLite 初始化"""
        safe_print("📦 使用傳統 SQLite 模式初始化...")
        self.mode = "traditional"
        self.db = None
        
        # 如果資料庫不存在，創建一個
        if not os.path.exists(self.db_file):
            safe_print(f"⚠️ 資料庫檔案不存在，將創建新的資料庫: {self.db_file}")
            self._create_database_traditional()
        else:
            safe_print(f"✅ 連接到現有資料庫: {self.db_file}")
        
        self._verify_database_structure_traditional()
        safe_print("✅ 用戶管理器初始化成功（傳統模式）")
    
    # ========== 抽象層模式的方法 ==========
    
    def _get_db_type(self) -> str:
        """獲取數據庫類型（抽象層模式）"""
        if self.mode == "adapter":
            return type(self.db).__name__.replace("Adapter", "").lower()
        return "sqlite"
    
    def _create_database_schema_adapter(self):
        """創建數據庫模式（抽象層模式）"""
        try:
            db_type = self._get_db_type()
            
            if db_type == "sqlite":
                users_table_sql = '''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT DEFAULT 'user',
                        is_active BOOLEAN DEFAULT 1,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        last_login TEXT,
                        failed_attempts INTEGER DEFAULT 0,
                        locked_until TEXT
                    )
                '''
                
                sessions_table_sql = '''
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        token TEXT UNIQUE NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        expires_at TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT 1,
                        ip_address TEXT,
                        user_agent TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                '''
                
                audit_logs_table_sql = '''
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        details TEXT,
                        ip_address TEXT,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                '''
            
            elif db_type == "postgresql":
                users_table_sql = '''
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(255) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        role VARCHAR(50) DEFAULT 'user',
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login VARCHAR(255),
                        failed_attempts INTEGER DEFAULT 0,
                        locked_until VARCHAR(255)
                    )
                '''
                
                sessions_table_sql = '''
                    CREATE TABLE IF NOT EXISTS sessions (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        token VARCHAR(512) UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at VARCHAR(255) NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        ip_address VARCHAR(45),
                        user_agent TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                '''
                
                audit_logs_table_sql = '''
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        action VARCHAR(100) NOT NULL,
                        details TEXT,
                        ip_address VARCHAR(45),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                '''
            else:
                raise ValueError(f"不支援的數據庫類型: {db_type}")
            
            # 執行創建語句
            with self.db.transaction():
                self.db.execute_update(users_table_sql)
                self.db.execute_update(sessions_table_sql)
                self.db.execute_update(audit_logs_table_sql)
            
        except Exception as e:
            safe_print(f"❌ 創建數據庫模式失敗: {e}")
            raise
    
    def _verify_and_migrate_schema_adapter(self):
        """驗證並遷移數據庫模式（抽象層模式）"""
        try:
            # 檢查必要的表格
            required_tables = ['users', 'sessions', 'audit_logs']
            
            for table in required_tables:
                if not self.db.table_exists(table):
                    safe_print(f"⚠️ 表格 {table} 不存在，重新創建...")
                    self._create_database_schema_adapter()
                    break
            
            # 檢查用戶表的必要列
            user_columns = self.db.get_table_columns('users')
            required_columns = [
                'id', 'username', 'email', 'password_hash', 'role',
                'is_active', 'created_at', 'last_login', 'failed_attempts', 'locked_until'
            ]
            
            missing_columns = [col for col in required_columns if col not in user_columns]
            
            if missing_columns:
                safe_print(f"⚠️ 用戶表缺少列: {missing_columns}")
                self._add_missing_columns_adapter(missing_columns)
                
        except Exception as e:
            safe_print(f"❌ 數據庫模式驗證失敗: {e}")
    
    def _add_missing_columns_adapter(self, missing_columns: List[str]):
        """添加缺失的列（抽象層模式）"""
        try:
            db_type = self._get_db_type()
            column_definitions = {
                'failed_attempts': 'INTEGER DEFAULT 0',
                'locked_until': 'TEXT' if db_type == 'sqlite' else 'VARCHAR(255)',
                'last_login': 'TEXT' if db_type == 'sqlite' else 'VARCHAR(255)',
                'is_active': 'BOOLEAN DEFAULT 1' if db_type == 'sqlite' else 'BOOLEAN DEFAULT TRUE',
                'role': 'TEXT DEFAULT "user"' if db_type == 'sqlite' else 'VARCHAR(50) DEFAULT \'user\''
            }
            
            for column in missing_columns:
                if column in column_definitions:
                    try:
                        self.db.execute_update(f"ALTER TABLE users ADD COLUMN {column} {column_definitions[column]}")
                        safe_print(f"✅ 添加列: {column}")
                    except Exception as e:
                        safe_print(f"⚠️ 添加列 {column} 失敗: {e}")
        except Exception as e:
            safe_print(f"❌ 添加缺失列失敗: {e}")
    
    # ========== 傳統模式的方法 ==========
    
    def _create_database_traditional(self):
        """創建新的資料庫和表格（傳統模式）"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # 創建 users 表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT DEFAULT 'user',
                        is_active BOOLEAN DEFAULT 1,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        last_login TEXT,
                        failed_attempts INTEGER DEFAULT 0,
                        locked_until TEXT
                    )
                ''')
                
                # 創建 sessions 表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        token TEXT UNIQUE NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        expires_at TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT 1,
                        ip_address TEXT,
                        user_agent TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                
                # 創建 audit_logs 表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        details TEXT,
                        ip_address TEXT,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                
                conn.commit()
                
                # 創建默認管理員
                self._create_default_admin()
                
                safe_print("✅ 資料庫和表格創建成功")
                
        except Exception as e:
            safe_print(f"❌ 創建資料庫失敗: {e}")
            raise
    
    def _verify_database_structure_traditional(self):
        """驗證並修復資料庫結構（傳統模式）"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # 定義所有應有的欄位及其類型
                required_columns = {
                    'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
                    'username': 'TEXT UNIQUE NOT NULL',
                    'email': 'TEXT UNIQUE NOT NULL', 
                    'password_hash': 'TEXT NOT NULL',
                    'role': 'TEXT DEFAULT "user"',
                    'is_active': 'BOOLEAN DEFAULT 1',
                    'created_at': 'TEXT DEFAULT CURRENT_TIMESTAMP',
                    'last_login': 'TEXT',
                    'failed_attempts': 'INTEGER DEFAULT 0',
                    'locked_until': 'TEXT'
                }
                
                # 檢查現有欄位
                cursor.execute("PRAGMA table_info(users)")
                existing_columns = {column[1] for column in cursor.fetchall()}
                
                # 添加缺失的欄位
                for col_name, col_type in required_columns.items():
                    if col_name not in existing_columns:
                        safe_print(f"⚠️ 檢測到缺失的資料庫欄位，正在自動新增: {col_name}")
                        try:
                            cursor.execute(f'ALTER TABLE users ADD COLUMN {col_name} {col_type}')
                        except Exception as e:
                            safe_print(f"⚠️ 新增欄位 {col_name} 失敗: {e}")
                
                conn.commit()
                safe_print("✅ 資料庫結構驗證通過")
                
        except Exception as e:
            safe_print(f"❌ 資料庫結構驗證失敗: {e}")
    
    @contextmanager
    def _get_db_connection(self):
        """獲取資料庫連接（上下文管理器，傳統模式）"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # ========== 統一的公共方法 ==========
    
    def _create_default_admin(self):
        """創建默認管理員用戶"""
        try:
            # 檢查是否已有管理員
            if self.mode == "adapter":
                results = self.db.execute_query(
                    "SELECT COUNT(*) as count FROM users WHERE role IN ('admin', 'super_admin')"
                )
                admin_count = results[0]['count'] if results else 0
            else:
                with self._get_db_connection() as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM users WHERE role IN ('admin', 'super_admin')")
                    admin_count = cursor.fetchone()[0]
            
            if admin_count == 0:
                # 創建默認管理員
                admin_password = os.getenv("ADMIN_PASSWORD", "ggyyggyyggyy")
                admin_user = User(
                    username="admin",
                    email="admin@example.com",
                    password_hash=self.hash_password(admin_password),
                    role="super_admin",
                    is_active=True
                )
                
                user_id = self.create_user(admin_user)
                if user_id:
                    safe_print(f"✅ 默認管理員創建成功 (用戶名: admin, 密碼: {admin_password})")
                else:
                    safe_print("⚠️ 默認管理員創建失敗")
            else:
                safe_print("ℹ️ 已存在管理員用戶，跳過創建默認管理員")
                
        except Exception as e:
            safe_print(f"⚠️ 創建默認管理員失敗: {e}")
    
    def hash_password(self, password: str) -> str:
        """密碼雜湊 - 使用 atalantis 專用鹽值"""
        salt = "atalantis-salt-2025"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """驗證密碼"""
        return self.hash_password(password) == password_hash
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """根據用戶名獲取用戶"""
        with self.lock:
            try:
                if self.mode == "adapter":
                    results = self.db.execute_query(
                        'SELECT * FROM users WHERE username = ?', (username,)
                    )
                    if results:
                        user_data = results[0]
                        return User(**user_data)
                else:
                    with self._get_db_connection() as conn:
                        row = conn.execute(
                            'SELECT * FROM users WHERE username = ?', (username,)
                        ).fetchone()
                        if row:
                            user_data = dict(row)
                            return User(**user_data)
                return None
                
            except Exception as e:
                safe_print(f"❌ 獲取用戶失敗: {e}")
                return None
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """根據 ID 獲取用戶"""
        with self.lock:
            try:
                if self.mode == "adapter":
                    results = self.db.execute_query(
                        'SELECT * FROM users WHERE id = ?', (user_id,)
                    )
                    if results:
                        user_data = results[0]
                        return User(**user_data)
                else:
                    with self._get_db_connection() as conn:
                        row = conn.execute(
                            'SELECT * FROM users WHERE id = ?', (user_id,)
                        ).fetchone()
                        if row:
                            user_data = dict(row)
                            return User(**user_data)
                return None
                
            except Exception as e:
                safe_print(f"❌ 獲取用戶失敗: {e}")
                return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """根據郵箱獲取用戶"""
        with self.lock:
            try:
                if self.mode == "adapter":
                    results = self.db.execute_query(
                        'SELECT * FROM users WHERE email = ?', (email,)
                    )
                    if results:
                        user_data = results[0]
                        return User(**user_data)
                else:
                    with self._get_db_connection() as conn:
                        row = conn.execute(
                            'SELECT * FROM users WHERE email = ?', (email,)
                        ).fetchone()
                        if row:
                            user_data = dict(row)
                            return User(**user_data)
                return None
                
            except Exception as e:
                safe_print(f"❌ 獲取用戶失敗: {e}")
                return None
    
    def create_user(self, user: User) -> Optional[int]:
        """創建新用戶"""
        with self.lock:
            try:
                # 檢查用戶名和郵箱是否已存在
                if self.get_user_by_username(user.username) or self.get_user_by_email(user.email):
                    return None
                
                user.created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                if self.mode == "adapter":
                    user_id = self.db.execute_insert('''
                        INSERT INTO users (username, email, password_hash, role, is_active, created_at, failed_attempts)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (user.username, user.email, user.password_hash, user.role, 
                          user.is_active, user.created_at, user.failed_attempts))
                else:
                    with self._get_db_connection() as conn:
                        cursor = conn.execute('''
                            INSERT INTO users (username, email, password_hash, role, is_active, created_at, failed_attempts)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (user.username, user.email, user.password_hash, user.role, 
                              user.is_active, user.created_at, user.failed_attempts))
                        user_id = cursor.lastrowid
                        conn.commit()
                
                if user_id:
                    user.id = user_id
                    self.log_audit(user.id, "user_created", f"新用戶註冊: {user.username}")
                
                return user_id
                
            except Exception as e:
                safe_print(f"❌ 創建用戶失敗: {e}")
                return None
    
    def update_user(self, user_id: int, updates: Dict[str, Any]) -> bool:
        """更新用戶信息"""
        with self.lock:
            try:
                # 構建動態 UPDATE 語句
                set_clauses = []
                values = []
                
                # 映射字段名稱（login_attempts -> failed_attempts）
                field_mapping = {
                    'login_attempts': 'failed_attempts'
                }
                
                allowed_fields = [
                    'username', 'email', 'password_hash', 'role', 'is_active', 
                    'last_login', 'failed_attempts', 'locked_until'
                ]
                
                for key, value in updates.items():
                    db_field = field_mapping.get(key, key)
                    if db_field in allowed_fields:
                        set_clauses.append(f"{db_field} = ?")
                        values.append(value)
                
                if not set_clauses:
                    return False
                
                values.append(user_id)
                
                if self.mode == "adapter":
                    affected_rows = self.db.execute_update(
                        f"UPDATE users SET {', '.join(set_clauses)} WHERE id = ?",
                        values
                    )
                else:
                    with self._get_db_connection() as conn:
                        cursor = conn.execute(
                            f"UPDATE users SET {', '.join(set_clauses)} WHERE id = ?",
                            values
                        )
                        affected_rows = cursor.rowcount
                        conn.commit()
                
                if affected_rows > 0:
                    self.log_audit(user_id, "user_updated", f"用戶資料更新: {list(updates.keys())}")
                    return True
                
                return False
                
            except Exception as e:
                safe_print(f"❌ 更新用戶失敗: {e}")
                return False
    
    def delete_user(self, user_id: int) -> bool:
        """刪除用戶（軟刪除 - 設為停用）"""
        try:
            return self.update_user(user_id, {"is_active": False})
        except Exception as e:
            safe_print(f"❌ 刪除用戶失敗: {e}")
            return False
    
    def authenticate(self, username: str, password: str, ip_address: str = "") -> tuple[bool, str, Optional[User]]:
        """用戶認證"""
        user = self.get_user_by_username(username)
        
        if not user:
            self.log_audit(0, "login_failed", f"用戶不存在: {username}", ip_address)
            return False, "用戶不存在", None
        
        if not user.is_active:
            self.log_audit(user.id, "login_failed", f"帳戶已停用", ip_address)
            return False, "帳戶已被停用", None
        
        # 檢查是否被鎖定
        if user.locked_until:
            try:
                locked_until = datetime.fromisoformat(user.locked_until.replace('Z', '+00:00'))
                if datetime.now() < locked_until:
                    self.log_audit(user.id, "login_failed", f"帳戶被鎖定", ip_address)
                    return False, f"帳戶被鎖定至 {user.locked_until}", None
            except:
                pass
        
        if not self.verify_password(password, user.password_hash):
            # 增加登入失敗次數
            new_attempts = user.failed_attempts + 1
            updates = {"failed_attempts": new_attempts}
            
            if new_attempts >= 5:
                # 鎖定帳戶1小時
                locked_until = (datetime.now() + timedelta(hours=1)).isoformat()
                updates["locked_until"] = locked_until
                updates["is_active"] = False
                self.log_audit(user.id, "account_locked", f"登入失敗次數過多", ip_address)
            
            self.update_user(user.id, updates)
            return False, "密碼錯誤", None
        
        # 登入成功
        self.update_user(user.id, {
            "last_login": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "failed_attempts": 0,
            "locked_until": None,
            "is_active": True
        })
        
        # 生成 token
        token = self._generate_token(user, ip_address)
        
        self.log_audit(user.id, "login_success", f"成功登入", ip_address)
        
        return True, token, user
    
    def _generate_token(self, user: User, ip_address: str = "") -> str:
        """生成 session token"""
        try:
            import jwt
            
            secret_key = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-change-this")
            
            # 生成 JWT token
            token_data = {
                "user_id": user.id,
                "username": user.username,
                "role": user.role,
                "iat": datetime.utcnow().timestamp(),
                "exp": (datetime.utcnow() + timedelta(hours=24)).timestamp()
            }
            
            token = jwt.encode(token_data, secret_key, algorithm="HS256")
            
            # 保存到 sessions 表
            expires_at = (datetime.now() + timedelta(hours=24)).isoformat()
            
            if self.mode == "adapter":
                self.db.execute_insert('''
                    INSERT INTO sessions (user_id, token, created_at, expires_at, ip_address)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user.id, token, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), expires_at, ip_address))
            else:
                with self._get_db_connection() as conn:
                    conn.execute('''
                        INSERT INTO sessions (user_id, token, created_at, expires_at, ip_address)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (user.id, token, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), expires_at, ip_address))
                    conn.commit()
            
            return token
            
        except ImportError:
            # 如果沒有 jwt 庫，生成簡單 token
            import uuid
            token = str(uuid.uuid4())
            
            expires_at = (datetime.now() + timedelta(hours=24)).isoformat()
            
            if self.mode == "adapter":
                self.db.execute_insert('''
                    INSERT INTO sessions (user_id, token, created_at, expires_at, ip_address)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user.id, token, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), expires_at, ip_address))
            else:
                with self._get_db_connection() as conn:
                    conn.execute('''
                        INSERT INTO sessions (user_id, token, created_at, expires_at, ip_address)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (user.id, token, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), expires_at, ip_address))
                    conn.commit()
            
            return token
    
    def validate_token(self, token: str) -> Optional[User]:
        """驗證 token 並返回用戶"""
        if not token:
            return None
        
        try:
            if self.mode == "adapter":
                results = self.db.execute_query('''
                    SELECT u.* FROM users u 
                    JOIN sessions s ON u.id = s.user_id
                    WHERE s.token = ? AND s.is_active = 1 AND s.expires_at > ?
                ''', (token, datetime.now().isoformat()))
                
                if results:
                    user_data = results[0]
                    return User(**user_data)
            else:
                with self._get_db_connection() as conn:
                    row = conn.execute('''
                        SELECT u.* FROM users u 
                        JOIN sessions s ON u.id = s.user_id
                        WHERE s.token = ? AND s.is_active = 1 AND s.expires_at > ?
                    ''', (token, datetime.now().isoformat())).fetchone()
                    
                    if row:
                        user_data = dict(row)
                        return User(**user_data)
            return None
        except Exception as e:
            safe_print(f"❌ 驗證 token 失敗: {e}")
            return None
    
    def logout(self, token: str) -> bool:
        """登出（使 token 失效）"""
        if not token:
            return False
        
        try:
            if self.mode == "adapter":
                affected_rows = self.db.execute_update(
                    'UPDATE sessions SET is_active = 0 WHERE token = ?',
                    (token,)
                )
            else:
                with self._get_db_connection() as conn:
                    cursor = conn.execute(
                        'UPDATE sessions SET is_active = 0 WHERE token = ?',
                        (token,)
                    )
                    affected_rows = cursor.rowcount
                    conn.commit()
            return affected_rows > 0
        except Exception as e:
            safe_print(f"❌ 登出失敗: {e}")
            return False
    
    def get_users(self, limit: int = 1000, offset: int = 0) -> List[User]:
        """獲取用戶列表 - 增加默認限制"""
        try:
            if self.mode == "adapter":
                results = self.db.execute_query(
                    'SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?',
                    (limit, offset)
                )
                return [User(**row) for row in results]
            else:
                with self._get_db_connection() as conn:
                    rows = conn.execute(
                        'SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?',
                        (limit, offset)
                    ).fetchall()
                    return [User(**dict(row)) for row in rows]
                    
        except Exception as e:
            safe_print(f"❌ 獲取用戶列表失敗: {e}")
            return []
    
    def get_total_users_count(self) -> int:
        """獲取用戶總數"""
        try:
            if self.mode == "adapter":
                results = self.db.execute_query('SELECT COUNT(*) as count FROM users')
                return results[0]['count'] if results else 0
            else:
                with self._get_db_connection() as conn:
                    return conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
        except Exception as e:
            safe_print(f"❌ 獲取用戶總數失敗: {e}")
            return 0
    
    def search_users(self, search_term: str, role_filter: str = "", limit: int = 100) -> List[User]:
        """搜索用戶"""
        try:
            query = 'SELECT * FROM users WHERE (username LIKE ? OR email LIKE ?)'
            params = [f'%{search_term}%', f'%{search_term}%']
            
            if role_filter:
                query += ' AND role = ?'
                params.append(role_filter)
            
            query += ' ORDER BY created_at DESC LIMIT ?'
            params.append(limit)
            
            if self.mode == "adapter":
                results = self.db.execute_query(query, params)
                return [User(**row) for row in results]
            else:
                with self._get_db_connection() as conn:
                    rows = conn.execute(query, params).fetchall()
                    return [User(**dict(row)) for row in rows]
                    
        except Exception as e:
            safe_print(f"❌ 搜索用戶失敗: {e}")
            return []
    
    def log_audit(self, user_id: int, action: str, details: str = "", ip_address: str = ""):
        """記錄審計日誌"""
        try:
            if self.mode == "adapter":
                self.db.execute_insert('''
                    INSERT INTO audit_logs (user_id, action, details, ip_address, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, action, details, ip_address, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            else:
                with self._get_db_connection() as conn:
                    conn.execute('''
                        INSERT INTO audit_logs (user_id, action, details, ip_address, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (user_id, action, details, ip_address, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    conn.commit()
        except Exception as e:
            safe_print(f"⚠️ 記錄審計日誌失敗: {e}")
    
    def get_audit_logs(self, user_id: Optional[int] = None, limit: int = 50) -> List[Dict]:
        """獲取審計日誌"""
        try:
            if user_id:
                query = 'SELECT * FROM audit_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?'
                params = (user_id, limit)
            else:
                query = 'SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?'
                params = (limit,)
            
            if self.mode == "adapter":
                return self.db.execute_query(query, params)
            else:
                with self._get_db_connection() as conn:
                    rows = conn.execute(query, params).fetchall()
                    return [dict(row) for row in rows]
                    
        except Exception as e:
            safe_print(f"❌ 獲取審計日誌失敗: {e}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """獲取系統統計"""
        try:
            if self.mode == "adapter":
                # 用戶統計
                user_stats = self.db.execute_query('''
                    SELECT 
                        COUNT(*) as total_users,
                        SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active_users,
                        SUM(CASE WHEN last_login IS NOT NULL THEN 1 ELSE 0 END) as users_with_login
                    FROM users
                ''')[0]
                
                # Session 統計
                session_results = self.db.execute_query('''
                    SELECT COUNT(*) as count FROM sessions 
                    WHERE is_active = 1 AND expires_at > ?
                ''', (datetime.now().isoformat(),))
                active_sessions = session_results[0]['count'] if session_results else 0
                
                # 今日登入
                today = datetime.now().date().strftime('%Y-%m-%d')
                today_login_results = self.db.execute_query('''
                    SELECT COUNT(*) as count FROM audit_logs 
                    WHERE action = 'login_success' AND DATE(timestamp) = ?
                ''', (today,))
                today_logins = today_login_results[0]['count'] if today_login_results else 0
                
                # 角色統計
                role_results = self.db.execute_query('''
                    SELECT role, COUNT(*) as count 
                    FROM users 
                    WHERE is_active = 1 
                    GROUP BY role
                ''')
                role_breakdown = {row['role']: row['count'] for row in role_results}
                
                storage_type = f"{self._get_db_type()}_with_adapter"
            else:
                # 傳統模式統計
                with self._get_db_connection() as conn:
                    # 用戶統計
                    user_stats = conn.execute('''
                        SELECT 
                            COUNT(*) as total_users,
                            SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active_users,
                            SUM(CASE WHEN last_login IS NOT NULL THEN 1 ELSE 0 END) as users_with_login
                        FROM users
                    ''').fetchone()
                    
                    # Session 統計
                    active_sessions = conn.execute('''
                        SELECT COUNT(*) FROM sessions 
                        WHERE is_active = 1 AND expires_at > ?
                    ''', (datetime.now().isoformat(),)).fetchone()[0]
                    
                    # 今日登入
                    today = datetime.now().date().strftime('%Y-%m-%d')
                    today_logins = conn.execute('''
                        SELECT COUNT(*) FROM audit_logs 
                        WHERE action = 'login_success' AND DATE(timestamp) = ?
                    ''', (today,)).fetchone()[0]
                    
                    # 角色統計
                    role_rows = conn.execute('''
                        SELECT role, COUNT(*) as count 
                        FROM users 
                        WHERE is_active = 1 
                        GROUP BY role
                    ''').fetchall()
                    role_breakdown = {row[0]: row[1] for row in role_rows}
                
                storage_type = "sqlite_enhanced"
            
            return {
                "total_users": user_stats['total_users'] if self.mode == "adapter" else user_stats[0],
                "active_users": user_stats['active_users'] if self.mode == "adapter" else user_stats[1],
                "inactive_users": (user_stats['total_users'] - user_stats['active_users']) if self.mode == "adapter" else (user_stats[0] - user_stats[1]),
                "users_with_login": user_stats['users_with_login'] if self.mode == "adapter" else user_stats[2],
                "active_sessions": active_sessions,
                "today_logins": today_logins,
                "role_breakdown": role_breakdown,
                "storage_type": storage_type,
                "mode": self.mode
            }
            
        except Exception as e:
            safe_print(f"❌ 獲取系統統計失敗: {e}")
            return {}
    
    def cleanup_expired_sessions(self):
        """清理過期的 session"""
        try:
            if self.mode == "adapter":
                affected_rows = self.db.execute_update(
                    'DELETE FROM sessions WHERE expires_at < ?',
                    (datetime.now().isoformat(),)
                )
            else:
                with self._get_db_connection() as conn:
                    cursor = conn.execute(
                        'DELETE FROM sessions WHERE expires_at < ?',
                        (datetime.now().isoformat(),)
                    )
                    affected_rows = cursor.rowcount
                    conn.commit()
            
            if affected_rows > 0:
                safe_print(f"🧹 清理了 {affected_rows} 個過期 session")
        except Exception as e:
            safe_print(f"❌ 清理過期 session 失敗: {e}")
    
    def has_permission(self, user: User, permission: str) -> bool:
        """檢查用戶權限"""
        if not user or not user.is_active:
            return False
        
        # 權限映射
        role_permissions = {
            "super_admin": ["*"],  # 所有權限
            "admin": ["manage_users", "view_conversations", "manage_bots", "manage_data", "view_stats"],
            "operator": ["view_conversations", "manage_bots", "view_stats"],
            "user": ["view_own_data"]
        }
        
        user_permissions = role_permissions.get(user.role, [])
        
        # super_admin 有所有權限
        if "*" in user_permissions:
            return True
        
        return permission in user_permissions
    
    def batch_update_users(self, user_ids: List[int], updates: Dict[str, Any]) -> int:
        """批量更新用戶"""
        success_count = 0
        with self.lock:
            for user_id in user_ids:
                if self.update_user(user_id, updates):
                    success_count += 1
        return success_count
    
    def get_user_statistics_by_role(self) -> Dict[str, int]:
        """按角色統計用戶數量"""
        try:
            if self.mode == "adapter":
                results = self.db.execute_query('''
                    SELECT role, COUNT(*) as count 
                    FROM users 
                    GROUP BY role
                ''')
                return {row['role']: row['count'] for row in results}
            else:
                with self._get_db_connection() as conn:
                    rows = conn.execute('''
                        SELECT role, COUNT(*) as count 
                        FROM users 
                        GROUP BY role
                    ''').fetchall()
                    return {row[0]: row[1] for row in rows}
        except Exception as e:
            safe_print(f"❌ 按角色統計失敗: {e}")
            return {}
    
    def get_recent_user_activity(self, days: int = 7) -> List[Dict]:
        """獲取最近用戶活動"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            if self.mode == "adapter":
                results = self.db.execute_query('''
                    SELECT 
                        u.username,
                        u.last_login,
                        COUNT(a.id) as activity_count
                    FROM users u
                    LEFT JOIN audit_logs a ON u.id = a.user_id 
                        AND DATE(a.timestamp) >= ?
                    WHERE u.is_active = 1
                    GROUP BY u.id, u.username, u.last_login
                    ORDER BY activity_count DESC, u.last_login DESC
                    LIMIT 20
                ''', (cutoff_date,))
                
                return [
                    {
                        "username": row['username'],
                        "last_login": row['last_login'],
                        "activity_count": row['activity_count']
                    }
                    for row in results
                ]
            else:
                with self._get_db_connection() as conn:
                    rows = conn.execute('''
                        SELECT 
                            u.username,
                            u.last_login,
                            COUNT(a.id) as activity_count
                        FROM users u
                        LEFT JOIN audit_logs a ON u.id = a.user_id 
                            AND DATE(a.timestamp) >= ?
                        WHERE u.is_active = 1
                        GROUP BY u.id, u.username, u.last_login
                        ORDER BY activity_count DESC, u.last_login DESC
                        LIMIT 20
                    ''', (cutoff_date,)).fetchall()
                    
                    return [
                        {
                            "username": row[0],
                            "last_login": row[1],
                            "activity_count": row[2]
                        }
                        for row in rows
                    ]
        except Exception as e:
            safe_print(f"❌ 獲取用戶活動失敗: {e}")
            return []
    
    def close(self):
        """關閉數據庫連接"""
        if self.mode == "adapter" and self.db:
            self.db.disconnect()

# 創建用戶管理器實例
try:
    user_manager = CompatibleUserManager(db_file="user_management.db")
    safe_print("✅ 增強版用戶管理器初始化成功")
except Exception as e:
    safe_print(f"❌ 用戶管理器初始化失敗: {e}")
    user_manager = None

# 導出
__all__ = ["User", "Session", "AuditLog", "CompatibleUserManager", "user_manager"]