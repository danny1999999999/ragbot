#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
user_manager.py - 全面重構為直接使用 PostgreSQL

- 移除所有 SQLite 和 database_adapter 的相關邏輯，消除複雜性和不穩定性。
- 強制使用 PostgreSQL，與系統其他部分保持一致，確保數據持久化。
- 使用 psycopg3 函式庫，提升性能和可靠性。
"""

import hashlib
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import psycopg # 使用現代的 psycopg3
import psycopg.rows

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Classes ---
@dataclass
class User:
    """用戶數據類"""
    id: int
    username: str
    email: str
    password_hash: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    failed_attempts: int
    locked_until: Optional[datetime]

    def to_dict(self) -> Dict[str, Any]:
        """將用戶數據轉換為 API 安全的字典格式"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active
        }

# --- User Manager --- 
class UserManager:
    """
    用戶管理器 - 直接使用 PostgreSQL
    """
    def __init__(self):
        """初始化用戶管理器，強制使用 PostgreSQL"""
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            error_msg = "環境變數 DATABASE_URL 未設定，使用者系統無法啟動。"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("🚀 初始化使用者管理器 (PostgreSQL-Only Mode)...")
        self._ensure_schema()
        self._create_default_admin()
        logger.info("✅ 使用者管理器初始化成功。")

    def _get_connection(self):
        """建立並返回一個新的 PostgreSQL 連接"""
        return psycopg.connect(self.database_url)

    def _ensure_schema(self):
        """確保 users 資料表存在且結構正確"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            id SERIAL PRIMARY KEY,
                            username VARCHAR(255) UNIQUE NOT NULL,
                            email VARCHAR(255) UNIQUE NOT NULL,
                            password_hash VARCHAR(255) NOT NULL,
                            role VARCHAR(50) DEFAULT 'user' NOT NULL,
                            is_active BOOLEAN DEFAULT TRUE NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
                            last_login TIMESTAMPTZ,
                            failed_attempts INTEGER DEFAULT 0 NOT NULL,
                            locked_until TIMESTAMPTZ
                        );
                    """)
                    # 可以在此處添加 ALTER TABLE 語句來更新舊的表結構
                    conn.commit()
                    logger.info("資料庫 'users' 表已準備就緒。")
        except Exception as e:
            logger.error(f"❌ 確保資料庫結構失敗: {e}")
            raise

    def _create_default_admin(self):
        """如果不存在任何管理員，則創建一個預設管理員"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM users WHERE role IN ('admin', 'super_admin')")
                    admin_count = cur.fetchone()[0]
                    
                    if admin_count == 0:
                        admin_password = os.getenv("ADMIN_PASSWORD", "ggyyggyyggyy")
                        hashed_password = self.hash_password(admin_password)
                        cur.execute("""
                            INSERT INTO users (username, email, password_hash, role, is_active)
                            VALUES (%s, %s, %s, %s, %s)
                        """, ('admin', 'admin@example.com', hashed_password, 'super_admin', True))
                        conn.commit()
                        logger.info(f"✅ 預設管理員已創建 (用戶名: admin, 密碼: {admin_password})")
        except Exception as e:
            logger.warning(f"⚠️ 創建預設管理員失敗: {e}")

    def hash_password(self, password: str) -> str:
        """密碼雜湊"""
        salt = "atalantis-salt-2025" # 保持與舊系統一樣的鹽
        return hashlib.sha256((password + salt).encode()).hexdigest()

    def verify_password(self, password: str, password_hash: str) -> bool:
        """驗證密碼"""
        return self.hash_password(password) == password_hash

    def _row_to_user(self, row: Dict[str, Any]) -> Optional[User]:
        """將資料庫行轉換為 User 物件"""
        if not row:
            return None
        return User(**row)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """根據用戶名獲取用戶"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
                    return self._row_to_user(cur.fetchone())
        except Exception as e:
            logger.error(f"❌ 根據用戶名獲取用戶失敗: {e}")
            return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """根據郵箱獲取用戶"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    cur.execute("SELECT * FROM users WHERE email = %s", (email,))
                    return self._row_to_user(cur.fetchone())
        except Exception as e:
            logger.error(f"❌ 根據郵箱獲取用戶失敗: {e}")
            return None

    def create_user(self, user_data: Dict[str, Any]) -> Optional[User]:
        """創建新用戶"""
        if not all(k in user_data for k in ['username', 'email', 'password']):
            raise ValueError("缺少必要的用戶數據：username, email, password")

        if self.get_user_by_username(user_data['username']) or self.get_user_by_email(user_data['email']):
            logger.warning(f"嘗試創建已存在的用戶: {user_data['username']}")
            return None

        hashed_password = self.hash_password(user_data['password'])
        role = user_data.get('role', 'user')
        is_active = user_data.get('is_active', True)

        try:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    cur.execute("""
                        INSERT INTO users (username, email, password_hash, role, is_active)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING *
                    """, (user_data['username'], user_data['email'], hashed_password, role, is_active))
                    new_user_row = cur.fetchone()
                    conn.commit()
                    logger.info(f"新用戶已創建: {user_data['username']}")
                    return self._row_to_user(new_user_row)
        except Exception as e:
            logger.error(f"❌ 創建用戶失敗: {e}")
            return None

    def update_user(self, user_id: int, updates: Dict[str, Any]) -> bool:
        """更新用戶信息"""
        if not updates:
            return False

        allowed_fields = ['username', 'email', 'password_hash', 'role', 'is_active', 'last_login', 'failed_attempts', 'locked_until']
        
        set_clauses = []
        values = []

        for key, value in updates.items():
            if key in allowed_fields:
                set_clauses.append(f"{key} = %s")
                values.append(value)
        
        if not set_clauses:
            return False

        values.append(user_id)

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"UPDATE users SET {', '.join(set_clauses)} WHERE id = %s", values)
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"❌ 更新用戶失敗 (ID: {user_id}): {e}")
            return False

    def delete_user(self, user_id: int) -> bool:
        """軟刪除用戶（設為停用）"""
        return self.update_user(user_id, {"is_active": False})

    def authenticate(self, username: str, password: str) -> tuple[bool, str, Optional[User]]:
        """用戶認證"""
        user = self.get_user_by_username(username)

        if not user:
            return False, "用戶不存在", None

        if not user.is_active:
            return False, "帳戶已被停用", None

        if user.locked_until and datetime.now(user.locked_until.tzinfo) < user.locked_until:
            return False, f"帳戶被鎖定至 {user.locked_until.strftime('%Y-%m-%d %H:%M')}", None

        if not self.verify_password(password, user.password_hash):
            new_attempts = user.failed_attempts + 1
            updates = {"failed_attempts": new_attempts}
            if new_attempts >= 5:
                locked_until_time = datetime.now() + timedelta(hours=1)
                updates["locked_until"] = locked_until_time
                logger.warning(f"用戶 {username} 因登入失敗次數過多而被鎖定。")
            self.update_user(user.id, updates)
            return False, "密碼錯誤", None

        # 登入成功
        self.update_user(user.id, {
            "last_login": datetime.now(),
            "failed_attempts": 0,
            "locked_until": None
        })
        return True, "登入成功", user

    def get_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """獲取用戶列表"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    cur.execute("SELECT * FROM users ORDER BY created_at DESC LIMIT %s OFFSET %s", (limit, offset))
                    return [self._row_to_user(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"❌ 獲取用戶列表失敗: {e}")
            return []

    def get_total_users_count(self) -> int:
        """獲取用戶總數"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM users")
                    return cur.fetchone()[0]
        except Exception as e:
            logger.error(f"❌ 獲取用戶總數失敗: {e}")
            return 0

    def search_users(self, search_term: str, limit: int = 100) -> List[User]:
        """搜索用戶"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    search_pattern = f"%{search_term}%"
                    cur.execute("""
                        SELECT * FROM users 
                        WHERE username ILIKE %s OR email ILIKE %s
                        ORDER BY created_at DESC LIMIT %s
                    """, (search_pattern, search_pattern, limit))
                    return [self._row_to_user(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"❌ 搜索用戶失敗: {e}")
            return []

# --- 實例化管理器 ---
try:
    user_manager = UserManager()
except Exception as e:
    logger.critical(f"🔥🔥🔥 無法初始化使用者管理器，應用程式可能無法正常處理使用者相關操作: {e}")
    user_manager = None
