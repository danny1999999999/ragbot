#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auth_middleware.py - 統一認證中間件
修正版：向後兼容，修復舊用戶登入問題
"""

import os
import jwt
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Union
from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 認證配置 ==========
class AuthConfig:
    """統一的認證配置類 - 所有認證相關參數在此定義"""
    
    # JWT 配置
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24小時
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))  # 7天
    
    # Cookie 配置 - 統一所有服務的 Cookie 設置
    COOKIE_CONFIG = {
        "key": "auth_token",           # Cookie 名稱
        "path": "/",                   # Cookie 路徑 - 全站有效
        "domain": None,                # Cookie 域名 - None 表示當前域名
        "max_age": 86400,             # 24小時 (秒)
        "httponly": True,             # 防止 XSS 攻擊
        "secure": os.getenv("ENVIRONMENT", "development") == "production",  # 生產環境使用 HTTPS
        "samesite": "lax"             # CSRF 保護
    }
    
    # 認證來源優先級 - 按順序嘗試
    AUTH_SOURCES = [
        "cookie",           # 1. 優先使用 Cookie 中的 JWT
        "header",          # 2. Authorization Header
        "query"            # 3. URL 查詢參數 (僅限特殊情況)
    ]
    
    # 權限等級配置
    ROLE_HIERARCHY = {
        "super_admin": 100,
        "admin": 80,
        "operator": 60,
        "user": 40,
        "guest": 20,
        "anonymous": 0
    }
    
    # 環境配置
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG_AUTH = os.getenv("DEBUG_AUTH", "false").lower() == "true"

# 初始化配置
auth_config = AuthConfig()

# 打印配置信息（僅開發環境）
if auth_config.DEBUG_AUTH:
    logger.info("🔐 認證中間件配置:")
    logger.info(f"  JWT 過期時間: {auth_config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES} 分鐘")
    logger.info(f"  Cookie 配置: {auth_config.COOKIE_CONFIG}")
    logger.info(f"  認證來源: {auth_config.AUTH_SOURCES}")
    logger.info(f"  環境: {auth_config.ENVIRONMENT}")

# ========== 用戶管理整合 ==========
USER_MANAGER_AVAILABLE = False
user_manager = None

try:
    from user_manager import user_manager as um, User as UserClass
    if um is not None and hasattr(um, 'get_user_by_id'):
        user_manager = um
        User = UserClass
        USER_MANAGER_AVAILABLE = True
        logger.info("✅ user_manager 載入成功")
    else:
        raise ImportError("user_manager 不完整")
except (ImportError, AttributeError) as e:
    logger.warning(f"⚠️ user_manager 載入失敗: {e}")
    USER_MANAGER_AVAILABLE = False
    
    # 備用 User 類別
    class User:
        def __init__(self, id: int = None, username: str = "anonymous", 
                     role: str = "anonymous", email: str = "", is_active: bool = True):
            self.id = id
            self.username = username
            self.role = role
            self.email = email
            self.is_active = is_active
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                "id": self.id,
                "username": self.username,
                "role": self.role,
                "email": self.email,
                "is_active": self.is_active
            }

# ========== JWT 工具函數 ==========
class JWTManager:
    """JWT Token 管理器"""
    
    @staticmethod
    def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
        """創建 Access Token"""
        try:
            if expires_delta is None:
                expires_delta = timedelta(minutes=auth_config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
            
            expire = datetime.now(timezone.utc) + expires_delta
            
            payload = {
                "sub": str(user.id),  # 用戶ID
                "username": user.username,
                "role": user.role,
                "email": user.email,
                "exp": expire,
                "iat": datetime.now(timezone.utc),
                "type": "access"  # 新增 Token 類型
            }
            
            token = jwt.encode(
                payload, 
                auth_config.JWT_SECRET_KEY, 
                algorithm=auth_config.JWT_ALGORITHM
            )
            
            if auth_config.DEBUG_AUTH:
                logger.info(f"🔑 創建 JWT Token: 用戶 {user.username}, 過期時間 {expire}")
            
            return token
            
        except Exception as e:
            logger.error(f"❌ 創建 JWT Token 失敗: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token 生成失敗"
            )
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """驗證 JWT Token - 向後兼容版本"""
        try:
            payload = jwt.decode(
                token,
                auth_config.JWT_SECRET_KEY,
                algorithms=[auth_config.JWT_ALGORITHM]
            )
            
            # 🔧 修復：Token 類型檢查改為可選（向後兼容）
            # 只有當 Token 明確有 type 欄位且不是 access 時才拒絕
            if "type" in payload and payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="無效的 Token 類型"
                )
            
            if auth_config.DEBUG_AUTH:
                logger.info(f"✅ JWT Token 驗證成功: 用戶 {payload.get('username')}")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            if auth_config.DEBUG_AUTH:
                logger.warning("⏰ JWT Token 已過期")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token 已過期，請重新登入",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            if auth_config.DEBUG_AUTH:
                logger.warning(f"🚫 JWT Token 無效: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="無效的 Token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            logger.error(f"❌ Token 驗證異常: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token 驗證失敗"
            )

# ========== 認證提取器 ==========
class AuthExtractor:
    """從不同來源提取認證信息"""
    
    @staticmethod
    def extract_from_cookie(request: Request) -> Optional[str]:
        """從 Cookie 提取 Token"""
        try:
            token = request.cookies.get(auth_config.COOKIE_CONFIG["key"])
            if token and auth_config.DEBUG_AUTH:
                logger.info(f"🍪 從 Cookie 提取 Token: {token[:20]}...")
            return token
        except Exception as e:
            if auth_config.DEBUG_AUTH:
                logger.warning(f"🍪 Cookie 提取失敗: {e}")
            return None
    
    @staticmethod
    def extract_from_header(request: Request) -> Optional[str]:
        """從 Authorization Header 提取 Token"""
        try:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]  # 移除 "Bearer " 前綴
                if auth_config.DEBUG_AUTH:
                    logger.info(f"📤 從 Header 提取 Token: {token[:20]}...")
                return token
            return None
        except Exception as e:
            if auth_config.DEBUG_AUTH:
                logger.warning(f"📤 Header 提取失敗: {e}")
            return None
    
    @staticmethod
    def extract_from_query(request: Request) -> Optional[str]:
        """從查詢參數提取 Token (僅限特殊情況)"""
        try:
            token = request.query_params.get("token")
            if token and auth_config.DEBUG_AUTH:
                logger.info(f"🔗 從查詢參數提取 Token: {token[:20]}...")
            return token
        except Exception as e:
            if auth_config.DEBUG_AUTH:
                logger.warning(f"🔗 查詢參數提取失敗: {e}")
            return None
    
    @staticmethod
    def extract_token(request: Request) -> Optional[str]:
        """按優先級從多個來源提取 Token"""
        for source in auth_config.AUTH_SOURCES:
            try:
                if source == "cookie":
                    token = AuthExtractor.extract_from_cookie(request)
                elif source == "header":
                    token = AuthExtractor.extract_from_header(request)
                elif source == "query":
                    token = AuthExtractor.extract_from_query(request)
                else:
                    continue
                
                if token:
                    if auth_config.DEBUG_AUTH:
                        logger.info(f"✅ 成功從 {source} 提取 Token")
                    return token
                    
            except Exception as e:
                if auth_config.DEBUG_AUTH:
                    logger.warning(f"⚠️ 從 {source} 提取 Token 失敗: {e}")
                continue
        
        if auth_config.DEBUG_AUTH:
            logger.info("🔍 未找到任何有效的 Token")
        return None

# ========== 核心認證函數 ==========
async def get_user_from_request(request: Request) -> Optional[User]:
    """
    從請求中獲取用戶信息 - 核心認證函數
    🔧 修復版：改善錯誤處理，確保向後兼容
    """
    try:
        # 1. 提取 Token
        token = AuthExtractor.extract_token(request)
        if not token:
            if auth_config.DEBUG_AUTH:
                logger.info("🔍 未找到 Token，返回匿名用戶")
            return None
        
        # 2. 驗證 Token
        payload = JWTManager.verify_token(token)
        
        # 3. 獲取用戶信息
        if USER_MANAGER_AVAILABLE and user_manager:
            try:
                # 使用 user_manager 驗證用戶
                user_id = int(payload.get("sub"))
                user = user_manager.get_user_by_id(user_id)
                
                if not user:
                    logger.warning(f"⚠️ 用戶 ID {user_id} 不存在")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="用戶不存在"
                    )
                
                if hasattr(user, 'is_active') and not user.is_active:
                    logger.warning(f"⚠️ 用戶 {user.username} 已被停用")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="用戶已被停用"
                    )
                
                if auth_config.DEBUG_AUTH:
                    logger.info(f"✅ 用戶認證成功: {user.username} ({getattr(user, 'role', 'user')})")
                
                return user
                
            except Exception as e:
                logger.warning(f"⚠️ user_manager 認證失敗，使用備用認證: {e}")
                # 如果 user_manager 失敗，回退到備用認證
        
        # 備用認證 - 從 Token payload 創建用戶對象
        user = User(
            id=int(payload.get("sub", 0)),
            username=payload.get("username", "unknown"),
            role=payload.get("role", "user"),
            email=payload.get("email", ""),
            is_active=True
        )
        
        if auth_config.DEBUG_AUTH:
            logger.info(f"✅ 備用認證成功: {user.username} ({user.role})")
        
        return user
            
    except HTTPException:
        # 重新拋出 HTTP 異常
        raise
    except Exception as e:
        logger.error(f"❌ 用戶認證異常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="認證系統異常"
        )

# ========== 權限檢查函數 ==========
def check_role_permission(user: User, required_role: str) -> bool:
    """檢查用戶角色權限"""
    if not user:
        return False
    
    user_level = auth_config.ROLE_HIERARCHY.get(user.role, 0)
    required_level = auth_config.ROLE_HIERARCHY.get(required_role, 100)
    
    has_permission = user_level >= required_level
    
    if auth_config.DEBUG_AUTH:
        logger.info(f"🔐 權限檢查: {user.username} ({user.role}:{user_level}) vs 需要 ({required_role}:{required_level}) = {has_permission}")
    
    return has_permission

def check_user_permissions(user: User, required_permissions: list) -> bool:
    """檢查用戶特定權限 (可擴展)"""
    # 超級管理員擁有所有權限
    if user and user.role == "super_admin":
        return True
    
    # 這裡可以實現更複雜的權限邏輯
    return False

# ========== FastAPI 依賴函數 ==========
async def get_current_user_optional(request: Request) -> Optional[User]:
    """可選的用戶認證 - 支援匿名訪問"""
    try:
        return await get_user_from_request(request)
    except HTTPException:
        # 認證失敗時返回 None，而不是拋出異常
        return None
    except Exception as e:
        logger.error(f"❌ 可選認證異常: {e}")
        return None

async def get_current_user_required(request: Request) -> User:
    """必需的用戶認證 - 必須登入"""
    user = await get_user_from_request(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="請先登入",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

async def get_admin_user(request: Request) -> User:
    """管理員權限認證"""
    user = await get_current_user_required(request)
    
    if not check_role_permission(user, "admin"):
        logger.warning(f"⚠️ 權限不足: {user.username} ({user.role}) 嘗試訪問管理功能")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理員權限"
        )
    
    return user

async def get_super_admin_user(request: Request) -> User:
    """超級管理員權限認證"""
    user = await get_current_user_required(request)
    
    if not check_role_permission(user, "super_admin"):
        logger.warning(f"⚠️ 權限不足: {user.username} ({user.role}) 嘗試訪問超級管理功能")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要超級管理員權限"
        )
    
    return user

# ========== 認證響應工具 ==========
class AuthResponse:
    """認證響應工具類"""
    
    @staticmethod
    def create_login_response(user: User, token: str) -> JSONResponse:
        """
        創建登入成功響應，包含統一的 Cookie 設置
        
        Args:
            user: 用戶對象
            token: JWT Token
            
        Returns:
            JSONResponse 包含用戶信息和正確的 Cookie
        """
        try:
            response_data = {
                "success": True,
                "message": "登入成功",
                "user": user.to_dict(),
                "token": token
            }
            
            response = JSONResponse(content=response_data)
            
            # 設置統一的 Cookie
            response.set_cookie(
                key=auth_config.COOKIE_CONFIG["key"],
                value=token,
                path=auth_config.COOKIE_CONFIG["path"],
                domain=auth_config.COOKIE_CONFIG["domain"],
                max_age=auth_config.COOKIE_CONFIG["max_age"],
                httponly=auth_config.COOKIE_CONFIG["httponly"],
                secure=auth_config.COOKIE_CONFIG["secure"],
                samesite=auth_config.COOKIE_CONFIG["samesite"]
            )
            
            if auth_config.DEBUG_AUTH:
                logger.info(f"🍪 設置登入 Cookie: {auth_config.COOKIE_CONFIG}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ 創建登入響應失敗: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="登入響應創建失敗"
            )
    
    @staticmethod
    def create_logout_response() -> JSONResponse:
        """
        創建登出響應，清除 Cookie
        
        Returns:
            JSONResponse 並清除認證 Cookie
        """
        try:
            response_data = {
                "success": True,
                "message": "登出成功"
            }
            
            response = JSONResponse(content=response_data)
            
            # 清除 Cookie
            response.delete_cookie(
                key=auth_config.COOKIE_CONFIG["key"],
                path=auth_config.COOKIE_CONFIG["path"],
                domain=auth_config.COOKIE_CONFIG["domain"]
            )
            
            if auth_config.DEBUG_AUTH:
                logger.info("🍪 清除登出 Cookie")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ 創建登出響應失敗: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="登出響應創建失敗"
            )

# ========== 便捷函數別名 ==========
# 為了向後兼容和使用方便，提供簡短的別名

# 認證依賴
OptionalAuth = get_current_user_optional  # 可選認證
RequiredAuth = get_current_user_required  # 必需認證  
AdminAuth = get_admin_user               # 管理員認證
SuperAdminAuth = get_super_admin_user    # 超級管理員認證

# 工具函數
jwt_auth = JWTManager()
auth_response = AuthResponse()

# 向後兼容的函數名
async def get_current_active_user(request: Request) -> User:
    """向後兼容的函數名"""
    return await get_current_user_required(request)

async def get_user_from_cookie(request: Request) -> Optional[User]:
    """向後兼容的函數名 - 僅從 Cookie 獲取用戶"""
    try:
        token = AuthExtractor.extract_from_cookie(request)
        if not token:
            return None
        
        payload = JWTManager.verify_token(token)
        
        if USER_MANAGER_AVAILABLE:
            user_id = int(payload.get("sub"))
            return user_manager.get_user_by_id(user_id)
        else:
            return User(
                id=int(payload.get("sub")),
                username=payload.get("username", "unknown"),
                role=payload.get("role", "user"),
                email=payload.get("email", ""),
                is_active=True
            )
    except Exception:
        return None

# ========== 模塊測試 ==========
def test_auth_config():
    """測試認證配置"""
    print("🧪 認證中間件配置測試")
    print("=" * 50)
    print(f"JWT Secret Key: {auth_config.JWT_SECRET_KEY[:10]}...")
    print(f"JWT 過期時間: {auth_config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES} 分鐘")
    print(f"Cookie 配置: {auth_config.COOKIE_CONFIG}")
    print(f"認證來源: {auth_config.AUTH_SOURCES}")
    print(f"角色層級: {auth_config.ROLE_HIERARCHY}")
    print(f"用戶管理器: {'可用' if USER_MANAGER_AVAILABLE else '不可用'}")
    print("=" * 50)

if __name__ == "__main__":
    test_auth_config()