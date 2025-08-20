#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_service_manager.py - 聊天機器人服務管理器 (修正版)

使用統一認證中間件的機器人服務管理器，與 simplified_admin_server.py 保持一致
"""

import os
import json
import subprocess
import sys
from pathlib import Path
import logging
from typing import Dict, IO, List, Optional
import threading
from datetime import datetime, timezone
import time
import hashlib
 
# 修正：在導入任何其他自訂模組前，首先載入 .env 檔案
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Response,Form 
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
from langchain_community.vectorstores.utils import filter_complex_metadata

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from config import app_config  # ⭐ 統一導入


USE_VECTOR_API = os.getenv("USE_VECTOR_API", "false").lower() == "true"
VECTOR_API_URL = os.getenv("VECTOR_API_URL", "http://localhost:9002")
BASE_PUBLIC_URL = os.getenv("BASE_PUBLIC_URL", "http://localhost:8000")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
GATEWAY_ADMIN_TOKEN = os.getenv("GATEWAY_ADMIN_TOKEN", "")

# 保留既有全域變數名稱，以免其他程式引用壞掉
try:
    if not USE_VECTOR_API:
        from vector_builder_langchain import OptimizedVectorSystem
        vector_system = OptimizedVectorSystem()
        VECTOR_SYSTEM_AVAILABLE = True
    else:
        vector_system = None
        VECTOR_SYSTEM_AVAILABLE = False
except Exception:
    vector_system = None
    VECTOR_SYSTEM_AVAILABLE = False


# 導入統一認證中間件 - 與 simplified_admin_server.py 完全一致
try:
    from auth_middleware import (
        OptionalAuth,      # 可選認證
        RequiredAuth,      # 必需認證
        AdminAuth,         # 管理員認證
        SuperAdminAuth,    # 超級管理員認證
        JWTManager,        # JWT 管理器
        AuthResponse,      # 認證響應工具
        User,              # 用戶類
        USER_MANAGER_AVAILABLE,  # 用戶管理器可用性
        auth_config        # 認證配置
    )
    AUTH_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ 統一認證中間件載入成功")
except ImportError as e:
    AUTH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"❌ 統一認證中間件載入失敗: {e}")

# 嘗試導入用戶管理（如果 auth_middleware 沒有成功導入）
if not USER_MANAGER_AVAILABLE:
    try:
        from user_manager import user_manager
        USER_MANAGER_AVAILABLE = True
        logger.info("✅ 直接從 user_manager 載入成功")
    except ImportError as e:
        logger.warning(f"❌ 用戶管理系統完全不可用: {e}")

# 嘗試導入向量系統
try:
    from vector_builder_langchain import OptimizedVectorSystem
    vector_system = OptimizedVectorSystem()
    VECTOR_SYSTEM_AVAILABLE = True
    logger.info("✅ 向量知識庫系統載入成功")
except ImportError as e:
    vector_system = None
    VECTOR_SYSTEM_AVAILABLE = False
    logger.warning(f"❌ 向量知識庫系統載入失敗: {e}")

# 配置日誌
logging.basicConfig(level=logging.INFO)


try:
    from conversation_logger_simple import create_logger_instance
    CONVERSATION_LOGGER_AVAILABLE = True
    logger.info("✅ PostgreSQL對話記錄器工廠函數載入成功")
except ImportError as e:
    CONVERSATION_LOGGER_AVAILABLE = False
    logger.error(f"❌ PostgreSQL對話記錄器載入失敗: {e}")

def get_database_config(bot_name: str) -> Dict:
    """
    獲取資料庫配置，優先使用 Railway 的 DATABASE_URL
    """
    # 首先檢查是否有 DATABASE_URL
    database_url = os.getenv("DATABASE_URL")
    if database_url and database_url.startswith("postgresql://"):
        print(f"✅ Bot '{bot_name}' 使用 Railway DATABASE_URL")
        # 確保有 SSL 參數
        if "sslmode=" not in database_url:
            separator = "&" if "?" in database_url else "?"
            database_url += f"{separator}sslmode=require"
        
        # 從 URL 解析元件以供返回
        from urllib.parse import urlparse
        parsed = urlparse(database_url)
        return {
            "type": "postgresql",
            "url": database_url, # 直接返回 URL
            "host": parsed.hostname,
            "port": parsed.port,
            "database": parsed.path[1:],
            "user": parsed.username,
            "password": parsed.password,
            "min_connections": int(os.getenv("POSTGRES_MIN_CONNECTIONS", "1")),
            "max_connections": int(os.getenv("POSTGRES_MAX_CONNECTIONS", "10")),
            "connect_timeout": int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "30")),
            "command_timeout": int(os.getenv("POSTGRES_COMMAND_TIMEOUT", "30"))
        }

    # 如果沒有 DATABASE_URL，則回退到舊的 SQLite 邏輯（適用於本地開發）
    print(f"⚠️ Bot '{bot_name}' 未找到 DATABASE_URL，回退到本地 SQLite")
    return {
        "type": "sqlite",
        "db_file": str(ROOT_DIR / f"{bot_name}_conversations.db"),
        "timeout": float(os.getenv("SQLITE_TIMEOUT", "30.0")),
        "journal_mode": os.getenv("SQLITE_JOURNAL_MODE", "WAL"),
        "synchronous": os.getenv("SQLITE_SYNCHRONOUS", "NORMAL"),
        "cache_size": int(os.getenv("SQLITE_CACHE_SIZE", "2000"))
    }


# 配置目錄
ROOT_DIR = Path(__file__).parent
BOT_CONFIGS_DIR = ROOT_DIR / "bot_configs"
BOT_DATA_DIR = ROOT_DIR / "data"
BOT_INSTANCE_SCRIPT = ROOT_DIR / "chatbot_instance.py"

# 確保目錄存在
BOT_CONFIGS_DIR.mkdir(exist_ok=True)
BOT_DATA_DIR.mkdir(exist_ok=True)

# 設定模板目錄
templates = Jinja2Templates(directory=str(ROOT_DIR))

# 全域狀態
global_bot_processes: Dict[str, subprocess.Popen] = {}

LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
global_bot_log_files: Dict[str, IO[str]] = {}

class BotServiceManager:
    """使用統一認證的機器人服務管理器"""
    
    def __init__(self):
        self.app = FastAPI(title="聊天機器人服務管理器")
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.use_vector_api = USE_VECTOR_API
        self.vector_api_url = VECTOR_API_URL
        
        # 🆕 添加連接池管理
        self._conversation_loggers = {}  # 連接池：{bot_name: logger_instance}
        self._logger_lock = threading.RLock()
        
        if self.use_vector_api:
            print(f"🔗 管理器使用向量API模式: {self.vector_api_url}")
        else:
            print("⚠️ 管理器使用直接模式")

        self.setup_routes()
        logger.info("✅ 使用統一認證的機器人服務管理器初始化完成")
        self.migrate_existing_records()

    def get_conversation_logger(self, bot_name: str):
        """獲取或創建對話記錄器（使用工廠函數 + 連接池）"""
        with self._logger_lock:
            if bot_name not in self._conversation_loggers:
                if not CONVERSATION_LOGGER_AVAILABLE:
                    raise ImportError("PostgreSQL對話記錄器不可用")
                
                try:
                    # ✅ 使用工廠函數創建實例
                    db_config = get_database_config(bot_name)
                    logger_instance = create_logger_instance(db_config)
                    
                    # 預先創建表格（工廠函數內部可能已經處理，但保險起見）
                    logger_instance.init_database()
                    
                    self._conversation_loggers[bot_name] = logger_instance
                    logger.info(f"✅ 為機器人 {bot_name} 創建對話記錄器（工廠+連接池模式）")
                except Exception as e:
                    logger.error(f"❌ 創建對話記錄器失敗 {bot_name}: {e}")
                    raise
            
            return self._conversation_loggers[bot_name]
    
    def close_all_loggers(self):
        """關閉所有連接（在服務關閉時調用）"""
        with self._logger_lock:
            for bot_name, logger_instance in self._conversation_loggers.items():
                try:
                    logger_instance.close()
                    logger.info(f"✅ 關閉機器人 {bot_name} 的對話記錄器")
                except Exception as e:
                    logger.error(f"❌ 關閉對話記錄器失敗 {bot_name}: {e}")
            self._conversation_loggers.clear()

    async def handle_get_conversations(self, bot_name: str, page: int = 1, limit: int = 20, search: str = ""):
        """獲取對話記錄列表 - 工廠函數+連接池版"""
        try:
            # ✅ 使用連接池+工廠函數獲取記錄器
            conv_logger = self.get_conversation_logger(bot_name)
            
            # 計算偏移量
            offset = (page - 1) * limit
            
            # 獲取對話記錄
            conversations, total = conv_logger.get_conversations(
                limit=limit, 
                offset=offset,
                search=search if search else None
            )
            
            # 處理數據格式，確保前端需要的字段
            formatted_conversations = []
            for conv in conversations:
                # 提取 chunk 信息 - 這部分邏輯保持不變
                chunk_ids = conv.get('chunk_ids', [])
                
                formatted_conversations.append({
                    'id': conv.get('id'),
                    'session_id': conv.get('user_id', 'unknown'),
                    'user_message': conv.get('user_query', '')[:200] + ('...' if len(conv.get('user_query', '')) > 200 else ''),
                    'bot_response': conv.get('ai_response', '')[:300] + ('...' if len(conv.get('ai_response', '')) > 300 else ''),
                    'chunk_ids': chunk_ids,
                    'timestamp': conv.get('timestamp'),
                    'created_at': conv.get('created_at'),
                    'collection_used': conv.get('collection_used'),
                    'processing_time': conv.get('processing_time_ms', 0),
                    'error_occurred': conv.get('error_occurred', False),
                    # 保留完整數據供詳細查看使用
                    'full_user_message': conv.get('user_query', ''),
                    'full_bot_response': conv.get('ai_response', ''),
                    'chunk_references': conv.get('chunk_references', [])
                })
            
            # 計算總頁數
            total_pages = (total + limit - 1) // limit if total > 0 else 1
            
            return JSONResponse({
                "success": True,
                "conversations": formatted_conversations,
                "total": total,
                "page": page,
                "limit": limit,
                "total_pages": total_pages
            })
            
        except Exception as e:
            logger.error(f"獲取對話記錄失敗 {bot_name}: {e}")
            return JSONResponse({
                "success": False,
                "message": f"獲取對話記錄失敗: {str(e)}"
            }, status_code=500)



    def setup_routes(self):
        """設置路由 - 使用統一認證中間件"""
        # 🆕 健康檢查端點
        @self.app.get("/health")
        async def health_check():
            """健康檢查端點"""
            try:
                return JSONResponse({
                    "status": "healthy",
                    "service": "bot_service_manager",
                    "timestamp": datetime.now().isoformat(),
                    "auth_available": AUTH_AVAILABLE,
                    "user_manager_available": USER_MANAGER_AVAILABLE,
                    "vector_system_available": VECTOR_SYSTEM_AVAILABLE,
                    "conversation_logger_available": CONVERSATION_LOGGER_AVAILABLE,
                    "active_bots": len(global_bot_processes),
                    "running_bots": len([p for p in global_bot_processes.values() if p.poll() is None])
                })
            except Exception as e:
                return JSONResponse({
                    "status": "error",
                    "error": str(e)
                }, status_code=500)
        
        @self.app.get("/")
        async def root():
            return RedirectResponse(url="/manager", status_code=302)
        
        
        @self.app.get("/")
        async def root():
            return RedirectResponse(url="/manager", status_code=302)

        @self.app.get("/login", response_class=HTMLResponse)
        async def login_page(request: Request):
            return templates.TemplateResponse("manager_login.html", {"request": request})

        @self.app.get("/manager", response_class=HTMLResponse)
        async def manager_page(request: Request, current_user: User = Depends(AdminAuth)):
            """管理器主頁面 - 需要管理員權限"""
            return templates.TemplateResponse("manager_ui.html", {"request": request})

        # 認證API - 使用統一的認證系統（與 simplified_admin_server.py 完全一致）
        @self.app.post("/api/login")
        async def login(request: Request):
            return await self.handle_login(request)

        @self.app.post("/api/logout")
        async def logout():
            return AuthResponse.create_logout_response()

        # 機器人管理API - 使用統一認證依賴
        @self.app.get("/api/bots")
        async def get_all_bots(current_user: User = Depends(AdminAuth)):
            return await self.handle_get_all_bots()

        @self.app.post("/api/bots/create")
        async def create_bot(
            request: Request, 
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_create_bot(request, current_user)

        @self.app.get("/api/bots/{bot_name}/config")
        async def get_bot_config(
            bot_name: str,
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_get_bot_config(bot_name)

        @self.app.post("/api/bots/{bot_name}/config")
        async def save_bot_config(
            bot_name: str,
            request: Request,
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_save_bot_config(bot_name, request, current_user)

        @self.app.post("/api/bots/{bot_name}/start")
        async def start_bot(
            bot_name: str, 
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_start_bot(bot_name, current_user)

        @self.app.post("/api/bots/{bot_name}/stop")
        async def stop_bot(
            bot_name: str, 
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_stop_bot(bot_name, current_user)

        @self.app.delete("/api/bots/{bot_name}")
        async def delete_bot(
            bot_name: str,
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_delete_bot(bot_name, current_user)

        # 知識庫管理API
        @self.app.get("/api/bots/{bot_name}/knowledge/files")
        async def get_knowledge_files(
            bot_name: str, 
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_get_knowledge_files(bot_name)

        @self.app.post("/api/bots/{bot_name}/knowledge/upload")
        async def upload_knowledge_file(
            bot_name: str, 
            file: UploadFile = File(...),
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_upload_knowledge_file(bot_name, file, current_user)

        @self.app.delete("/api/bots/{bot_name}/knowledge/files/{filename}")
        async def delete_knowledge_file(
            bot_name: str, 
            filename: str,
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_delete_knowledge_file(bot_name, filename, current_user)

        @self.app.get("/api/bots/{bot_name}/knowledge/files/{filename}/details")
        async def get_knowledge_file_details(
            bot_name: str,
            filename: str,
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_get_knowledge_file_details(bot_name, filename)
        
        # 🆕 新增：同步端點 - 修正版
        @self.app.post("/api/bots/{bot_name}/knowledge/sync")
        async def sync_knowledge_base(
            bot_name: str,
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_sync_knowledge_base(bot_name, current_user)
        
        # 在現有知識庫管理API後面添加對話記錄管理API
        @self.app.get("/api/bots/{bot_name}/conversations")
        async def get_conversations(
            bot_name: str,
            page: int = 1,
            limit: int = 20,
            search: str = "",
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_get_conversations(bot_name, page, limit, search)

        @self.app.get("/api/bots/{bot_name}/conversations/chunk/{chunk_index}")
        async def get_chunk_content(
            bot_name: str,
            chunk_index: int,
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_get_chunk_content(bot_name, chunk_index)

        @self.app.delete("/api/bots/{bot_name}/conversations/{conversation_id}")
        async def delete_conversation(
            bot_name: str,
            conversation_id: int,
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_delete_conversation(bot_name, conversation_id, current_user)

        @self.app.delete("/api/bots/{bot_name}/conversations")
        async def delete_conversations_batch(
            bot_name: str,
            request: Request,
            current_user: User = Depends(AdminAuth)
        ):
            return await self.handle_delete_conversations_batch(bot_name, request, current_user)
    

    async def handle_login(self, request: Request):
        """處理登入 - 使用統一的認證和響應格式（與 simplified_admin_server.py 完全一致）"""
        try:
            data = await request.json()
            username = data.get("username", "").strip()
            password = data.get("password", "")

            if not username or not password:
                return JSONResponse({
                    "success": False, 
                    "message": "請填寫用戶名和密碼"
                }, status_code=400)

            # 使用 user_manager 認證
            if USER_MANAGER_AVAILABLE:
                try:
                    from user_manager import user_manager
                    
                    success, token_or_msg, user = user_manager.authenticate(
                        username, password,
                        ip_address=request.client.host if request.client else "unknown"
                    )
                    
                    if success and user.role in ["admin", "super_admin"]:
                        # 創建 JWT Token
                        jwt_token = JWTManager.create_access_token(user)
                        
                        # 使用統一的登入響應格式
                        response = AuthResponse.create_login_response(user, jwt_token)
                        
                        logger.info(f"✅ 用戶 {username} 登入機器人管理器成功")
                        return response
                        
                    elif success:
                        return JSONResponse({
                            "success": False, 
                            "message": "需要管理員權限才能訪問機器人管理界面"
                        }, status_code=403)
                    else:
                        logger.warning(f"機器人管理器認證失敗: {username}")
                        return JSONResponse({
                            "success": False, 
                            "message": token_or_msg
                        }, status_code=401)
                        
                except Exception as e:
                    logger.error(f"用戶管理器認證異常: {e}")
            
            # 🔧 備用認證 - 環境變數 admin 帳戶
            admin_pw = os.getenv("ADMIN_PASSWORD", "ggyyggyyggyy")
            if username == "admin" and password == admin_pw:
                # 創建備用用戶對象
                backup_user = User(
                    id=1,
                    username="admin", 
                    role="admin",
                    email="admin@example.com"
                )
                
                # 創建 JWT Token
                jwt_token = JWTManager.create_access_token(backup_user)
                
                # 使用統一的登入響應格式
                response = AuthResponse.create_login_response(backup_user, jwt_token)
                
                logger.info(f"✅ 機器人管理器備用認證成功: {username}")
                return response
            
            return JSONResponse({
                "success": False, 
                "message": "用戶名或密碼錯誤"
            }, status_code=401)

        except Exception as e:
            logger.error(f"機器人管理器登入處理異常: {e}")
            return JSONResponse({
                "success": False, 
                "message": f"登入系統異常: {str(e)}"
            }, status_code=500)

    async def handle_get_all_bots(self):
        """獲取所有機器人列表 - 包含顯示名稱與公開連結"""
        try:
            bots = []
            for config_file in sorted(BOT_CONFIGS_DIR.glob("*.json")):
                bot_name = config_file.stem
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        config = json.load(f)

                    process = global_bot_processes.get(bot_name)
                    status = "running" if process and process.poll() is None else "stopped"

                    # 顯示名稱（空字串時回退）
                    display_name = config.get("display_name") or bot_name

                    # system_role 預覽（避免 UI 爆版面）
                    system_role_preview = config.get("system_role", "")
                    if len(system_role_preview) > 100:
                        system_role_preview = system_role_preview[:100] + "..."

                    bots.append({
                        "name": bot_name,                              # 技術名稱
                        "display_name": display_name,                  # 顯示名稱
                        "port": config.get("port"),
                        "status": status,
                        "data_dir": str(BOT_DATA_DIR / bot_name),
                        "system_role": system_role_preview,
                        "temperature": config.get("temperature", 0.7),
                        "max_tokens": config.get("max_tokens", 2000),
                        "public_url": f"{BASE_PUBLIC_URL.rstrip('/')}/{bot_name}",
                    })
                except Exception as e:
                    logger.error(f"讀取機器人配置失敗 {bot_name}: {e}")
                    continue

            # 讓 running 的 bot 排前面，接著按名稱排序
            bots.sort(key=lambda b: (b["status"] != "running", b["name"]))

            # 🔧 修復：直接返回數組格式，符合前端期望
            return JSONResponse(bots)

        except Exception as e:
            logger.exception(f"獲取機器人列表失敗: {e}")
            # 🔧 保持原有錯誤處理：返回錯誤響應，讓前端的 catch 塊處理
            return JSONResponse({"success": False, "message": str(e)}, status_code=500)


    async def handle_create_bot(self, request: Request, current_user: User):
        """創建新機器人"""
        try:
            data = await request.json()
            bot_name = data.get("bot_name", "").strip()
            port = data.get("port")
            system_role = data.get("system_role", "")

            if not bot_name or not port:
                return JSONResponse({"success": False, "message": "機器人名稱和端口為必填項"}, status_code=400)

            if not bot_name.replace("_", "").replace("-", "").isalnum():
                return JSONResponse({"success": False, "message": "機器人名稱只能包含字母、數字、下劃線和連字符"}, status_code=400)

            config_path = BOT_CONFIGS_DIR / f"{bot_name}.json"
            if config_path.exists():
                return JSONResponse({"success": False, "message": "此機器人名稱已存在"}, status_code=409)

            for existing_config_file in BOT_CONFIGS_DIR.glob("*.json"):
                try:
                    with open(existing_config_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if existing_data.get("port") == int(port):
                            return JSONResponse({"success": False, "message": f"端口 {port} 已被機器人 {existing_config_file.stem} 使用"}, status_code=409)
                except Exception:
                    continue

            (BOT_DATA_DIR / bot_name).mkdir(exist_ok=True)

            default_config = {
                "bot_name": bot_name,
                "port": int(port),
                "system_role": system_role or f"這是一個名為 {bot_name} 的 AI 助理。我會盡力幫助你解答問題和提供協助。",
                "temperature": data.get("temperature", 0.7),
                "max_tokens": data.get("max_tokens", 2000),
                "dynamic_recommendations_enabled": data.get("dynamic_recommendations_enabled", False),
                "dynamic_recommendations_count": data.get("dynamic_recommendations_count", 0),
                "cite_sources_enabled": data.get("cite_sources_enabled", False),
                "line_config": {
                    "token_env_var": data.get("line_token", ""),
                    "secret_env_var": data.get("line_secret", "")
                },
                "created_by": current_user.username,
                "created_at": datetime.now().isoformat()
            }

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=4)

            logger.info(f"✅ 機器人 {bot_name} 創建成功，創建者: {current_user.username}")
            return JSONResponse({"success": True, "message": f"機器人 {bot_name} 創建成功"})

        except Exception as e:
            logger.error(f"創建機器人失敗: {e}")
            return JSONResponse({"success": False, "message": f"創建機器人異常: {str(e)}"}, status_code=500)

    async def handle_get_bot_config(self, bot_name: str):
        """獲取機器人配置"""
        try:
            config_path = BOT_CONFIGS_DIR / f"{bot_name}.json"
            if not config_path.exists():
                return JSONResponse({"success": False, "message": "機器人配置不存在"}, status_code=404)

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            return JSONResponse({"success": True, "config": config})

        except Exception as e:
            logger.error(f"獲取機器人配置失敗: {e}")
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    async def handle_save_bot_config(self, bot_name: str, request: Request, current_user: User):
        """保存機器人配置 - 添加顯示名稱支持"""
        try:
            config_path = BOT_CONFIGS_DIR / f"{bot_name}.json"
            if not config_path.exists():
                return JSONResponse({"success": False, "message": "機器人配置不存在"}, status_code=404)

            data = await request.json()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 處理顯示名稱
            if "display_name" in data:
                display_name = data["display_name"].strip()
                if len(display_name) > 50:
                    return JSONResponse({
                        "success": False, 
                        "error": "顯示名稱不能超過50個字符"
                    }, status_code=400)
                if not display_name:
                    display_name = config.get("bot_name", bot_name)  # 默認使用技術名稱
                config["display_name"] = display_name

            # 處理其他基本配置欄位
            config.update({
                "system_role": data.get("system_role", config.get("system_role", "")),
                "temperature": data.get("temperature", config.get("temperature", 0.7)),
                "max_tokens": data.get("max_tokens", config.get("max_tokens", 2000)),
                "port": data.get("port", config.get("port")),
                "updated_by": current_user.username,
                "updated_at": datetime.now().isoformat()
            })

            # 處理 checkbox 布林值
            if "dynamic_recommendations_enabled" in data:
                config["dynamic_recommendations_enabled"] = bool(data["dynamic_recommendations_enabled"])
            
            if "dynamic_recommendations_count" in data:
                config["dynamic_recommendations_count"] = int(data.get("dynamic_recommendations_count", 0))
            
            if "cite_sources_enabled" in data:
                config["cite_sources_enabled"] = bool(data["cite_sources_enabled"])

            if "line_config" in data:
                config["line_config"] = data["line_config"]

            # 保存配置文件
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)

            display_name = config.get("display_name", bot_name)
            logger.info(f"✅ 機器人 {bot_name} (顯示名稱: {display_name}) 配置已保存，更新者: {current_user.username}")
            
            return JSONResponse({
                "success": True, 
                "message": f"機器人 {display_name} 的配置已保存成功！"
            })

        except Exception as e:
            logger.error(f"保存機器人配置失敗: {e}", exc_info=True)
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    

    async def handle_delete_conversation(self, bot_name: str, conversation_id: int, current_user: User):
        """刪除單筆對話記錄 - 工廠函數版"""
        try:
            # ✅ 使用連接池+工廠函數獲取記錄器
            conv_logger = self.get_conversation_logger(bot_name)
            
            # 刪除對話記錄
            success = conv_logger.delete_conversation(conversation_id)
            
            if success:
                logger.info(f"✅ 對話記錄 {conversation_id} 已從 {bot_name} 刪除，操作者: {current_user.username}")
                return JSONResponse({
                    "success": True,
                    "message": "對話記錄已刪除"
                })
            else:
                return JSONResponse({
                    "success": False,
                    "message": "刪除失敗，記錄可能不存在"
                }, status_code=404)
            
        except Exception as e:
            logger.error(f"刪除對話記錄失敗 {bot_name}/{conversation_id}: {e}")
            return JSONResponse({
                "success": False,
                "message": f"刪除對話記錄失敗: {str(e)}"
            }, status_code=500)

    async def handle_get_chunk_content(self, bot_name: str, chunk_index: int):
        """
        🔧 修正版：從對話記錄直接獲取chunk內容
        
        優先策略：
        1. 從對話記錄中查找包含該chunk_index的會話
        2. 提取對話記錄中保存的chunk信息（最準確）
        3. 如果找不到，備用向量數據庫查詢
        
        Args:
            bot_name: 機器人名稱
            chunk_index: chunk索引（來自前端點擊）
            
        Returns:
            JSONResponse: 包含chunk詳細信息
        """
        try:
            logger.info(f"🔍 查詢 chunk {chunk_index} for bot {bot_name}")
            
            # 🎯 方法1：從對話記錄中查找chunk內容（推薦方案）
            chunk_from_conversation = await self._get_chunk_from_conversations(bot_name, chunk_index)
            if chunk_from_conversation:
                return chunk_from_conversation
            
            # 🔄 方法2：備用 - 從向量數據庫查詢
            logger.info(f"💡 對話記錄中未找到chunk {chunk_index}，嘗試向量數據庫")
            return await self._get_chunk_from_vector_store(bot_name, chunk_index)
        
        except Exception as e:
            logger.error(f"❌ 獲取chunk內容失敗 {bot_name}/chunk_{chunk_index}: {e}")
            return JSONResponse({
                "success": False, 
                "message": f"獲取chunk內容失敗: {str(e)}"
            }, status_code=500)

    async def _get_chunk_from_conversations(self, bot_name: str, chunk_index: int):
        """從對話記錄中查找chunk內容 - 工廠函數版"""
        try:
            logger.info(f"🔍 查詢 chunk {chunk_index} for bot {bot_name}")
            
            # ✅ 使用連接池+工廠函數獲取記錄器
            conv_logger = self.get_conversation_logger(bot_name)
            
            # 查詢最近的對話記錄
            conversations, _ = conv_logger.get_conversations(
                limit=100,  # 查詢最近100筆對話
                offset=0
            )
            
            # 在每個對話的chunk_references中查找匹配的chunk
            for conv in conversations:
                chunk_refs = conv.get('chunk_references', [])
                if isinstance(chunk_refs, list):
                    for chunk_ref in chunk_refs:
                        if isinstance(chunk_ref, dict) and chunk_ref.get('index') == chunk_index:
                            logger.info(f"✅ 找到 chunk {chunk_index} 在對話 {conv['id']}")
                            
                            full_content = await self._extract_full_content(chunk_ref, bot_name)
                            
                            chunk_detail = {
                                'chunk_id': chunk_ref.get('id', f"chunk_{chunk_index}"),
                                'chunk_index': chunk_index,
                                'content': full_content,
                                'content_preview': chunk_ref.get('content_preview', ''),
                                'similarity': chunk_ref.get('similarity'),
                                'source_file': chunk_ref.get('filename') or chunk_ref.get('source', 'unknown'),
                                'found_in_conversation': conv['id'],
                                'conversation_timestamp': conv['timestamp'],
                                'collection_used': conv['collection_used'],
                                'metadata': {
                                    'chunk_index': chunk_index,
                                    'data_source': 'conversation_record',
                                    'conversation_id': conv['id']
                                }
                            }
                            
                            return JSONResponse({
                                "success": True, 
                                "chunk": chunk_detail,
                                "data_source": "conversation_record",
                                "message": f"從對話記錄 {conv['id']} 中獲取"
                            })
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 從對話記錄查詢chunk失敗: {e}")
            return None

    async def _extract_full_content(self, chunk_ref: dict, bot_name: str) -> str:
        """
        🔧 提取chunk的完整內容
        
        優先順序：
        1. chunk_ref中的完整內容
        2. 通過相似性搜索從向量庫獲取
        3. 返回預覽內容（最後手段）
        """
        try:
            # 1. 檢查是否已有完整內容
            full_content = chunk_ref.get('content') or chunk_ref.get('full_content')
            if full_content and len(full_content) > 200:  # 假設完整內容應該比較長
                logger.debug("✅ 使用chunk_ref中的完整內容")
                return full_content
            
            # 2. 嘗試從向量數據庫通過相似性搜索獲取
            preview = chunk_ref.get('content_preview', '')
            if preview and len(preview) > 20:  # 確保有足夠的預覽內容進行搜索
                try:
                    if VECTOR_SYSTEM_AVAILABLE and vector_system:
                        collection_name = f"collection_{bot_name}"
                        vectorstore = vector_system.get_or_create_vectorstore(collection_name)
                        
                        # 使用預覽內容的前50個字符進行相似性搜索
                        search_query = preview.replace("...", "").strip()[:50]
                        search_results = vectorstore.similarity_search(
                            search_query,
                            k=3  # 獲取前3個最相似的結果
                        )
                        
                        # 找到最匹配的文檔
                        for doc in search_results:
                            # 檢查預覽內容是否在文檔中
                            doc_start = doc.page_content[:300]  # 檢查文檔開頭300字符
                            if search_query in doc_start:
                                logger.debug("💡 通過相似性搜索找到完整內容")
                                return doc.page_content
                                
                except Exception as e:
                    logger.warning(f"⚠️ 相似性搜索失敗: {e}")
            
            # 3. 最後手段：返回預覽內容
            logger.debug("📝 使用預覽內容作為fallback")
            return preview or f"[Chunk 內容暫時無法完整獲取]"
            
        except Exception as e:
            logger.warning(f"⚠️ 提取完整內容時出錯: {e}")
            return chunk_ref.get('content_preview', f"[Chunk 內容獲取失敗: {str(e)}]")

    async def _get_chunk_from_vector_store(self, bot_name: str, chunk_index: int):
        """
        🔄 備用方法：從向量數據庫獲取chunk內容
        
        當對話記錄中找不到chunk時使用此方法
        """
        try:
            collection_name = f"collection_{bot_name}"
            
            # API模式處理
            if self.use_vector_api:
                logger.info(f"🌐 使用向量API模式查詢 chunk {chunk_index}")
                return JSONResponse({
                    "success": False,
                    "message": "API模式下的chunk詳情功能正在開發中",
                    "collection_name": collection_name,
                    "chunk_index": chunk_index,
                    "data_source": "vector_api_fallback"
                }, status_code=200)
            
            # 直連模式處理
            if not VECTOR_SYSTEM_AVAILABLE or not vector_system:
                return JSONResponse({
                    "success": False, 
                    "message": "向量系統不可用，且對話記錄中未找到該chunk"
                }, status_code=503)

            logger.info(f"🔗 使用向量數據庫直連模式查詢 chunk {chunk_index}")
            vectorstore = vector_system.get_or_create_vectorstore(collection_name)
            all_docs = vectorstore.get()

            if not all_docs or not all_docs.get('documents'):
                return JSONResponse({
                    "success": False, 
                    "message": f"機器人 {bot_name} 的向量數據庫中沒有文檔"
                }, status_code=404)
                
            # 檢查索引範圍
            total_docs = len(all_docs['documents'])
            if chunk_index >= total_docs:
                return JSONResponse({
                    "success": False, 
                    "message": f"Chunk 索引 {chunk_index} 超出範圍 (最大: {total_docs-1})"
                }, status_code=404)

            # 獲取文檔和元數據
            document = all_docs['documents'][chunk_index]
            metadata = all_docs['metadatas'][chunk_index] if all_docs.get('metadatas') else {}
            
            chunk_detail = {
                'chunk_id': metadata.get('chunk_id', f"chunk_{chunk_index}"),
                'chunk_index': chunk_index,
                'content': document,
                'source_file': metadata.get('filename') or metadata.get('original_filename') or metadata.get('source', 'unknown'),
                'metadata': metadata,
                'data_source': 'vector_store_fallback'
            }
            
            logger.info(f"⚠️ 使用備用方法從向量數據庫獲取 chunk {chunk_index}")
            return JSONResponse({
                "success": True, 
                "chunk": chunk_detail,
                "data_source": "vector_store_fallback",
                "message": "從向量數據庫獲取（備用方法）"
            })
            
        except Exception as e:
            logger.error(f"❌ 向量數據庫備用查詢失敗: {e}")
            return JSONResponse({
                "success": False, 
                "message": f"所有獲取方法均失敗: {str(e)}"
            }, status_code=500)


    async def handle_sync_knowledge_base(self):
        try:
            if self.use_vector_api:
                url = f"{self.vector_api_url}/sync"
                timeout = httpx.Timeout(60.0, connect=5.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url)
                if resp.status_code == 200:
                    payload = resp.json()
                    return JSONResponse({
                        "success": payload.get("success", True),
                        "message": payload.get("message", "同步完成"),
                        "changes": payload.get("changes")
                    })
                else:
                    return JSONResponse({"success": False, "message": f"API错误: {resp.text}"}, status_code=resp.status_code)
            else:
                if not VECTOR_SYSTEM_AVAILABLE or not vector_system:
                    return JSONResponse({"success": False, "message": "向量系统不可用"}, status_code=503)
                if hasattr(vector_system, "sync_collections"):
                    changes = vector_system.sync_collections()
                    return JSONResponse({"success": True, "message": f"同步完成，处理了 {changes} 个变更", "changes": changes})
                return JSONResponse({"success": False, "message": "当前版本不支持同步功能"}, status_code=501)
        except Exception as e:
            logger.error(f"同步知识库失败: {e}")
            return JSONResponse({"success": False, "message": str(e)}, status_code=500)


    async def handle_delete_conversations_batch(self, bot_name: str, request: Request, current_user: User):
        """批量刪除對話記錄 - 工廠函數版"""
        try:
            data = await request.json()
            conversation_ids = data.get("conversation_ids", [])
            
            if not conversation_ids:
                return JSONResponse({
                    "success": False,
                    "message": "請選擇要刪除的對話記錄"
                }, status_code=400)

            # ✅ 使用連接池+工廠函數獲取記錄器
            conv_logger = self.get_conversation_logger(bot_name)
            
            # 批量刪除對話記錄
            deleted_count = conv_logger.delete_conversations_batch(conversation_ids)
            
            if deleted_count > 0:
                logger.info(f"✅ 批量刪除 {deleted_count} 筆對話記錄從 {bot_name}，操作者: {current_user.username}")
                return JSONResponse({
                    "success": True,
                    "message": f"成功刪除 {deleted_count} 筆對話記錄"
                })
            else:
                return JSONResponse({
                    "success": False,
                    "message": "沒有記錄被刪除，可能記錄不存在"
                }, status_code=404)
            
        except Exception as e:
            logger.error(f"批量刪除對話記錄失敗 {bot_name}: {e}")
            return JSONResponse({
                "success": False,
                "message": f"批量刪除對話記錄失敗: {str(e)}"
            }, status_code=500)


    async def handle_start_bot(self, bot_name: str, current_user: User):
        """啟動機器人（寫檔避免 PIPE、啟動後註冊到 Gateway）"""
        try:
            if bot_name in global_bot_processes and global_bot_processes[bot_name].poll() is None:
                return JSONResponse({"success": False, "message": "機器人已在運行中"}, status_code=409)

            config_path = BOT_CONFIGS_DIR / f"{bot_name}.json"
            if not config_path.exists():
                return JSONResponse({"success": False, "message": "機器人配置不存在"}, status_code=404)

            # 讀取設定檔取得 port
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                port = int(cfg.get("port")) if cfg.get("port") else None
                if not port:
                    return JSONResponse({"success": False, "message": "機器人設定缺少 port"}, status_code=400)
            except Exception as ce:
                return JSONResponse({"success": False, "message": f"讀取機器人設定失敗: {ce}"}, status_code=400)

            if not BOT_INSTANCE_SCRIPT.exists():
                return JSONResponse({"success": False, "message": "機器人實例腳本不存在"}, status_code=500)

            command = [
                sys.executable,
                str(BOT_INSTANCE_SCRIPT),
                "--bot-name", bot_name
            ]

            # 將 stdout/stderr 寫入檔案，避免 PIPE 未讀造成子程序阻塞
            LOGS_DIR.mkdir(exist_ok=True)
            log_path = LOGS_DIR / f"{bot_name}.log"
            log_file = open(log_path, "a", encoding="utf-8", buffering=1)  # 行緩衝

            # 明確傳遞環境變數給子程序（避免在服務形態下讀不到 .env）
            child_env = os.environ.copy()
            child_env["USE_VECTOR_API"] = "true" if USE_VECTOR_API else "false"
            child_env["VECTOR_API_URL"] = VECTOR_API_URL

            try:
                process = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=log_file,
                    cwd=ROOT_DIR,
                    env=child_env,
                    bufsize=1
                )
            except Exception as pe:
                try:
                    log_file.flush(); log_file.close()
                except Exception:
                    pass
                raise pe

            global_bot_processes[bot_name] = process
            global_bot_log_files[bot_name] = log_file

            # 非阻塞：通知 Gateway 註冊（失敗不擋流程）
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(3.0, connect=1.0)) as client:
                    await client.post(
                        f"{GATEWAY_URL}/_gateway/register",
                        headers={"X-Admin-Token": GATEWAY_ADMIN_TOKEN} if GATEWAY_ADMIN_TOKEN else {},
                        json={"bot": bot_name, "port": port}
                    )
            except Exception as ge:
                logger.warning(f"[manager] Gateway 註冊失敗（忽略不擋流程）：{ge}")

            logger.info(f"✅ 機器人 {bot_name} 已啟動，操作者: {current_user.username}")
            return JSONResponse({"success": True, "message": f"機器人 {bot_name} 已啟動"})

        except Exception as e:
            logger.error(f"啟動機器人失敗: {e}")
            return JSONResponse({"success": False, "message": f"啟動機器人異常: {str(e)}"}, status_code=500)

    async def handle_stop_bot(self, bot_name: str, current_user: User):
        try:
            process = global_bot_processes.get(bot_name)
            if not process or process.poll() is not None:
                return JSONResponse({"success": False, "message": "機器人未在運行中或已停止"}, status_code=404)

            try:
                # 先請求優雅停止
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            finally:
                # ✅ 關閉 log 檔案並註銷 Gateway 路由
                lf = global_bot_log_files.pop(bot_name, None)
                if lf:
                    try:
                        lf.flush()
                    finally:
                        try:
                            lf.close()
                        except Exception:
                            pass

                try:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=1.0)) as client:
                        await client.post(
                            f"{GATEWAY_URL}/_gateway/unregister",
                            headers={"X-Admin-Token": GATEWAY_ADMIN_TOKEN} if GATEWAY_ADMIN_TOKEN else {},
                            json={"bot": bot_name}
                        )
                except Exception as e:
                    logger.warning(f"Gateway 取消註冊失敗（忽略）：{e}")

            # 再移除子行程紀錄（一定要在 unregister 之後）
            if bot_name in global_bot_processes:
                del global_bot_processes[bot_name]

            logger.info(f"✅ 機器人 {bot_name} 已停止，操作者: {current_user.username}")
            return JSONResponse({"success": True, "message": f"機器人 {bot_name} 已停止"})

        except Exception as e:
            logger.error(f"停止機器人失敗: {e}")
            return JSONResponse({"success": False, "message": f"停止機器人異常: {str(e)}"}, status_code=500)


    async def handle_delete_bot(self, bot_name: str, current_user: User):
        """刪除機器人"""
        try:
            # 先停止機器人（如果正在運行）
            if bot_name in global_bot_processes:
                await self.handle_stop_bot(bot_name, current_user)

            config_path = BOT_CONFIGS_DIR / f"{bot_name}.json"
            if config_path.exists():
                config_path.unlink()

            # 刪除數據目錄
            bot_data_dir = BOT_DATA_DIR / bot_name
            if bot_data_dir.exists():
                import shutil
                shutil.rmtree(bot_data_dir)

            logger.info(f"✅ 機器人 {bot_name} 已刪除，操作者: {current_user.username}")

            return JSONResponse({
                "success": True,
                "message": f"機器人 {bot_name} 已刪除"
            })

        except Exception as e:
            logger.error(f"刪除機器人失敗: {e}")
            return JSONResponse({
                "success": False,
                "message": f"刪除機器人異常: {str(e)}"
            }, status_code=500)

    # --- Knowledge Base Management ---

    _records_lock = threading.Lock()

    def _get_records_path(self) -> Path:
        """返回記錄檔的路徑"""
        return ROOT_DIR / "chroma_langchain_db" / "file_records.json"

    def _read_records(self) -> Dict:
        """安全地讀取記錄檔 - 增加調試信息"""
        records_path = self._get_records_path()
        logger.info(f"📖 讀取記錄檔案: {records_path}")
        
        if not records_path.exists():
            logger.info("📝 記錄檔案不存在，返回空字典")
            return {}
        
        with self._records_lock:
            try:
                with open(records_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        logger.info("📝 記錄檔案為空")
                        return {}
                    
                    data = json.loads(content)
                    logger.info(f"✅ 成功讀取記錄: {len(data)} 個集合")
                    return data
                    
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON解析失敗: {e}")
                return {}
            except Exception as e:
                logger.error(f"❌ 讀取記錄檔案失敗: {e}")
                return {}

    def _write_records(self, data: Dict):
        """安全地寫入記錄檔 - 增加調試信息"""
        records_path = self._get_records_path()
        logger.info(f"💾 寫入記錄檔案: {records_path}")
        
        # 確保目錄存在
        records_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._records_lock:
            try:
                with open(records_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ 成功寫入記錄: {len(data)} 個集合")
            except Exception as e:
                logger.error(f"❌ 寫入記錄檔案失敗: {e}", exc_info=True)


    def _update_file_record_with_upload_info(self, collection_name: str, filename: str, 
                                       uploaded_by: str, file_path: str = None):
        """🔧 修正：更新向量系統中的文件記錄"""
        try:
            logger.info(f"🔧 更新向量系統文件記錄: {filename}")
            
            # 直接操作向量系統的文件記錄
            if not hasattr(vector_system, 'file_records'):
                logger.warning("向量系統沒有 file_records 屬性")
                return False

            if collection_name not in vector_system.file_records:
                vector_system.file_records[collection_name] = {}
            
            # 🆕 修正：嘗試多種可能的鍵值格式
            possible_keys = [
                file_path,  # 完整路徑
                filename,   # 檔案名
            ]
            
            # 如果有檔案路徑，也嘗試相對路徑
            if file_path:
                try:
                    rel_path = Path(file_path).relative_to(Path.cwd())
                    possible_keys.append(str(rel_path))
                except ValueError:
                    pass  # 無法計算相對路徑

            record_updated = False
            for key in possible_keys:
                if key and key in vector_system.file_records[collection_name]:
                    file_info = vector_system.file_records[collection_name][key]
                    
                    # 🆕 修復：將管理字段作為屬性添加到 FileInfo 對象
                    if hasattr(file_info, '__dict__'):
                        file_info.uploaded_by = uploaded_by
                        file_info.uploaded_at = datetime.now().isoformat()
                        file_info.file_source = "upload"  # 🆕 標記來源
                        record_updated = True
                        logger.info(f"✅ 使用鍵值 {key} 更新記錄成功")
                        break
            
            if record_updated:
                # 保存更新後的記錄
                try:
                    vector_system._save_file_records()
                    logger.info(f"✅ 向量系統記錄更新並保存成功: {filename}")
                    return True
                except Exception as save_error:
                    logger.error(f"保存記錄失敗: {save_error}")
                    return False
            else:
                logger.warning(f"⚠️ 無法找到檔案記錄進行更新: {filename}")
                # 列出現有的鍵值用於調試
                if collection_name in vector_system.file_records:
                    existing_keys = list(vector_system.file_records[collection_name].keys())
                    logger.warning(f"現有記錄鍵值: {existing_keys[:5]}...")  # 只顯示前5個
                return False
            
        except Exception as e:
            logger.error(f"更新向量系統記錄失敗: {e}", exc_info=True)
            return False
    
    async def _upload_file_with_correct_path(self, bot_name: str, file: UploadFile, 
                                       current_user: User, file_content: bytes, 
                                       collection_name: str, safe_filename: str):
        """🔧 修正版備用上傳方式 - 保存到正確位置"""
        try:
            # 🆕 修正：保存到正確的 data/ 目錄結構
            target_dir = ROOT_DIR / "data" / bot_name  # 使用正確的 data/ 路徑
            
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return JSONResponse({
                    "success": False, 
                    "message": f"無法創建目標目錄: {str(e)}"
                }, status_code=500)
            
            file_path = target_dir / safe_filename
            
            # 🔧 處理檔案衝突
            if file_path.exists():
                logger.info(f"⚠️ 檔案 {safe_filename} 已存在，將會覆蓋")

            # 🆕 保存檔案到正確位置
            try:
                with open(file_path, "wb") as f:
                    f.write(file_content)
                
                # 驗證檔案寫入
                if not file_path.exists() or file_path.stat().st_size != len(file_content):
                    raise IOError("檔案寫入驗證失敗")
                    
                logger.info(f"💾 檔案已保存到: {file_path}")
            except Exception as e:
                # 清理不完整的檔案
                try:
                    if file_path.exists():
                        file_path.unlink()
                except:
                    pass
                return JSONResponse({
                    "success": False, 
                    "message": f"檔案保存失敗: {str(e)}"
                }, status_code=500)

            # 🆕 使用向量系統載入文檔
            try:
                chunks = vector_system.load_document(file_path)
                if not chunks:
                    file_path.unlink()  # 清理失敗的文件
                    return JSONResponse({
                        "success": False, 
                        "message": "文件內容為空或無法處理。"
                    }, status_code=400)
            except Exception as e:
                try:
                    file_path.unlink()  # 清理失敗的文件
                except:
                    pass
                return JSONResponse({
                    "success": False, 
                    "message": f"文檔處理失敗: {str(e)}"
                }, status_code=500)

            # 🆕 更新元數據
            current_timestamp = time.time()
            for doc in chunks:
                doc.metadata.update({
                    'collection': collection_name,
                    'original_filename': safe_filename,
                    'upload_timestamp': current_timestamp,
                    'file_source': 'upload',
                    'source': str(file_path),  # 使用正確路徑
                    'saved_to_data_dir': True
                })

            # 🆕 添加到向量庫
            try:
                vectorstore = vector_system.get_or_create_vectorstore(collection_name)
                
                # 🔧 刪除現有同名文件的向量
                try:
                    delete_conditions = [
                        {"source": str(file_path)},
                        {"original_filename": safe_filename},
                        {"filename": safe_filename}
                    ]
                    
                    for condition in delete_conditions:
                        try:
                            vectorstore.delete(where=condition)
                        except Exception as e:
                            logger.warning(f"刪除現有向量時出現警告 {condition}: {e}")
                except Exception as e:
                    logger.warning(f"批量刪除現有向量時出現警告: {e}")
                
                # 🔧 過濾複雜元數據（如果需要）
                try:
                    from langchain_community.vectorstores.utils import filter_complex_metadata
                    filtered_chunks = filter_complex_metadata(chunks)
                except ImportError:
                    # 如果沒有這個函數，使用我們的元數據處理
                    filtered_chunks = []
                    for chunk in chunks:
                        safe_metadata = vector_system._ensure_simple_metadata(chunk.metadata)
                        from langchain_core.documents import Document
                        safe_chunk = Document(page_content=chunk.page_content, metadata=safe_metadata)
                        filtered_chunks.append(safe_chunk)
                
                vectorstore.add_documents(filtered_chunks)
                
            except Exception as e:
                # 向量化失敗，清理檔案
                try:
                    file_path.unlink()
                except:
                    pass
                return JSONResponse({
                    "success": False, 
                    "message": f"向量化處理失敗: {str(e)}"
                }, status_code=500)
            
            logger.info(f"✅ 知識庫文件 {safe_filename} 已上傳到正確位置: {file_path}")
            logger.info(f"   👤 操作者: {current_user.username}")
            logger.info(f"   📄 分塊數: {len(chunks)}")
            
            return JSONResponse({
                "success": True, 
                "message": f"文件 {safe_filename} 已上傳並保存到 data/{bot_name}/{safe_filename}，索引了 {len(chunks)} 個區塊。",
                "saved_path": str(file_path),
                "total_chunks": len(chunks),
                "file_source": "upload",
                "filename": safe_filename
            })

        except Exception as e:
            logger.error(f"修正版上傳失敗: {e}")
            # 清理失敗時的文件
            if 'file_path' in locals() and file_path.exists():
                try:
                    file_path.unlink()
                except:
                    pass
            return JSONResponse({
                "success": False, 
                "message": f"上傳失敗: {str(e)}"
            }, status_code=500)


    # 🆕 另一個新增方法也放在這裡
    def migrate_existing_records(self):
        """🔧 遷移現有記錄到新格式"""
        try:
            records_path = self._get_records_path()
            if not records_path.exists():
                logger.info("📝 沒有找到需要遷移的記錄文件")
                return
                
            all_records = self._read_records()
            
            migrated = False
            for collection_name, files in all_records.items():
                if isinstance(files, dict):
                    for filename, file_info in files.items():
                        # 檢查是否需要遷移
                        if isinstance(file_info, dict) and 'uploaded_by' not in file_info:
                            # 添加缺失的字段
                            mtime = file_info.get('mtime', time.time())
                            file_info['uploaded_by'] = '系統遷移'
                            file_info['uploaded_at'] = datetime.fromtimestamp(mtime).isoformat()
                            migrated = True
            
            if migrated:
                # 備份原文件
                backup_path = records_path.with_suffix('.json.backup')
                if records_path.exists():
                    import shutil
                    shutil.copy2(records_path, backup_path)
                
                # 保存遷移後的記錄
                self._write_records(all_records)
                logger.info(f"✅ 記錄遷移完成，原文件備份至: {backup_path}")
            else:
                logger.info("📝 記錄已是最新格式，無需遷移")
                
        except Exception as e:
            logger.error(f"記錄遷移失敗: {e}")

    async def handle_get_knowledge_files(self, bot_name: str):
        """🔧 修正：在管理器層級清理來自API的資料"""
        try:
            collection_name = f"collection_{bot_name}"

            # === API 模式 ===
            if self.use_vector_api:
                url = f"{self.vector_api_url}/collections/{collection_name}/files"
                timeout = httpx.Timeout(30.0, connect=5.0)
                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        resp = await client.get(url)
                except httpx.RequestError as re:
                    return JSONResponse({"success": False, "message": f"向量API請求失敗：{re}"}, status_code=503)

                if resp.status_code == 200:
                    payload = resp.json()
                    # --- 在這裡加入清理邏輯 ---
                    if payload and isinstance(payload.get('documents'), list):
                        import os
                        cleaned_docs = []
                        for doc in payload.get('documents', []):
                            if isinstance(doc, dict) and 'filename' in doc and isinstance(doc['filename'], str):
                                new_doc = doc.copy()
                                new_doc['filename'] = os.path.basename(doc['filename'])
                                cleaned_docs.append(new_doc)
                            else:
                                cleaned_docs.append(doc)
                        payload['documents'] = cleaned_docs
                        payload['total'] = len(cleaned_docs)
                    # --- 清理邏輯結束 ---
                    return JSONResponse(payload)

                if resp.status_code == 503:
                    return JSONResponse({"success": False, "message": "向量API服务不可用"}, status_code=503)

                return JSONResponse(
                    {"success": False, "message": f"API错误: {resp.text}"},
                    status_code=resp.status_code
                )

            # === 直連模式 (這部分邏輯不變) ===
            if not VECTOR_SYSTEM_AVAILABLE or not vector_system:
                bot_data_dir = BOT_DATA_DIR / bot_name
                docs = []
                if bot_data_dir.exists():
                    files = [f.name for f in bot_data_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
                    docs = [{"filename": name} for name in sorted(files)]
                return JSONResponse({"success": True, "documents": docs, "total": len(docs)})

            try:
                if hasattr(vector_system, 'get_collection_documents'):
                    docs_result = vector_system.get_collection_documents(collection_name, page=1, limit=1000)
                    if docs_result.get('success', False):
                        documents = docs_result.get('documents', [])
                        return JSONResponse({"success": True, "documents": documents, "total": len(documents)})
            except Exception as ve:
                logger.warning(f"get_collection_documents 失敗，改用向量庫直查: {ve}")

            try:
                vectorstore = vector_system.get_or_create_vectorstore(collection_name)
                all_docs = vectorstore.get()
                filenames = set()
                if all_docs and all_docs.get('metadatas'):
                    for md in all_docs['metadatas']:
                        fn = md.get('filename') or md.get('original_filename') or md.get('source', 'unknown')
                        if fn and fn != 'unknown':
                            filenames.add(os.path.basename(str(fn)))
                documents = sorted([{"filename": fn} for fn in filenames], key=lambda x: x['filename'])
                return JSONResponse({"success": True, "documents": documents, "total": len(documents)})
            except Exception as e:
                logger.warning(f"向量庫直查失敗，回退到檔案系統: {e}")
                bot_data_dir = BOT_DATA_DIR / bot_name
                docs = []
                if bot_data_dir.exists():
                    files = [f.name for f in bot_data_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
                    docs = [{"filename": name} for name in sorted(files)]
                return JSONResponse({"success": True, "documents": docs, "total": len(docs)})

        except Exception as e:
            logger.error(f"獲取知識庫文件列表失敗 for {bot_name}: {e}")
            return JSONResponse({"success": False, "message": f"獲取文件列表時發生錯誤: {e}"}, status_code=500)

    async def handle_get_knowledge_file_details(self, bot_name: str, filename: str):
        """🔧 修正：改進文件詳情查詢，增加調試信息"""
        try:
            collection_name = f"collection_{bot_name}"
            chunk_count = 0
            
            logger.info(f"🔍 查詢文件詳情: {filename}, 集合: {collection_name}")

            # 1. 獲取 chunk 數量
            if VECTOR_SYSTEM_AVAILABLE:
                try:
                    vectorstore = vector_system.get_or_create_vectorstore(collection_name)
                    
                    # 🔧 修正：使用正確的查詢條件並增加調試
                    query_conditions = [
                        {"filename": filename},
                        {"original_filename": filename},
                        {"source": filename}
                    ]
                    
                    for condition in query_conditions:
                        try:
                            results = vectorstore.get(where=condition)
                            if results and results.get("ids"):
                                chunk_count = len(results["ids"])
                                logger.info(f"✅ 找到chunks: {chunk_count}, 條件: {condition}")
                                break
                            else:
                                logger.info(f"🔍 條件無結果: {condition}")
                        except Exception as e:
                            logger.warning(f"查詢條件失敗 {condition}: {e}")
                            
                    logger.info(f"📊 最終chunk數量: {chunk_count}")
                            
                except Exception as e:
                    logger.warning(f"從向量庫獲取chunk數量失敗: {e}")

            # 2. 獲取文件記錄信息
            record_info = {}
            try:
                all_records = self._read_records()
                logger.info(f"📖 讀取記錄: 集合數量 {len(all_records)}")
                
                if collection_name in all_records:
                    bot_records = all_records[collection_name]
                    logger.info(f"📁 集合 {collection_name} 中有 {len(bot_records)} 個文件")
                    
                    if filename in bot_records:
                        record = bot_records[filename]
                        logger.info(f"📝 找到文件記錄，類型: {type(record)}")
                        
                        if isinstance(record, dict):
                            logger.info(f"📄 記錄內容: {list(record.keys())}")
                            record_info = {
                                "uploaded_by": record.get("uploaded_by", "未知"),
                                "uploaded_at": record.get("uploaded_at", "未知")
                            }
                            
                            # 檢查時間格式
                            if record_info["uploaded_at"] != "未知":
                                try:
                                    # 驗證ISO格式
                                    datetime.fromisoformat(record_info["uploaded_at"].replace('Z', '+00:00'))
                                    logger.info(f"✅ 時間格式正確: {record_info['uploaded_at']}")
                                except:
                                    logger.warning(f"⚠️ 時間格式異常: {record_info['uploaded_at']}")
                                    # 嘗試從 mtime 轉換
                                    if "mtime" in record:
                                        try:
                                            record_info["uploaded_at"] = datetime.fromtimestamp(
                                                record["mtime"]
                                            ).isoformat()
                                        except:
                                            record_info["uploaded_at"] = "未知"
                        
                        elif hasattr(record, 'mtime'):
                            # FileInfo 對象格式
                            logger.info("🔄 FileInfo對象格式")
                            try:
                                upload_time = datetime.fromtimestamp(record.mtime).isoformat()
                            except (ValueError, OSError):
                                upload_time = "未知"
                            
                            record_info = {
                                "uploaded_by": "系統遷移",
                                "uploaded_at": upload_time
                            }
                    else:
                        logger.warning(f"⚠️ 集合中未找到文件: {filename}")
                        # 列出集合中的所有文件用於調試
                        logger.info(f"📋 集合中的文件: {list(bot_records.keys())}")
                else:
                    logger.warning(f"⚠️ 未找到集合: {collection_name}")
                    logger.info(f"📋 現有集合: {list(all_records.keys())}")
                    
            except Exception as e:
                logger.error(f"讀取文件記錄失敗: {e}", exc_info=True)

            # 3. 如果沒有找到記錄，提供默認值
            if not record_info:
                logger.info("🔧 使用默認記錄信息")
                record_info = {
                    "uploaded_by": "未知",
                    "uploaded_at": "未知"
                }

            logger.info(f"📤 返回結果 - chunks: {chunk_count}, 上傳者: {record_info.get('uploaded_by')}, 時間: {record_info.get('uploaded_at')}")

            return JSONResponse({
                "success": True,
                "chunk_count": chunk_count,
                "uploaded_by": record_info.get("uploaded_by", "未知"),
                "uploaded_at": record_info.get("uploaded_at", "未知")
            })

        except Exception as e:
            logger.error(f"獲取文件詳細信息失敗: {e}", exc_info=True)
            return JSONResponse({
                "success": False, 
                "message": str(e)
            }, status_code=500)


    async def handle_upload_knowledge_file(self, bot_name: str, file: UploadFile = File(...),current_user: User = None):
        """處理知識庫文件上傳 - 修復版本"""
        try:
            # 🆕 第一步：定義 collection_name
            collection_name = f"collection_{bot_name}"
            
            # 如果使用向量 API
            if self.use_vector_api:
                url = f"{self.vector_api_url}/upload"
                timeout = httpx.Timeout(120.0, connect=5.0)
                file_bytes = await file.read()
                files = {"file": (file.filename, file_bytes, file.content_type or "application/octet-stream")}
                data = {"collection_name": collection_name}  # ✅ 現在已定義
                
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, data=data, files=files)
                    
                if resp.status_code == 200:
                    payload = resp.json()
                    return JSONResponse({
                        "success": payload.get("success", True),
                        "message": payload.get("message", f"已上傳 {file.filename}"),
                        "filename": payload.get("filename", file.filename),
                        "total_chunks": payload.get("total_chunks")
                    })
                else:
                    return JSONResponse({
                        "success": False, 
                        "message": f"API錯誤: {resp.text}"
                    }, status_code=resp.status_code)
            
            else:
                # 直連模式處理
                if not VECTOR_SYSTEM_AVAILABLE or not vector_system:
                    return JSONResponse({
                        "success": False, 
                        "message": "向量系統不可用"
                    }, status_code=503)
                    
                file_bytes = await file.read()
                
                if hasattr(vector_system, "upload_single_file"):
                    result = vector_system.upload_single_file(
                        file_bytes, file.filename, collection_name
                    )
                    return JSONResponse({
                        "success": result.get("success", True),
                        "message": result.get("message", f"已上傳 {file.filename}"),
                        "filename": result.get("filename", file.filename),
                        "total_chunks": result.get("total_chunks")
                    })
                else:
                    return JSONResponse({
                        "success": False, 
                        "message": "當前版本不支持直連上傳"
                    }, status_code=501)
                    
        except Exception as e:
            logger.error(f"上傳知識庫文件失敗: {e}")
            return JSONResponse({
                "success": False, 
                "message": str(e)
            }, status_code=500)
        
    async def _upload_file_traditional_way(self, bot_name: str, file: UploadFile, current_user: User, file_content: bytes, collection_name: str):
        """🔧 備用上傳方式"""
        try:
            # 1. 保存文件到本地
            bot_data_dir = BOT_DATA_DIR / bot_name
            bot_data_dir.mkdir(exist_ok=True)
            
            file_path = bot_data_dir / file.filename
            if file_path.exists():
                return JSONResponse({"success": False, "message": f"文件 '{file.filename}' 已存在。"}, status_code=409)

            with open(file_path, "wb") as f:
                f.write(file_content)

            # 2. 載入文檔並分割
            chunks = vector_system.load_document(file_path)
            if not chunks:
                file_path.unlink()  # 清理失敗的文件
                return JSONResponse({"success": False, "message": "文件內容為空或無法處理。"}, status_code=400)

            # 3. 添加到向量庫
            vectorstore = vector_system.get_or_create_vectorstore(collection_name)
            
            # 🔧 過濾複雜元數據（如果需要）
            try:
                from langchain_community.vectorstores.utils import filter_complex_metadata
                filtered_chunks = filter_complex_metadata(chunks)
            except ImportError:
                # 如果沒有這個函數，就直接使用原始chunks
                filtered_chunks = chunks
                
            vectorstore.add_documents(filtered_chunks)
            
            logger.info(f"✅ 知識庫文件 {file.filename} 已上傳到 {bot_name} (傳統方式)，操作者: {current_user.username}")
            return JSONResponse({
                "success": True, 
                "message": f"文件 {file.filename} 已上傳並索引了 {len(chunks)} 個區塊。"
            })

        except Exception as e:
            logger.error(f"傳統上傳方式失敗: {e}")
            # 清理失敗時的文件
            if 'file_path' in locals() and file_path.exists():
                file_path.unlink()
            return JSONResponse({
                "success": False, 
                "message": f"上傳失敗: {str(e)}"
            }, status_code=500)

    async def handle_delete_knowledge_file(self, bot_name: str, filename: str):
        try:
            collection_name = f"collection_{bot_name}"
            if self.use_vector_api:
                url = f"{self.vector_api_url}/collections/{collection_name}/files/{filename}"
                timeout = httpx.Timeout(30.0, connect=5.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.delete(url)
                if resp.status_code == 200:
                    payload = resp.json()
                    return JSONResponse({
                        "success": payload.get("success", True),
                        "message": payload.get("message", f"已删除 {filename}")
                    })
                else:
                    return JSONResponse({"success": False, "message": f"API错误: {resp.text}"}, status_code=resp.status_code)
            else:
                # 🔽 直連模式：保留你原本的實作（多條件 where 刪除）
                if not VECTOR_SYSTEM_AVAILABLE or not vector_system:
                    return JSONResponse({"success": False, "message": "向量系统不可用"}, status_code=503)
                vectorstore = vector_system.get_or_create_vectorstore(collection_name)
                deleted = 0
                for cond in [{"filename": filename}, {"original_filename": filename}, {"source": filename}]:
                    try:
                        vectorstore.delete(where=cond)
                        deleted += 1
                    except Exception:
                        pass
                if deleted:
                    return JSONResponse({"success": True, "message": f"文件 {filename} 删除成功"})
                return JSONResponse({"success": False, "message": f"未找到文件 {filename}"}, status_code=404)
        except Exception as e:
            logger.error(f"删除知识库文件失败: {e}")
            return JSONResponse({"success": False, "message": str(e)}, status_code=500)
# 需要導入 datetime
from datetime import datetime



def main():
    """主函數 - 添加清理邏輯"""
    server = BotServiceManager()
    app = server.app
    port = int(os.getenv("MANAGER_PORT", 9001))
    
    print("=" * 60)
    print("🤖 聊天機器人服務管理器 - 統一認證版")
    print("=" * 60)
    print(f"🌐 服務端口: {port}")
    print(f"🔗 登入頁面: http://localhost:{port}/login")
    print(f"🖥️ 管理界面: http://localhost:{port}/manager")
    print(f"📁 配置目錄: {BOT_CONFIGS_DIR}")
    print(f"📁 數據目錄: {BOT_DATA_DIR}")
    print()
    print("🎯 系統狀態:")
    print(f"  用戶管理: {'✅' if USER_MANAGER_AVAILABLE else '❌ (備用模式)'}")
    print(f"  認證系統: {'✅' if AUTH_AVAILABLE else '❌'} 統一JWT認證")
    print(f"  向量系統: {'✅' if VECTOR_SYSTEM_AVAILABLE else '❌ (知識庫功能不可用)'}")
    print(f"  對話記錄器: {'✅' if CONVERSATION_LOGGER_AVAILABLE else '❌'} PostgreSQL版")
    print(f"  機器人實例腳本: {'✅' if BOT_INSTANCE_SCRIPT.exists() else '❌'}")
    print("=" * 60)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
    except KeyboardInterrupt:
        print("\n正在關閉服務...")
    except Exception as e:
        print(f"❌ 服務器啟動失敗: {e}")
    finally:
        # 🆕 關閉所有數據庫連接
        try:
            server.close_all_loggers()
            print("✅ 所有數據庫連接已關閉")
        except Exception as cleanup_error:
            print(f"⚠️ 清理過程中出現警告: {cleanup_error}")

if __name__ == "__main__":
    main()