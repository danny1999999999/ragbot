#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chatbot_instance.py - (API相容版)

修復內容：
- __init__ 方法現在直接接收一個 config 字典，而不是 bot_name。
- 移除了從檔案系統讀取 JSON 設定的 _load_config 方法。

"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import json
import argparse
import sys
import re
import time
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from auth_middleware import OptionalAuth, User, JWTManager
from conversation_logger_simple import EnhancedConversationLogger as ConversationLogger


import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from config import app_config  # ⭐統一導入

# 🔧 檢查API模式依賴
try:
    import httpx
    HTTPX_AVAILABLE = True
    print("✅ httpx 可用，支援向量API模式")
except ImportError:
    HTTPX_AVAILABLE = False
    print("⚠️ httpx 不可用，僅支援直接向量存取模式")

# 🔧 檢查向量系統
try:
    from vector_builder_langchain import OptimizedVectorSystem
    VECTOR_SYSTEM_AVAILABLE = True
    print("✅ 向量系統可用")
except ImportError:
    VECTOR_SYSTEM_AVAILABLE = False
    print("❌ 向量系統不可用")

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent
# 🗑️ 移除：不再需要從檔案系統讀取設定
# BOT_CONFIGS_DIR = ROOT_DIR / "bot_configs"

class ChatbotInstance:
    # ✨ 關鍵更動：__init__ 現在接收一個完整的 config 字典
    def __init__(self, config: dict, **kwargs):
        if not config or not isinstance(config, dict):
            raise ValueError("A valid configuration dictionary must be provided.")
        
        self.config = config
        self.bot_name = self.config.get("bot_name")
        if not self.bot_name:
            raise ValueError("Config dictionary must contain a 'bot_name'.")

        self.collection_name = f"collection_{self.bot_name}"
        
        # 🔧 向量API配置
        self.vector_api_url = app_config.get_vector_api_url()
        self.use_api_mode = (
            HTTPX_AVAILABLE and 
            os.getenv("USE_VECTOR_API", "false").lower() == "true"
        )
        
        # 🔧 向量系統初始化 - 保持原有邏輯
        self.vector_system = None
        if VECTOR_SYSTEM_AVAILABLE:
            try:
                self.vector_system = OptimizedVectorSystem()
                logger.info(f"✅ 向量系統初始化成功 (機器人: {self.bot_name})")
            except Exception as e:
                logger.error(f"向量系統初始化失敗: {e}")
                self.vector_system = None
        
        # 🔧 模式選擇邏輯
        if self.use_api_mode and HTTPX_AVAILABLE:
            logger.info(f"🔗 使用向量API模式: {self.vector_api_url}")
            self.search_mode = "api"
        elif self.vector_system:
            logger.info("🔧 使用直接向量存取模式")
            self.search_mode = "direct"
        else:
            logger.warning("⚠️ 向量搜尋功能不可用")
            self.search_mode = "disabled"
        
        # 每個機器人實例使用獨立的對話記錄資料庫
        db_config = self._get_db_config()
        self.logger = ConversationLogger(db_config=db_config)
        
        self.app = FastAPI(title=f"{self.bot_name} Chatbot")
        try:
            from starlette.middleware.proxy_headers import ProxyHeadersMiddleware
            self.app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
            logger.info("✅ ProxyHeadersMiddleware 已啟用")
        except ImportError:
            logger.info("ℹ️ ProxyHeadersMiddleware 不可用，使用本地開發模式")
        except Exception as e:
            logger.warning(f"⚠️ ProxyHeadersMiddleware 啟用失敗: {e}")


        # CORS 支援
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 檢查並建立範本目錄
        self.template_dir = ROOT_DIR / "chatbot_templates" / "modern"
        self.static_dir = self.template_dir / "static"
        
        # 確保目錄存在
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        
        # 檢查必要檔案是否存在
        chat_html_path = self.template_dir / "chat.html"
        if not chat_html_path.exists():
            logger.warning(f"⚠️ 範本檔案不存在: {chat_html_path}")
            logger.warning("請確保 chat.html 位於 chatbot_templates/modern/ 目錄中")
        
        self.modern_templates = Jinja2Templates(directory=str(self.template_dir))
        
        # 靜態檔案掛載檢查
        if self.static_dir.exists():
            self.app.mount("/modern/static", StaticFiles(directory=str(self.static_dir)), name="modern_static")
            logger.info(f"✅ 靜態檔案掛載成功: {self.static_dir}")
        else:
            logger.warning(f"⚠️ 靜態檔案目錄不存在: {self.static_dir}")
        
        self.session_counters: Dict[str, int] = {}
        
        # 新增統計資訊
        self.total_conversations = 0
        self.successful_conversations = 0
        self.failed_conversations = 0
        # 記錄每個 session 已顯示的連線
        self.session_shown_links = {}
        
        self.setup_routes()
        logger.info(f"✅ 機器人實例 '{self.bot_name}' 初始化完成，對話記錄資料庫配置類型：{db_config.get('type')}")

    # 🗑️ 移除：不再需要從檔案系統讀取設定
    # def _load_config(self) -> dict:
    #     ...

    def _get_db_config(self) -> dict:
        """
        獲取資料庫設定。
        此版本強制要求使用 PostgreSQL，並透過 DATABASE_URL 進行設定。
        """
        database_url = os.getenv("DATABASE_URL")

        if not database_url:
            error_msg = "環境變數 DATABASE_URL 未設定。對話紀錄功能需要 PostgreSQL資料庫。"
            logger.error(f"❌ 機器人 ' {self.bot_name}' - {error_msg}")
            raise ValueError(error_msg)

        logger.info(f"✅ 機器人 ' {self.bot_name}' 將使用 PostgreSQL (透過 DATABASE_URL)記錄對話。")
        return {
            "type": "postgresql",
            "connection_string": database_url
        }
  
    def create_anonymous_user(self, request: Request) -> User:
        """建立匿名使用者物件"""
        client_ip = request.client.host if request.client else "unknown"
        return User(
            id=0, 
            username=f"anonymous_{client_ip}", 
            role="anonymous", 
            email="anonymous@example.com", 
            is_active=True,
            password_hash=""
        )

    def get_user_identifier(self, user: Optional[User], session_id: str) -> str:
        """獲取使用者唯一識別碼，用於對話記錄"""
        if user and user.id:
            return f"auth_user_{user.id}_{user.username}"
        else:
            return f"anonymous_session_{session_id}"

    def _get_document_content(self, doc) -> str:
        """統一獲取文件內容的方法"""
        # 嘗試常見的內容屬性名稱
        for attr in ['page_content', 'content', 'text']:
            if hasattr(doc, attr):
                content = getattr(doc, attr)
                if content:
                    return str(content)
        
        # 如果是字典格式
        if isinstance(doc, dict):
            for key in ['page_content', 'content', 'text']:
                if key in doc and doc[key]:
                    return str(doc[key])
        
        # 如果都沒有，返回字串表示
        return str(doc)

    # 🔧 新增：透過API搜尋向量
    async def _search_vectors_via_api(self, query: str, k: int = 3):
        """透過API搜尋向量"""
        if not HTTPX_AVAILABLE:
            logger.warning("httpx 不可用，無法使用API模式")
            return []
            
        try:
            timeout = httpx.Timeout(30.0, connect=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.vector_api_url}/api/search",
                    json={
                        "query": query,
                        "collection_name": self.collection_name,
                        "k": k
                    }
                )
                
                if response.status_code == 200:
                    response_data = response.json() # Safely get the list
                    api_results = response_data.get("results", []) 
                    
                    # 轉換為相容原有程式碼的文件格式
                    documents = []
                    if api_results: # Check if there are any results
                        for result in api_results:
                            # 建立相容的文件物件
                            doc = type('Document', (), {
                                'page_content': result.get('content', ''),
                                'metadata': result.get('metadata', {})
                            })()
                            documents.append(doc)
                    
                    logger.info(f"API搜尋成功: {len(documents)} 個結果")
                    return documents
                    
                elif response.status_code == 503:
                    logger.warning("向量API服務不可用")
                    return []
                else:
                    logger.error(f"API搜尋失敗: {response.status_code} - {response.text}")
                    return []
                    
        except httpx.TimeoutException:
            logger.error("向量搜尋API逾時")
            return []
        except httpx.ConnectError:
            logger.error("無法連線到向量API服務")
            return []
        except Exception as e:
            logger.error(f"向量搜尋API呼叫異常: {e}")
            return []

    # 🔧 保持原有的直接搜尋方法
    def _search_vectors_direct(self, query: str, k: int = 3):
        """直接搜尋向量"""
        if not self.vector_system:
            logger.warning("向量系統不可用")
            return []
        
        try:
            return self.vector_system.search(query, collection_name=self.collection_name, k=k)
        except Exception as e:
            logger.error(f"直接向量搜尋失敗: {e}")
            return []

    # 🔧 智慧向量搜尋方法
    async def _search_vectors_smart(self, query: str, k: int = 3):
        """智慧選擇搜尋方式"""
        context_docs = []
        
        # 主要模式嘗試
        if self.search_mode == "api":
            context_docs = await self._search_vectors_via_api(query, k)
            search_method = "API"
        elif self.search_mode == "direct":
            context_docs = self._search_vectors_direct(query, k)
            search_method = "直接"
        else:
            logger.warning("向量搜尋功能不可用")
            return []
        
        # 如果主要模式失敗，嘗試備用模式
        if not context_docs:
            if self.search_mode == "api" and self.vector_system:
                logger.warning("API模式無結果，嘗試直接模式...")
                context_docs = self._search_vectors_direct(query, k)
                if context_docs:
                    search_method = "直接(備用)"
                    logger.info(f"✅ 直接模式備用成功: {len(context_docs)} 個文件")
            elif self.search_mode == "direct" and HTTPX_AVAILABLE:
                logger.warning("直接模式無結果，嘗試API模式...")
                context_docs = await self._search_vectors_via_api(query, k)
                if context_docs:
                    search_method = "API(備用)"
                    logger.info(f"✅ API模式備用成功: {len(context_docs)} 個文件")
        
        if context_docs:
            logger.info(f"🔍 機器人 '{self.bot_name}' {search_method}檢索結果: {len(context_docs)} 個文件")
        else:
            logger.warning(f"🔍 機器人 '{self.bot_name}' 未找到相關文件")
        
        return context_docs

    # 保持原有的其他方法不變
    def get_chunk_info_safely(self, search_item, fallback_index: int) -> dict:
        """安全地從搜尋結果中獲取chunk資訊"""
        try:
            chunk_info = {
                'chunk_id': f'chunk_{fallback_index}',
                'chunk_index': fallback_index,
                'content': '',
                'source': 'unknown',
                'filename': 'unknown'
            }
            
            # 方法1: 處理不同類型的搜尋結果
            if hasattr(search_item, 'page_content'):
                # LangChain Document 物件
                chunk_info['content'] = search_item.page_content
                if hasattr(search_item, 'metadata') and search_item.metadata:
                    metadata = search_item.metadata
                    chunk_info['chunk_id'] = metadata.get('chunk_id', f'chunk_{fallback_index}')
                    chunk_info['chunk_index'] = metadata.get('chunk_index', fallback_index)
                    chunk_info['source'] = metadata.get('source', 'unknown')
                    chunk_info['filename'] = metadata.get('filename', metadata.get('original_filename', 'unknown'))
            
            elif isinstance(search_item, dict):
                # 字典格式的搜尋結果
                chunk_info['content'] = search_item.get('content', search_item.get('page_content', ''))
                metadata = search_item.get('metadata', {})
                chunk_info['chunk_id'] = metadata.get('chunk_id', f'chunk_{fallback_index}')
                chunk_info['chunk_index'] = metadata.get('chunk_index', fallback_index)
                chunk_info['source'] = metadata.get('source', 'unknown')
                chunk_info['filename'] = metadata.get('filename', metadata.get('original_filename', 'unknown'))
            
            elif hasattr(search_item, 'content'):
                # 自訂搜尋結果物件
                chunk_info['content'] = str(search_item.content)
                if hasattr(search_item, 'metadata'):
                    metadata = search_item.metadata
                    chunk_info['chunk_id'] = metadata.get('chunk_id', f'chunk_{fallback_index}')
                    chunk_info['chunk_index'] = metadata.get('chunk_index', fallback_index)
                    chunk_info['source'] = metadata.get('source', 'unknown')
                    chunk_info['filename'] = metadata.get('filename', 'unknown')
            
            else:
                # 備用：將整個物件轉為字串
                chunk_info['content'] = str(search_item)
            
            return chunk_info
            
        except Exception as e:
            logger.warning(f"獲取chunk資訊失敗: {e}")
            return {
                'chunk_id': f'chunk_{fallback_index}',
                'chunk_index': fallback_index,
                'content': str(search_item) if search_item else '',
                'source': 'unknown',
                'filename': 'unknown'
            }

    def _find_real_chunk_index(self, content: str, fallback_index: int) -> int:
        """尋找內容在向量庫中的真實索引"""
        try:
            if self.vector_system:
                vectorstore = self.vector_system.get_or_create_vectorstore(self.collection_name)
                all_docs = vectorstore.get()
                
                if all_docs and all_docs.get('documents'):
                    # 尋找完全符合的內容
                    for idx, doc_content in enumerate(all_docs['documents']):
                        if doc_content.strip() == content.strip():
                            return idx
                    
                    # 如果沒有完全符合，尋找部分符合
                    for idx, doc_content in enumerate(all_docs['documents']):
                        if content[:100] in doc_content or doc_content[:100] in content:
                            return idx
            
            # 如果都找不到，返回備用索引
            return fallback_index
            
        except Exception as e:
            logger.debug(f"尋找真實chunk索引失敗: {e}")
            return fallback_index

    def _get_chunk_index_from_doc(self, doc, fallback_index: int) -> int:
        """從文件物件中獲取 chunk 的實際索引 - 改進版"""
        try:
            logger.debug(f"🔍 嘗試獲取文件的真實索引，fallback: {fallback_index}")
            
            # 方法1: 從 metadata 中獲取 - 改進版
            metadata = None
            if hasattr(doc, 'metadata') and doc.metadata:
                metadata = doc.metadata
            elif isinstance(doc, dict) and 'metadata' in doc:
                metadata = doc['metadata']
            
            if metadata:
                # 🆕 記錄元資料內容用於偵錯
                logger.debug(f"📋 文件元資料: {list(metadata.keys())}")
                
                # 擴展搜尋欄位
                index_fields = [
                    'chunk_index', 'index', 'chunk_id', 'id', 
                    'chunk_num', 'chunk_number', 'doc_index',
                    'document_index', 'position', 'seq'
                ]
                
                for index_field in index_fields:
                    if index_field in metadata:
                        value = metadata[index_field]
                        logger.debug(f"📍 找到索引欄位 {index_field}: {value} (type: {type(value)})")
                        
                        if isinstance(value, int) and value >= 0:
                            logger.info(f"✅ 從 metadata.{index_field} 獲取索引: {value}")
                            return value
                        elif isinstance(value, str):
                            # 嘗試多種字串格式
                            if value.startswith('chunk_'):
                                try:
                                    idx = int(value.split('_')[-1])
                                    logger.info(f"✅ 從 metadata.{index_field} 解析索引: {idx}")
                                    return idx
                                except ValueError:
                                    continue
                            elif value.isdigit():
                                idx = int(value)
                                logger.info(f"✅ 從 metadata.{index_field} 轉換索引: {idx}")
                                return idx
            
            # 方法2: 從文件ID中提取索引 - 保持不變
            doc_id = None
            if hasattr(doc, 'id') and doc.id:
                doc_id = str(doc.id)
            elif isinstance(doc, dict) and 'id' in doc:
                doc_id = str(doc['id'])
            
            if doc_id:
                logger.debug(f"📄 文件ID: {doc_id}")
                # 嘗試多種ID格式
                patterns = [
                    r'_(\d+)$',           # ending with _123
                    r'chunk_(\d+)',       # chunk_123
                    r'doc_(\d+)',         # doc_123
                    r'-(\d+)$',           # ending with -123
                    r'(\d+)$'             # ending with 123
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, doc_id)
                    if match:
                        idx = int(match.group(1))
                        logger.info(f"✅ 從文件ID解析索引: {idx} (pattern: {pattern})")
                        return idx
            
            # 方法3: 從向量資料庫查詢實際位置 - 改進版
            if self.vector_system:
                try:
                    logger.debug(f"🔍 嘗試從向量資料庫查詢實際位置...")
                    vectorstore = self.vector_system.get_or_create_vectorstore(self.collection_name)
                    all_docs_result = vectorstore.get()
                    
                    if all_docs_result and all_docs_result.get('documents'):
                        all_documents = all_docs_result['documents']
                        doc_content = self._get_document_content(doc)
                        
                        # 🆕 改進：嘗試多種符合策略
                        
                        # 策略1: 完全符合
                        for idx, stored_doc_content in enumerate(all_documents):
                            if stored_doc_content.strip() == doc_content.strip():
                                logger.info(f"✅ 透過完全符合找到索引: {idx}")
                                return idx
                        
                        # 策略2: 前100字元符合
                        doc_prefix = doc_content[:100].strip()
                        if len(doc_prefix) > 20:  # 確保有足夠的內容進行符合
                            for idx, stored_doc_content in enumerate(all_documents):
                                stored_prefix = stored_doc_content[:100].strip()
                                if doc_prefix == stored_prefix:
                                    logger.info(f"✅ 透過前綴符合找到索引: {idx}")
                                    return idx
                        
                        # 策略3: 關鍵詞符合（如果內容較短）
                        if len(doc_content) < 500:
                            doc_words = set(doc_content.split())
                            if len(doc_words) > 3:  # 至少要有幾個詞
                                best_match_idx = -1
                                best_match_score = 0
                                
                                for idx, stored_doc_content in enumerate(all_documents):
                                    stored_words = set(stored_doc_content.split())
                                    common_words = doc_words.intersection(stored_words)
                                    score = len(common_words) / len(doc_words.union(stored_words))
                                    
                                    if score > 0.8 and score > best_match_score:
                                        best_match_score = score
                                        best_match_idx = idx
                                
                                if best_match_idx >= 0:
                                    logger.info(f"✅ 透過詞彙符合找到索引: {best_match_idx} (相似度: {best_match_score:.2f})")
                                    return best_match_idx
                    
                    logger.warning("⚠️ 向量資料庫查詢無符合結果")
                            
                except Exception as query_error:
                    logger.warning(f"⚠️ 查詢向量資料庫索引失敗: {query_error}")
            
            # 方法4: 使用備用索引
            logger.warning(f"⚠️ 所有方法都失敗，使用備用索引: {fallback_index}")
            return fallback_index
            
        except Exception as e:
            logger.error(f"❌ 獲取 chunk 索引時發生異常: {e}")
            return fallback_index

    def setup_routes(self):
        """設定路由"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_modern_chat_ui(request: Request):
            """主聊天介面"""
            try:
                display_name = self.config.get("display_name", self.bot_name)
                return self.modern_templates.TemplateResponse(
                    "chat.html", 
                    {
                        "request": request, 
                        "bot_name": self.bot_name,
                        "display_name": display_name
                    }
                )
            except Exception as e:
                logger.error(f"❌ 載入聊天介面失敗: {e}")
                return HTMLResponse(f"""
                <!DOCTYPE html>
                <html>
                <head><title>{self.bot_name}</title></head>
                <body>
                    <h1>聊天介面載入失敗</h1>
                    <p>錯誤：{e}</p>
                    <p>請確保範本檔案是否存在於 chatbot_templates/modern/chat.html</p>
                </body>
                </html>
                """, status_code=500)

        @self.app.post("/api/chat")
        async def chat(request: Request, user: User = Depends(OptionalAuth)):
            """聊天API端點 - 支援API和直接兩種模式"""
            start_time = time.time()
            conversation_id = None
            chunk_references = []
            
            try:
                data = await request.json()
                query = data.get("message")
                session_id = data.get("session_id", "default_session")
                
                if not query:
                    raise HTTPException(status_code=400, detail="Message cannot be empty")
                
                # 處理使用者識別
                if not user:
                    user = self.create_anonymous_user(request)
                
                user_identifier = self.get_user_identifier(user, session_id)
                
                # 更新統計資訊
                self.total_conversations += 1
                if session_id not in self.session_counters:
                    self.session_counters[session_id] = 0
                self.session_counters[session_id] += 1
                self._current_query = query
                logger.info(f"📩 機器人 '{self.bot_name}' 收到查詢 [{user_identifier}]: {query[:50]}...")

                # 🔧 修改：使用智慧向量搜尋
                context_docs = await self._search_vectors_smart(query, k=3)
                
                # 詳細偵錯檢索結果
                logger.info(f"🔍 目前查詢: {query}")
                logger.info(f"📄 檢索到 {len(context_docs) if context_docs else 0} 個文件")

                if context_docs:
                    for i, doc in enumerate(context_docs):
                        content_preview = self._get_document_content(doc)[:100]
                        metadata = getattr(doc, 'metadata', {})
                        contained_urls = metadata.get('contained_urls', '')
                        
                        logger.info(f"📄 文件 {i+1}:")
                        logger.info(f"  內容預覽: {content_preview}")
                        logger.info(f"  包含連線: {contained_urls}")
                        logger.info(f"  來源檔案: {metadata.get('filename', 'unknown')}")

                context = "\n".join([self._get_document_content(doc) for doc in context_docs]) if context_docs else ""
                
                # 準備 chunk 引用資訊
                retrieved_docs_content = []
                
                if context_docs:
                    for i, doc in enumerate(context_docs):
                        doc_content = self._get_document_content(doc)
                        retrieved_docs_content.append(doc_content)
                        
                        # 獲取元資料
                        metadata = {}
                        if hasattr(doc, 'metadata') and doc.metadata:
                            metadata = doc.metadata
                        elif isinstance(doc, dict) and 'metadata' in doc:
                            metadata = doc['metadata']
                        
                        # 修正：直接從 metadata 獲取真實的 chunk_id
                        chunk_id = metadata.get('chunk_id', f'fallback_{i}')

                        chunk_ref = {
                            "chunk_id": chunk_id,  # <-- 使用真實的 chunk_id
                            "index": i, # 使用循環的索引 i 作為備用
                            "content_preview": doc_content[:100] + "..." if len(doc_content) > 100 else doc_content,
                            "source": metadata.get('source', 'unknown'),
                            "filename": metadata.get('filename', metadata.get('original_filename', 'unknown'))
                        }
                        chunk_references.append(chunk_ref)

                logger.info(f"🔍 機器人 '{self.bot_name}' 檢索結果: {len(context_docs)} 個文件, chunk_refs: {len(chunk_references)}")
                
                # 生成回應
                system_prompt = self.config.get("system_role", "你是一個樂於助人的 AI 助理。")
                response_text, recommended_questions = self._generate_response(
                    query, context, system_prompt, session_id
                )
                
                # 處理引用來源
                logger.info(f"[Cite Sources] Checking... Enabled: {self.config.get('cite_sources_enabled', False)}, Docs found: {bool(context_docs)}")
                if self.config.get("cite_sources_enabled", False) and context_docs:
                    all_sources = self._extract_source_urls(context_docs)
                    logger.info(f"[Cite Sources] _extract_source_urls returned {len(all_sources)} potential sources.")

                    # 只處理包含真實URL的來源
                    url_sources = [s for s in all_sources if s.get("url")]
                    logger.info(f"[Cite Sources] Found {len(url_sources)} items with a 'url' key.")
                    
                    if url_sources:
                        # 過濾重複連線
                        url_sources = self._filter_duplicate_links(url_sources, session_id)
                        logger.info(f"[Cite Sources] After filtering duplicates, {len(url_sources)} sources remain.")
                        
                        if url_sources:  # 確保過濾後還有連線
                            response_text += self._format_source_links(url_sources)
                            logger.info(f"🔗 新增了 {len(url_sources)} 個參考連結")
                        else:
                            logger.info("🔄 所有連線都是重複的，跳過顯示")
                    else:
                        logger.info("📝 未在文件內容中找到可引用的URL")

                processing_time_ms = int((time.time() - start_time) * 1000)
                
                # 記錄對話
                try:
                    conversation_id = self.logger.log_conversation(
                        user_id=user_identifier,
                        user_query=query,
                        ai_response=response_text,
                        collection_used=self.collection_name if context_docs else None,
                        retrieved_docs=retrieved_docs_content[:3],
                        doc_similarities=[],
                        processing_time_ms=processing_time_ms,
                        is_image_generation=False,
                        error_occurred=False,
                        error_message=None,
                        authenticated_user_id=user.id if user and hasattr(user, 'id') and user.id else None,
                        user_role=user.role if user and hasattr(user, 'role') else "anonymous",
                        chunk_references=chunk_references
                    )
                    
                    self.successful_conversations += 1
                    logger.info(f"✅ 對話記錄成功：ID={conversation_id}, 機器人：{self.bot_name}, chunks={len(chunk_references)}")
                    
                except Exception as log_error:
                    logger.error(f"❌ 記錄對話失敗（機器人：{self.bot_name}）: {log_error}")

                logger.info(f"📤 機器人 '{self.bot_name}' API 回應偵錯:")
                logger.info(f"  - response_text 長度: {len(response_text)}")
                logger.info(f"  - recommended_questions: {recommended_questions}")
                logger.info(f"  - 找到文件數量: {len(context_docs) if context_docs else 0}")
                logger.info(f"  - chunk_references: {len(chunk_references)}")
                logger.info(f"  - 處理時間: {processing_time_ms}ms")

                return JSONResponse({
                    "response": response_text,
                    "recommended_questions": recommended_questions if recommended_questions else [],
                    "error": False,
                    "metadata": {
                        "bot_name": self.bot_name,
                        "session_id": session_id,
                        "processing_time_ms": processing_time_ms,
                        "documents_found": len(context_docs) if context_docs else 0,
                        "conversation_id": conversation_id,
                        "chunk_count": len(chunk_references),
                        "search_mode": self.search_mode
                    }
                })

            except HTTPException as http_exc:
                error_message = str(http_exc.detail)
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                try:
                    user_identifier = self.get_user_identifier(user, session_id) if 'session_id' in locals() else "unknown"
                    query = data.get("message", "未知查詢") if 'data' in locals() else "未知查詢"
                    
                    conversation_id = self.logger.log_conversation(
                        user_id=user_identifier,
                        user_query=query,
                        ai_response=f"HTTP錯誤: {error_message}",
                        collection_used=None,
                        retrieved_docs=[],
                        doc_similarities=[],
                        processing_time_ms=processing_time_ms,
                        is_image_generation=False,
                        error_occurred=True,
                        error_message=error_message,
                        authenticated_user_id=user.id if user and hasattr(user, 'id') and user.id else None,
                        user_role=user.role if user and hasattr(user, 'role') else "anonymous"
                    )
                    
                    self.failed_conversations += 1
                    logger.error(f"❌ HTTP錯誤記錄：{conversation_id}, 機器人：{self.bot_name}")
                    
                except Exception as log_error:
                    logger.error(f"❌ HTTP錯誤對話記錄失敗（機器人：{self.bot_name}）: {log_error}")
                
                raise http_exc
                
            except Exception as e:
                error_message = str(e)
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                logger.error(f"❌ 機器人 '{self.bot_name}' 聊天 API 發生嚴重錯誤: {e}", exc_info=True)
                
                try:
                    user_identifier = self.get_user_identifier(user, session_id) if 'session_id' in locals() else "unknown"
                    query = data.get("message", "未知查詢") if 'data' in locals() else "未知查詢"
                    
                    conversation_id = self.logger.log_conversation(
                        user_id=user_identifier,
                        user_query=query,
                        ai_response=f"系統錯誤: {error_message}",
                        collection_used=None,
                        retrieved_docs=[],
                        doc_similarities=[],
                        processing_time_ms=processing_time_ms,
                        is_image_generation=False,
                        error_occurred=True,
                        error_message=error_message,
                        authenticated_user_id=user.id if user and hasattr(user, 'id') and user.id else None,
                        user_role=user.role if user and hasattr(user, 'role') else "anonymous"
                    )
                    
                    self.failed_conversations += 1
                    logger.error(f"❌ 系統錯誤記錄：{conversation_id}, 機器人：{self.bot_name}")
                    
                except Exception as log_error:
                    logger.error(f"❌ 系統錯誤對話記錄失敗（機器人：{self.bot_name}）: {log_error}")
                
                return JSONResponse({
                    "response": f"抱歉，機器人 '{self.bot_name}' 發生內部錯誤，請稍後再試。",
                    "error": True,
                    "error_message": error_message,
                    "recommended_questions": [],
                    "metadata": {
                        "bot_name": self.bot_name,
                        "processing_time_ms": processing_time_ms,
                        "conversation_id": conversation_id,
                        "search_mode": self.search_mode
                    }
                }, status_code=500)

        @self.app.get("/api/stats")
        async def get_bot_stats():
            """獲取機器人統計資訊"""
            try:
                db_stats = self.logger.get_statistics() if hasattr(self.logger, 'get_statistics') else {}
                
                return JSONResponse({
                    "bot_name": self.bot_name,
                    "display_name": self.config.get("display_name", self.bot_name),
                    "collection_name": self.collection_name,
                    "db_type": "postgresql",
                    "search_mode": self.search_mode,
                    "vector_api_url": self.vector_api_url if self.search_mode == "api" else None,
                    "session_stats": {
                        "total_conversations": self.total_conversations,
                        "successful_conversations": self.successful_conversations,
                        "failed_conversations": self.failed_conversations,
                        "active_sessions": len(self.session_counters)
                    },
                    "db_stats": db_stats,
                    "config": {
                        "temperature": self.config.get("temperature", 0.7),
                        "max_tokens": self.config.get("max_tokens", 2000),
                        "dynamic_recommendations_enabled": self.config.get("dynamic_recommendations_enabled", False),
                        "cite_sources_enabled": self.config.get("cite_sources_enabled", False)
                    }
                })
            except Exception as e:
                logger.error(f"獲取機器人統計失敗: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/health")
        async def health_check():
            """健康檢查端點"""
            return JSONResponse({
                "status": "healthy",
                "bot_name": self.bot_name,
                "display_name": self.config.get("display_name", self.bot_name),
                "search_mode": self.search_mode,
                "timestamp": time.time(),
                "conversation_count": self.total_conversations
            })

    def _generate_response(self, query: str, context: str, system_prompt: str, session_id: str) -> Tuple[str, List[str]]:
        """生成回應"""
        if not OPENAI_AVAILABLE:
            return "系統 AI 模組未載入。", []

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return "系統配置錯誤：缺少 OpenAI API Key。", []

        llm = ChatOpenAI(
            model=self.config.get("model", "gpt-4o-mini"), 
            temperature=self.config.get("temperature", 0.7), 
            api_key=openai_key
        )

        # 生成主要回答
        main_answer_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""參考資料：
{context}

使用者問題：{query}""" )
        ]
        main_response = llm.invoke(main_answer_messages)
        main_answer = main_response.content.strip()

        # 生成推薦問題
        recommended_questions = []
        should_recommend = self.config.get("dynamic_recommendations_enabled", False)
        recommend_count = self.config.get("dynamic_recommendations_count", 0)
        conversation_count = self.session_counters.get(session_id, 1)

        logger.info(f"🔍 機器人 '{self.bot_name}' 推薦問題偵錯:")
        logger.info(f"  - should_recommend: {should_recommend}")
        logger.info(f"  - recommend_count: {recommend_count}")
        logger.info(f"  - conversation_count: {conversation_count}")

        if should_recommend and (recommend_count == 0 or conversation_count <= recommend_count):
            logger.info(f"✅ 機器人 '{self.bot_name}' 開始生成推薦問題...")
            
            recommend_prompt = f"""
**原始對話**
使用者問：「{query}」
你的回答：「{main_answer}」

---
**指令**
根據以上對話，生成三個相關的延伸問題。

**格式**
- 每個問題一行
- 不要編號
- 不要包含任何其他文字"""
            
            try:
                recommend_messages = [HumanMessage(content=recommend_prompt)]
                recommend_response = llm.invoke(recommend_messages)
                questions_text = recommend_response.content.strip()
                
                logger.info(f"🤖 機器人 '{self.bot_name}' LLM 原始回應:")
                logger.info(f"'{questions_text}'")
                
                questions_list = [q.strip() for q in questions_text.split('\n') if q.strip()]
                recommended_questions = list(dict.fromkeys(questions_list))
                
                logger.info(f"📝 機器人 '{self.bot_name}' 解析後的推薦問題: {recommended_questions}")
                
            except Exception as e:
                logger.error(f"❌ 機器人 '{self.bot_name}' 生成推薦問題時發生錯誤: {e}", exc_info=True)
                recommended_questions = []
        else:
            logger.info(f"❌ 機器人 '{self.bot_name}' 推薦問題未啟用或超出限制")

        return main_answer, recommended_questions
    
    def _extract_links_from_content(self, content: str) -> List[dict]:
        """從文件內容中提取連結和標題（穩健判定 URL/標題順序）- 修復版本"""
        links = []
        seen_urls = set()  # 🔧 新增：URL去重集合
        
        #正規表示式樣版符合各種連結格式
        patterns = [
            # Markdown格式: [標題](URL) -> (title, url)
            r'\[([^\]]+)\]\((https?://[^\s)]+)\)',
            # HTML格式: <a href="URL">標題</a> -> (url, title)
            r'<a[^>]+href=["\\]([^\"\\]+)["\\][^>]*>([^<]+)</a>',
            # 純文字格式: 標題: URL  -> (title, url)
            r'([^:\n]+):\s*(https?://[^\s]+)',
            # 純文字格式: 標題 - URL  -> (title, url)
            r'([^-\n]+)\s*-\s*(https?://[^\s]+)'
        ]
        
        url_like = re.compile(r'^https?://', re.IGNORECASE)

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) != 2:
                    continue
                a, b = match[0].strip(), match[1].strip()

                # 以是否像 URL 來決定欄位對應，統一輸出為 (title, url)
                if url_like.match(a) and not url_like.match(b):
                    url, title = a, b
                elif url_like.match(b) and not url_like.match(a):
                    url, title = b, a
                else:
                    # 回退策略：沿用原本 (title, url) 假設
                    title, url = a, b

                # 🔧 新增：檢查URL是否已存在
                if url in seen_urls:
                    continue

                # 清理標題
                title = title.strip().strip('"\'')
                if not title or len(title) < 3 or len(title) > 200:
                    continue

                # 排除無意義的標題
                if re.match(r'^(點擊這裡|閱讀更多|更多資訊|連結|網址|click here|read more|more info|link|url)$', title, re.IGNORECASE):
                    continue
                if re.match(r'^\d+$', title) or re.match(r'^[^\w\u4e00-\u9fff]+$', title):
                    continue

                links.append({"title": title, "url": url})
                seen_urls.add(url)  # 🔧 新增：記錄已處理的URL

        return links
    
    def _find_title_for_url_in_content(self, content: str, target_url: str) -> str:
        """在文件內容中尋找特定URL對應的標題"""
        try:
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if target_url in line:
                    # 查看目前行和前後幾行尋找標題
                    context_lines = []
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context_lines = lines[start:end]
                    
                    for context_line in context_lines:
                        # 跳過包含URL的行
                        if target_url in context_line:
                            continue
                        
                        # 尋找標題樣板
                        context_line = context_line.strip()
                        if context_line and len(context_line) > 5 and len(context_line) < 200:
                            # 移除常見的標記符號
                            cleaned = re.sub(r'^[#*\-\•\d\.\s]+', '', context_line)
                            if cleaned and len(cleaned) > 5:
                                return cleaned
            
            return ""
            
        except Exception as e:
            logger.debug(f"在內容中尋找標題失敗: {e}")
            return ""

    def _generate_smart_title(self, metadata: dict, url: str) -> str:
        """智慧生成標題"""
        try:
            # 優先級1: 使用檔案標題
            title = metadata.get('title', '')
            if title and title != 'unknown' and len(title) > 3:
                return title
            
            # 優先級2: 使用原始檔案名稱（去除副檔名）
            filename = metadata.get('original_filename', metadata.get('filename', ''))
            if filename and filename != 'unknown':
                title = os.path.splitext(filename)[0]
                # 美化檔案名稱
                title = title.replace('-', ' ').replace('_', ' ')
                title = ' '.join(word.capitalize() for word in title.split() if word)
                if len(title) > 3:
                    return title
            
            # 優先級3: 從URL路徑提取智慧標題
            from urllib.parse import urlparse, unquote
            parsed = urlparse(url)
            
            # 獲取路徑中的有用資訊
            path_parts = [part for part in parsed.path.strip('/').split('/') if part]
            if path_parts:
                # 使用最後一個有意義的路徑段
                last_part = path_parts[-1]
                # URL解碼
                last_part = unquote(last_part)
                # 移除常見的副檔名
                last_part = re.sub(r'\.(html|php|aspx|jsp|htm)$', '', last_part)
                # 替換分隔符並美化
                title = last_part.replace('-', ' ').replace('_', ' ')
                title = ' '.join(word.capitalize() for word in title.split() if word)
                if len(title) > 3:
                    return title
            
            # 優先級4: 使用查詢參數中的資訊
            if parsed.query:
                query_parts = parsed.query.split('&')
                for part in query_parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        if key in ['title', 'q', 'query', 'search'] and value:
                            decoded_value = unquote(value).replace('+', ' ')
                            if len(decoded_value) > 3:
                                return decoded_value
            
            # 優先級5: 使用域名（最後的備選）
            domain = parsed.netloc
            if domain:
                # 移除www前綴
                domain = re.sub(r'^www\.', '', domain)
                return f"來源：{domain}"
            
            return url
            
        except Exception as e:
            logger.warning(f"生成智慧標題失敗: {e}")
            return self._extract_domain_from_url(url)

    def _extract_source_urls(self, docs: List) -> List[dict]:
        """從文件元資料中提取URL和完整標題資訊"""
        sources = []
        seen_urls = set()

        for doc in docs:
            try:
                metadata = getattr(doc, 'metadata', {})
                
                # 方法1: 檢查是否有預處理的標題-URL對應
                title_url_mapping = metadata.get('title_url_mapping', {})
                if isinstance(title_url_mapping, str):
                    try:
                        title_url_mapping = json.loads(title_url_mapping)
                    except json.JSONDecodeError:
                        title_url_mapping = {}
                
                # 如果有對應，直接使用
                if title_url_mapping:
                    for title, url in title_url_mapping.items():
                        if url not in seen_urls:
                            sources.append({
                                "title": title,
                                "url": url
                            })
                            seen_urls.add(url)
                    continue
                
                # 方法2: 從文件內容中提取標題和URL
                doc_content = self._get_document_content(doc)
                extracted_links = self._extract_links_from_content(doc_content)
                for link_info in extracted_links:
                    if link_info['url'] not in seen_urls:
                        sources.append(link_info)
                        seen_urls.add(link_info['url'])
                
                # 🆕 方法2.5: 如果沒有找到格式化連線，則從內容中提取原始URL
                if not extracted_links:
                    raw_urls = re.findall(r'https?://[^\s<>"\\]+', doc_content)
                    for url in raw_urls:
                        if url not in seen_urls:
                            title = self._generate_smart_title(metadata, url)
                            sources.append({"title": title, "url": url})
                            seen_urls.add(url)
                
                # 方法3: 從contained_urls獲取URL，然後嘗試符合標題
                url_string = metadata.get('contained_urls', '')
                if url_string and not extracted_links:
                    urls_in_chunk = [url.strip() for url in url_string.split('|') if url.strip()]
                    
                    for url in urls_in_chunk:
                        if url not in seen_urls:
                            # 嘗試從文件內容中找到對應的標題
                            title = self._find_title_for_url_in_content(doc_content, url)
                            if not title:
                                # 如果找不到，使用智慧標題提取
                                title = self._generate_smart_title(metadata, url)
                            
                            sources.append({
                                "title": title,
                                "url": url
                            })
                            seen_urls.add(url)

            except Exception as e:
                logger.warning(f"從元資料提取URL時出錯: {e}")

        logger.info(f"從元資料中提取到 {len(sources)} 個連線")
        return sources

    def _is_valid_url(self, url: str) -> bool:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return (parsed.scheme in ['http', 'https'] and 
                    parsed.netloc and 
                    '.' in parsed.netloc and
                    len(url) > 10)
        except:
            return False


    def _extract_domain_from_url(self, url: str) -> str:
        """從URL中提取域名作為標題"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            return domain if domain else url
        except:
            return url

    def _filter_duplicate_links(self, sources: List[dict], session_id: str) -> List[dict]:
        """過濾重複連線"""
        if session_id not in self.session_shown_links:
            self.session_shown_links[session_id] = set()
        
        shown_links = self.session_shown_links[session_id]
        filtered_sources = []
        
        for source in sources:
            url = source.get('url', '')
            if url and url not in shown_links:
                filtered_sources.append(source)
                shown_links.add(url)
                logger.info(f"✅ 新連線: {source.get('title', 'unknown')}")
            else:
                logger.info(f"🔄 跳過重複連線: {source.get('title', 'unknown')}")
        
        return filtered_sources

    def _format_source_links(self, sources: List[dict]) -> str:
        """格式化參考連結 - 確保正確換行"""
        if not sources:
            return ""
        
        # ✅ 修正：在開頭增加一個額外的換行符，來產生更多間距
        source_links = "\n\n\n💡 你可能想知道\n\n"  # 簡化版本
        
        formatted_items = []
        for source in sources:
            title = source["title"]
            url = source["url"]
            formatted_items.append(f"- [{title}]({url})")
        
        source_links += "\n".join(formatted_items)
        return source_links
    
    def _clean_and_validate_title(self, title: str) -> str:
        """清理"""
        if not title:
            return ""
        
        # 清理標題
        title = title.strip()
        title = re.sub(r'\s+', ' ', title)  # 合併多個空格
        title = re.sub(r'^[^\w\u4e00-\u9fff]+', '', title)  # 移除開頭的符號
        title = re.sub(r'[^\w\u4e00-\u9fff]+$', '', title)  # 移除結尾的符號
        
        # 驗證標題長度
        if len(title) < 3 or len(title) > 200:
            return ""
        
        # 排除無意義的標題
        meaningless_patterns = [
            r'^(點擊這裡|閱讀更多|更多資訊|連線|網址|click here|read more|more info|link|url)$',
            r'^\d+$',  # 純數字
            r'^[^\w\u4e00-\u9fff]+$',  # 只有符號
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, title, re.IGNORECASE):
                return ""
        
        return title


def main():
    # ✨ 關鍵更動：獨立啟動時需要從資料庫加載設定
    parser = argparse.ArgumentParser(description="啟動一個獨立的聊天機器人實例。")
    parser.add_argument("--bot-name", type=str, required=True, help="要啟動的機器人名稱")
    args = parser.parse_args()

    try:
        # 獨立啟動時，需要一個方法來從DB獲取設定
        # 這需要 BotConfigManager 的一個實例
        from bot_config_manager import BotConfigManager
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set.")
        
        config_manager = BotConfigManager(database_url)
        config = config_manager.get_bot_config(args.bot_name)
        
        if not config:
            raise ValueError(f"Bot '{args.bot_name}' not found in the database.")

        instance = ChatbotInstance(config=config)
        port = instance.config.get("port")
        if not port:
            raise ValueError(f"Port not specified in the config for bot '{args.bot_name}'.")
        
        logger.info(f"🤖 機器人 '{instance.bot_name}' 正在 http://localhost:{port} 上啟動")
        logger.info(f"📊 對話記錄資料庫已在實例中配置。")
        logger.info(f"📚 知識庫集合：{instance.collection_name}")
        logger.info(f"🔍 搜尋模式：{instance.search_mode}")
        if instance.search_mode == "api":
            logger.info(f"🔗 向量API地址：{instance.vector_api_url}")
        logger.info(f"🔧 偵錯模式已啟用，將輸出詳細的對話記錄和推薦問題生成日誌")
        
        uvicorn.run(instance.app, host="0.0.0.0", port=int(port))

    except Exception as e:
        logger.error(f"❌ 啟動機器人 '{args.bot_name}' 失敗: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
