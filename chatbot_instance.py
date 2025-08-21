#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chatbot_instance.py -  (API兼容版)

修复内容：

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
from config import app_config  # ⭐ 統一導入

# 🔧 检查API模式依赖
try:
    import httpx
    HTTPX_AVAILABLE = True
    print("✅ httpx 可用，支持向量API模式")
except ImportError:
    HTTPX_AVAILABLE = False
    print("⚠️ httpx 不可用，仅支持直接向量访问模式")

# 🔧 检查向量系统
try:
    from vector_builder_langchain import OptimizedVectorSystem
    VECTOR_SYSTEM_AVAILABLE = True
    print("✅ 向量系统可用")
except ImportError:
    VECTOR_SYSTEM_AVAILABLE = False
    print("❌ 向量系统不可用")

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent
BOT_CONFIGS_DIR = ROOT_DIR / "bot_configs"

class ChatbotInstance:
    def __init__(self, bot_name: str, **kwargs):
        self.bot_name = bot_name
        self.config = self._load_config()
        self.collection_name = f"collection_{self.bot_name}"
        
        # 🔧 向量API配置
        self.vector_api_url = app_config.get_vector_api_url()
        self.use_api_mode = (
            HTTPX_AVAILABLE and 
            os.getenv("USE_VECTOR_API", "false").lower() == "true"
        )
        
        # 🔧 向量系统初始化 - 保持原有逻辑
        self.vector_system = None
        if VECTOR_SYSTEM_AVAILABLE:
            try:
                self.vector_system = OptimizedVectorSystem()
                logger.info(f"✅ 向量系统初始化成功 (机器人: {bot_name})")
            except Exception as e:
                logger.error(f"向量系统初始化失败: {e}")
                self.vector_system = None
        
        # 🔧 模式选择逻辑
        if self.use_api_mode and HTTPX_AVAILABLE:
            logger.info(f"🔗 使用向量API模式: {self.vector_api_url}")
            self.search_mode = "api"
        elif self.vector_system:
            logger.info("🔧 使用直接向量访问模式")
            self.search_mode = "direct"
        else:
            logger.warning("⚠️ 向量搜索功能不可用")
            self.search_mode = "disabled"
        
        # 每个机器人实例使用獨立的對話記錄資料庫
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


        # CORS 支持
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 检查并创建模板目录
        self.template_dir = ROOT_DIR / "chatbot_templates" / "modern"
        self.static_dir = self.template_dir / "static"
        
        # 确保目录存在
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查必要文件是否存在
        chat_html_path = self.template_dir / "chat.html"
        if not chat_html_path.exists():
            logger.warning(f"⚠️ 模板文件不存在: {chat_html_path}")
            logger.warning("请确保 chat.html 位于 chatbot_templates/modern/ 目录中")
        
        self.modern_templates = Jinja2Templates(directory=str(self.template_dir))
        
        # 静态文件挂载检查
        if self.static_dir.exists():
            self.app.mount("/modern/static", StaticFiles(directory=str(self.static_dir)), name="modern_static")
            logger.info(f"✅ 静态文件挂载成功: {self.static_dir}")
        else:
            logger.warning(f"⚠️ 静态文件目录不存在: {self.static_dir}")
        
        self.session_counters: Dict[str, int] = {}
        
        # 添加统计信息
        self.total_conversations = 0
        self.successful_conversations = 0
        self.failed_conversations = 0
        # 记录每个 session 已显示的连接
        self.session_shown_links = {}
        
        self.setup_routes()
        logger.info(f"✅ 机器人实例 '{self.bot_name}' 初始化完成，对话记录数据库配置类型：{db_config.get('type')}")

    def _load_config(self) -> dict:
        """载入机器人配置"""
        config_path = BOT_CONFIGS_DIR / f"{self.bot_name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"机器人设定文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            config = json.load(f)
        
        # 向后兼容：如果没有 display_name，自动添加
        if "display_name" not in config:
            config["display_name"] = config.get("bot_name", self.bot_name)
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=4)
                logger.info(f"为机器人 {self.bot_name} 自动添加显示名称: {config['display_name']}")
            except Exception as e:
                logger.warning(f"自动保存显示名称失败: {e}")
        
        return config

    def _get_db_config(self) -> dict:
        """Gets the database configuration, forcing PostgreSQL on Railway."""
        database_url = os.getenv("DATABASE_URL")

        if not database_url:
            error_msg = "環境變數 DATABASE_URL 未設定。對話紀錄功能需要 PostgreSQL資料庫。"
            logger.error(f"❌ 機器人 ' {self.bot_name}' - {error_msg}")
            raise ValueError(error_msg)
        logger.info(f"✅ 機器人 ' {self.bot_name}' 將使用 PostgreSQL (透過 DATABASE_URL)記錄對話。")
        return {
             "type": "postgresql"
             "connection_string": database_url
        } 
  
    def create_anonymous_user(self, request: Request) -> User:
        """创建匿名用户对象"""
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
        """获取用户唯一标识符，用于对话记录"""
        if user and user.id:
            return f"auth_user_{user.id}_{user.username}"
        else:
            return f"anonymous_session_{session_id}"

    def _get_document_content(self, doc) -> str:
        """统一获取文档内容的方法"""
        # 尝试常见的内容属性名称
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
        
        # 如果都没有，返回字符串表示
        return str(doc)

    # 🔧 新增：通过API搜索向量
    async def _search_vectors_via_api(self, query: str, k: int = 3):
        """通过API搜索向量"""
        if not HTTPX_AVAILABLE:
            logger.warning("httpx 不可用，无法使用API模式")
            return []
            
        try:
            timeout = httpx.Timeout(30.0, connect=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.vector_api_url}/search",
                    json={
                        "query": query,
                        "collection_name": self.collection_name,
                        "k": k
                    }
                )
                
                if response.status_code == 200:
                    api_results = response.json()
                    
                    # 转换为兼容原有代码的文档格式
                    documents = []
                    for result in api_results:
                        # 创建兼容的文档对象
                        doc = type('Document', (), {
                            'page_content': result.get('content', ''),
                            'metadata': result.get('metadata', {})
                        })()
                        documents.append(doc)
                    
                    logger.info(f"API搜索成功: {len(documents)} 个结果")
                    return documents
                    
                elif response.status_code == 503:
                    logger.warning("向量API服务不可用")
                    return []
                else:
                    logger.error(f"API搜索失败: {response.status_code} - {response.text}")
                    return []
                    
        except httpx.TimeoutException:
            logger.error("向量搜索API超时")
            return []
        except httpx.ConnectError:
            logger.error("无法连接到向量API服务")
            return []
        except Exception as e:
            logger.error(f"向量搜索API调用异常: {e}")
            return []

    # 🔧 保持原有的直接搜索方法
    def _search_vectors_direct(self, query: str, k: int = 3):
        """直接搜索向量"""
        if not self.vector_system:
            logger.warning("向量系统不可用")
            return []
        
        try:
            return self.vector_system.search(query, collection_name=self.collection_name, k=k)
        except Exception as e:
            logger.error(f"直接向量搜索失败: {e}")
            return []

    # 🔧 智能向量搜索方法
    async def _search_vectors_smart(self, query: str, k: int = 3):
        """智能选择搜索方式"""
        context_docs = []
        
        # 主要模式尝试
        if self.search_mode == "api":
            context_docs = await self._search_vectors_via_api(query, k)
            search_method = "API"
        elif self.search_mode == "direct":
            context_docs = self._search_vectors_direct(query, k)
            search_method = "直接"
        else:
            logger.warning("向量搜索功能不可用")
            return []
        
        # 如果主要模式失败，尝试备用模式
        if not context_docs:
            if self.search_mode == "api" and self.vector_system:
                logger.warning("API模式无结果，尝试直接模式...")
                context_docs = self._search_vectors_direct(query, k)
                if context_docs:
                    search_method = "直接(备用)"
                    logger.info(f"✅ 直接模式备用成功: {len(context_docs)} 个文档")
            elif self.search_mode == "direct" and HTTPX_AVAILABLE:
                logger.warning("直接模式无结果，尝试API模式...")
                context_docs = await self._search_vectors_via_api(query, k)
                if context_docs:
                    search_method = "API(备用)"
                    logger.info(f"✅ API模式备用成功: {len(context_docs)} 个文档")
        
        if context_docs:
            logger.info(f"🔍 机器人 '{self.bot_name}' {search_method}检索结果: {len(context_docs)} 个文档")
        else:
            logger.warning(f"🔍 机器人 '{self.bot_name}' 未找到相关文档")
        
        return context_docs

    # 保持原有的其他方法不变
    def get_chunk_info_safely(self, search_item, fallback_index: int) -> dict:
        """安全地从搜索结果中获取chunk信息"""
        try:
            chunk_info = {
                'chunk_id': f'chunk_{fallback_index}',
                'chunk_index': fallback_index,
                'content': '',
                'source': 'unknown',
                'filename': 'unknown'
            }
            
            # 方法1: 处理不同类型的搜索结果
            if hasattr(search_item, 'page_content'):
                # LangChain Document 对象
                chunk_info['content'] = search_item.page_content
                if hasattr(search_item, 'metadata') and search_item.metadata:
                    metadata = search_item.metadata
                    chunk_info['chunk_id'] = metadata.get('chunk_id', f'chunk_{fallback_index}')
                    chunk_info['chunk_index'] = metadata.get('chunk_index', fallback_index)
                    chunk_info['source'] = metadata.get('source', 'unknown')
                    chunk_info['filename'] = metadata.get('filename', metadata.get('original_filename', 'unknown'))
            
            elif isinstance(search_item, dict):
                # 字典格式的搜索结果
                chunk_info['content'] = search_item.get('content', search_item.get('page_content', ''))
                metadata = search_item.get('metadata', {})
                chunk_info['chunk_id'] = metadata.get('chunk_id', f'chunk_{fallback_index}')
                chunk_info['chunk_index'] = metadata.get('chunk_index', fallback_index)
                chunk_info['source'] = metadata.get('source', 'unknown')
                chunk_info['filename'] = metadata.get('filename', metadata.get('original_filename', 'unknown'))
            
            elif hasattr(search_item, 'content'):
                # 自定义搜索结果对象
                chunk_info['content'] = str(search_item.content)
                if hasattr(search_item, 'metadata'):
                    metadata = search_item.metadata
                    chunk_info['chunk_id'] = metadata.get('chunk_id', f'chunk_{fallback_index}')
                    chunk_info['chunk_index'] = metadata.get('chunk_index', fallback_index)
                    chunk_info['source'] = metadata.get('source', 'unknown')
                    chunk_info['filename'] = metadata.get('filename', 'unknown')
            
            else:
                # 备用：将整个对象转为字符串
                chunk_info['content'] = str(search_item)
            
            return chunk_info
            
        except Exception as e:
            logger.warning(f"获取chunk信息失败: {e}")
            return {
                'chunk_id': f'chunk_{fallback_index}',
                'chunk_index': fallback_index,
                'content': str(search_item) if search_item else '',
                'source': 'unknown',
                'filename': 'unknown'
            }

    def _find_real_chunk_index(self, content: str, fallback_index: int) -> int:
        """查找内容在向量库中的真实索引"""
        try:
            if self.vector_system:
                vectorstore = self.vector_system.get_or_create_vectorstore(self.collection_name)
                all_docs = vectorstore.get()
                
                if all_docs and all_docs.get('documents'):
                    # 查找完全匹配的内容
                    for idx, doc_content in enumerate(all_docs['documents']):
                        if doc_content.strip() == content.strip():
                            return idx
                    
                    # 如果没有完全匹配，查找部分匹配
                    for idx, doc_content in enumerate(all_docs['documents']):
                        if content[:100] in doc_content or doc_content[:100] in content:
                            return idx
            
            # 如果都找不到，返回备用索引
            return fallback_index
            
        except Exception as e:
            logger.debug(f"查找真实chunk索引失败: {e}")
            return fallback_index

    def _get_chunk_index_from_doc(self, doc, fallback_index: int) -> int:
        """從文檔對象中獲取 chunk 的實際索引 - 改進版"""
        try:
            logger.debug(f"🔍 嘗試獲取文檔的真實索引，fallback: {fallback_index}")
            
            # 方法1: 從 metadata 中獲取 - 改進版
            metadata = None
            if hasattr(doc, 'metadata') and doc.metadata:
                metadata = doc.metadata
            elif isinstance(doc, dict) and 'metadata' in doc:
                metadata = doc['metadata']
            
            if metadata:
                # 🆕 記錄元數據內容用於調試
                logger.debug(f"📋 文檔元數據: {list(metadata.keys())}")
                
                # 擴展搜索字段
                index_fields = [
                    'chunk_index', 'index', 'chunk_id', 'id', 
                    'chunk_num', 'chunk_number', 'doc_index',
                    'document_index', 'position', 'seq'
                ]
                
                for index_field in index_fields:
                    if index_field in metadata:
                        value = metadata[index_field]
                        logger.debug(f"📍 找到索引字段 {index_field}: {value} (type: {type(value)})")
                        
                        if isinstance(value, int) and value >= 0:
                            logger.info(f"✅ 從 metadata.{index_field} 獲取索引: {value}")
                            return value
                        elif isinstance(value, str):
                            # 嘗試多種字符串格式
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
            
            # 方法2: 從文檔ID中提取索引 - 保持不變
            doc_id = None
            if hasattr(doc, 'id') and doc.id:
                doc_id = str(doc.id)
            elif isinstance(doc, dict) and 'id' in doc:
                doc_id = str(doc['id'])
            
            if doc_id:
                logger.debug(f"📄 文檔ID: {doc_id}")
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
                        logger.info(f"✅ 從文檔ID解析索引: {idx} (pattern: {pattern})")
                        return idx
            
            # 方法3: 從向量數據庫查詢實際位置 - 改進版
            if self.vector_system:
                try:
                    logger.debug(f"🔍 嘗試從向量數據庫查詢實際位置...")
                    vectorstore = self.vector_system.get_or_create_vectorstore(self.collection_name)
                    all_docs_result = vectorstore.get()
                    
                    if all_docs_result and all_docs_result.get('documents'):
                        all_documents = all_docs_result['documents']
                        doc_content = self._get_document_content(doc)
                        
                        # 🆕 改進：嘗試多種匹配策略
                        
                        # 策略1: 完全匹配
                        for idx, stored_doc_content in enumerate(all_documents):
                            if stored_doc_content.strip() == doc_content.strip():
                                logger.info(f"✅ 通過完全匹配找到索引: {idx}")
                                return idx
                        
                        # 策略2: 前100字符匹配
                        doc_prefix = doc_content[:100].strip()
                        if len(doc_prefix) > 20:  # 確保有足夠的內容進行匹配
                            for idx, stored_doc_content in enumerate(all_documents):
                                stored_prefix = stored_doc_content[:100].strip()
                                if doc_prefix == stored_prefix:
                                    logger.info(f"✅ 通過前綴匹配找到索引: {idx}")
                                    return idx
                        
                        # 策略3: 關鍵詞匹配（如果內容較短）
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
                                    logger.info(f"✅ 通過詞匯匹配找到索引: {best_match_idx} (相似度: {best_match_score:.2f})")
                                    return best_match_idx
                    
                    logger.warning(f"⚠️ 向量數據庫查詢無匹配結果")
                            
                except Exception as query_error:
                    logger.warning(f"⚠️ 查詢向量數據庫索引失敗: {query_error}")
            
            # 方法4: 使用備用索引
            logger.warning(f"⚠️ 所有方法都失敗，使用備用索引: {fallback_index}")
            return fallback_index
            
        except Exception as e:
            logger.error(f"❌ 獲取 chunk 索引時發生異常: {e}")
            return fallback_index

    def setup_routes(self):
        """设置路由"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_modern_chat_ui(request: Request):
            """主聊天界面"""
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
                logger.error(f"❌ 载入聊天界面失败: {e}")
                return HTMLResponse(f"""
                <!DOCTYPE html>
                <html>
                <head><title>{self.bot_name}</title></head>
                <body>
                    <h1>聊天界面载入失败</h1>
                    <p>错误：{e}</p>
                    <p>请检查模板文件是否存在于 chatbot_templates/modern/chat.html</p>
                </body>
                </html>
                """, status_code=500)

        @self.app.post("/api/chat")
        async def chat(request: Request, user: User = Depends(OptionalAuth)):
            """聊天API端点 - 支持API和直接两种模式"""
            start_time = time.time()
            conversation_id = None
            chunk_references = []
            
            try:
                data = await request.json()
                query = data.get("message")
                session_id = data.get("session_id", "default_session")
                
                if not query:
                    raise HTTPException(status_code=400, detail="Message cannot be empty")
                
                # 处理用户识别
                if not user:
                    user = self.create_anonymous_user(request)
                
                user_identifier = self.get_user_identifier(user, session_id)
                
                # 更新统计
                self.total_conversations += 1
                if session_id not in self.session_counters:
                    self.session_counters[session_id] = 0
                self.session_counters[session_id] += 1
                self._current_query = query
                logger.info(f"📩 机器人 '{self.bot_name}' 收到查询 [{user_identifier}]: {query[:50]}...")

                # 🔧 修改：使用智能向量搜索
                context_docs = await self._search_vectors_smart(query, k=3)
                
                # 详细调试检索结果
                logger.info(f"🔍 当前查询: {query}")
                logger.info(f"📄 检索到 {len(context_docs) if context_docs else 0} 个文档")

                if context_docs:
                    for i, doc in enumerate(context_docs):
                        content_preview = self._get_document_content(doc)[:100]
                        metadata = getattr(doc, 'metadata', {})
                        contained_urls = metadata.get('contained_urls', '')
                        
                        logger.info(f"📄 文档 {i+1}:")
                        logger.info(f"  内容预览: {content_preview}")
                        logger.info(f"  包含连接: {contained_urls}")
                        logger.info(f"  来源文件: {metadata.get('filename', 'unknown')}")

                context = "\n".join([self._get_document_content(doc) for doc in context_docs]) if context_docs else ""
                
                # 准备 chunk 引用信息
                retrieved_docs_content = []
                
                if context_docs:
                    for i, doc in enumerate(context_docs):
                        doc_content = self._get_document_content(doc)
                        retrieved_docs_content.append(doc_content)
                        
                        chunk_index = self._get_chunk_index_from_doc(doc, i)
                        
                        # 获取元数据
                        metadata = {}
                        if hasattr(doc, 'metadata') and doc.metadata:
                            metadata = doc.metadata
                        elif isinstance(doc, dict) and 'metadata' in doc:
                            metadata = doc['metadata']
                        
                        chunk_ref = {
                            "index": chunk_index,
                            "content_preview": doc_content[:100] + "..." if len(doc_content) > 100 else doc_content,
                            "source": metadata.get('source', 'unknown'),
                            "filename": metadata.get('filename', metadata.get('original_filename', 'unknown'))
                        }
                        chunk_references.append(chunk_ref)

                logger.info(f"🔍 机器人 '{self.bot_name}' 检索结果: {len(context_docs)} 个文档, chunk_refs: {len(chunk_references)}")
                
                # 生成回应
                system_prompt = self.config.get("system_role", "你是一个乐于助人的 AI 助理。")
                response_text, recommended_questions = self._generate_response(
                    query, context, system_prompt, session_id
                )
                
                # 处理引用来源
                if self.config.get("cite_sources_enabled", False) and context_docs:
                    all_sources = self._extract_source_urls(context_docs)
                    # 只处理包含真实URL的来源
                    url_sources = [s for s in all_sources if s.get("url")]
                    
                    if url_sources:
                        # 过滤重复连接
                        url_sources = self._filter_duplicate_links(url_sources, session_id)
                        
                        if url_sources:  # 确保过滤后还有连接
                            response_text += self._format_source_links(url_sources)
                            logger.info(f"🔗 添加了 {len(url_sources)} 个参考连接")
                        else:
                            logger.info("🔄 所有连接都是重复的，跳过显示")
                    else:
                        logger.info("📝 未在文档内容中找到可引用的URL")

                processing_time_ms = int((time.time() - start_time) * 1000)
                
                # 记录对话
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
                    logger.info(f"✅ 对话记录成功：ID={conversation_id}, 机器人：{self.bot_name}, chunks={len(chunk_references)}")
                    
                except Exception as log_error:
                    logger.error(f"❌ 记录对话失败（机器人：{self.bot_name}）: {log_error}")

                logger.info(f"📤 机器人 '{self.bot_name}' API 回应调试:")
                logger.info(f"  - response_text 长度: {len(response_text)}")
                logger.info(f"  - recommended_questions: {recommended_questions}")
                logger.info(f"  - 找到文档数量: {len(context_docs) if context_docs else 0}")
                logger.info(f"  - chunk_references: {len(chunk_references)}")
                logger.info(f"  - 处理时间: {processing_time_ms}ms")

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
                    query = data.get("message", "未知查询") if 'data' in locals() else "未知查询"
                    
                    conversation_id = self.logger.log_conversation(
                        user_id=user_identifier,
                        user_query=query,
                        ai_response=f"HTTP错误: {error_message}",
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
                    logger.error(f"❌ HTTP错误记录：{conversation_id}, 机器人：{self.bot_name}")
                    
                except Exception as log_error:
                    logger.error(f"❌ HTTP错误对话记录失败（机器人：{self.bot_name}）: {log_error}")
                
                raise http_exc
                
            except Exception as e:
                error_message = str(e)
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                logger.error(f"❌ 机器人 '{self.bot_name}' 聊天 API 发生严重错误: {e}", exc_info=True)
                
                try:
                    user_identifier = self.get_user_identifier(user, session_id) if 'session_id' in locals() else "unknown"
                    query = data.get("message", "未知查询") if 'data' in locals() else "未知查询"
                    
                    conversation_id = self.logger.log_conversation(
                        user_id=user_identifier,
                        user_query=query,
                        ai_response=f"系统错误: {error_message}",
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
                    logger.error(f"❌ 系统错误记录：{conversation_id}, 机器人：{self.bot_name}")
                    
                except Exception as log_error:
                    logger.error(f"❌ 系统错误对话记录失败（机器人：{self.bot_name}）: {log_error}")
                
                return JSONResponse({
                    "response": f"抱歉，机器人 '{self.bot_name}' 发生内部错误，请稍后再试。",
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
            """获取机器人统计信息"""
            try:
                db_stats = self.logger.get_statistics() if hasattr(self.logger, 'get_statistics') else {}
                
                return JSONResponse({
                    "bot_name": self.bot_name,
                    "display_name": self.config.get("display_name", self.bot_name),
                    "collection_name": self.collection_name,
                    "conversation_db_path": self.conversation_db_path,
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
                logger.error(f"获取机器人统计失败: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/health")
        async def health_check():
            """健康检查端点"""
            return JSONResponse({
                "status": "healthy",
                "bot_name": self.bot_name,
                "display_name": self.config.get("display_name", self.bot_name),
                "search_mode": self.search_mode,
                "timestamp": time.time(),
                "conversation_count": self.total_conversations
            })

    def _generate_response(self, query: str, context: str, system_prompt: str, session_id: str) -> Tuple[str, List[str]]:
        """生成回应"""
        if not OPENAI_AVAILABLE:
            return "系统 AI 模块未载入。", []

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return "系统配置错误：缺少 OpenAI API Key。", []

        llm = ChatOpenAI(
            model=self.config.get("model", "gpt-4o-mini"), 
            temperature=self.config.get("temperature", 0.7), 
            api_key=openai_key
        )

        # 生成主要回答
        main_answer_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""参考资料：
{context}

用户问题：{query}""")
        ]
        main_response = llm.invoke(main_answer_messages)
        main_answer = main_response.content.strip()

        # 生成推荐问题
        recommended_questions = []
        should_recommend = self.config.get("dynamic_recommendations_enabled", False)
        recommend_count = self.config.get("dynamic_recommendations_count", 0)
        conversation_count = self.session_counters.get(session_id, 1)

        logger.info(f"🔍 机器人 '{self.bot_name}' 推荐问题调试:")
        logger.info(f"  - should_recommend: {should_recommend}")
        logger.info(f"  - recommend_count: {recommend_count}")
        logger.info(f"  - conversation_count: {conversation_count}")

        if should_recommend and (recommend_count == 0 or conversation_count <= recommend_count):
            logger.info(f"✅ 机器人 '{self.bot_name}' 开始生成推荐问题...")
            
            recommend_prompt = f"""**原始对话**
用户问：「{query}」
你的回答：「{main_answer}」

---
**指令**
根据以上对话，生成三个相关的延伸问题。

**格式**
- 每个问题一行
- 不要编号
- 不要包含任何其他文字"""
            
            try:
                recommend_messages = [HumanMessage(content=recommend_prompt)]
                recommend_response = llm.invoke(recommend_messages)
                questions_text = recommend_response.content.strip()
                
                logger.info(f"🤖 机器人 '{self.bot_name}' LLM 原始回应:")
                logger.info(f"'{questions_text}'")
                
                recommended_questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
                
                logger.info(f"📝 机器人 '{self.bot_name}' 解析后的推荐问题: {recommended_questions}")
                
            except Exception as e:
                logger.error(f"❌ 机器人 '{self.bot_name}' 生成推荐问题时发生错误: {e}", exc_info=True)
                recommended_questions = []
        else:
            logger.info(f"❌ 机器人 '{self.bot_name}' 推荐问题未启用或超出限制")

        return main_answer, recommended_questions
    
    def _extract_links_from_content(self, content: str) -> List[dict]:
        """从文件内容中提取连接和标题（稳健判定 URL/标题顺序）"""
        links = []
        
        # 正则表达式模式匹配各种连接格式
        patterns = [
            # Markdown格式: [标题](URL) -> (title, url)
            r'\[([^\]]+)\]\((https?://[^\s)]+)\)',
            # HTML格式: <a href="URL">标题</a> -> (url, title)
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>',
            # 纯文字格式: 标题: URL  -> (title, url)
            r'([^:\n]+):\s*(https?://[^\s]+)',
            # 纯文字格式: 标题 - URL  -> (title, url)
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

                # 清理标题
                title = title.strip().strip('"\'')
                if not title or len(title) < 3 or len(title) > 200:
                    continue

                # 排除無意義標題
                if re.match(r'^(点击这里|阅读更多|更多信息|连接|网址|click here|read more|more info|link|url)$', title, re.IGNORECASE):
                    continue
                if re.match(r'^\d+$', title) or re.match(r'^[^\w\u4e00-\u9fff]+$', title):
                    continue

                links.append({"title": title, "url": url})

        return links
    
    def _find_title_for_url_in_content(self, content: str, target_url: str) -> str:
        """在文件内容中查找特定URL对应的标题"""
        try:
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if target_url in line:
                    # 查看当前行和前后几行寻找标题
                    context_lines = []
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context_lines = lines[start:end]
                    
                    for context_line in context_lines:
                        # 跳过包含URL的行
                        if target_url in context_line:
                            continue
                        
                        # 查找标题模式
                        context_line = context_line.strip()
                        if context_line and len(context_line) > 5 and len(context_line) < 200:
                            # 移除常见的标记符号
                            cleaned = re.sub(r'^[#*\-•\d\.\s]+', '', context_line)
                            if cleaned and len(cleaned) > 5:
                                return cleaned
            
            return ""
            
        except Exception as e:
            logger.debug(f"在内容中查找标题失败: {e}")
            return ""

    def _generate_smart_title(self, metadata: dict, url: str) -> str:
        """智慧生成标题"""
        try:
            # 优先级1: 使用文件标题
            title = metadata.get('title', '')
            if title and title != 'unknown' and len(title) > 3:
                return title
            
            # 优先级2: 使用原始文件名称（去除副档名）
            filename = metadata.get('original_filename', metadata.get('filename', ''))
            if filename and filename != 'unknown':
                title = os.path.splitext(filename)[0]
                # 美化文件名称
                title = title.replace('-', ' ').replace('_', ' ')
                title = ' '.join(word.capitalize() for word in title.split() if word)
                if len(title) > 3:
                    return title
            
            # 优先级3: 从URL路径提取智慧标题
            from urllib.parse import urlparse, unquote
            parsed = urlparse(url)
            
            # 获取路径中的有用信息
            path_parts = [part for part in parsed.path.strip('/').split('/') if part]
            if path_parts:
                # 使用最后一个有意义的路径段
                last_part = path_parts[-1]
                # URL解码
                last_part = unquote(last_part)
                # 移除常见的副档名
                last_part = re.sub(r'\.(html|php|aspx|jsp|htm)$', '', last_part)
                # 替换分隔符并美化
                title = last_part.replace('-', ' ').replace('_', ' ')
                title = ' '.join(word.capitalize() for word in title.split() if word)
                if len(title) > 3:
                    return title
            
            # 优先级4: 使用查询参数中的信息
            if parsed.query:
                query_parts = parsed.query.split('&')
                for part in query_parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        if key in ['title', 'q', 'query', 'search'] and value:
                            decoded_value = unquote(value).replace('+', ' ')
                            if len(decoded_value) > 3:
                                return decoded_value
            
            # 优先级5: 使用域名（最后的备选）
            domain = parsed.netloc
            if domain:
                # 移除www前缀
                domain = re.sub(r'^www\.', '', domain)
                return f"来源：{domain}"
            
            return url
            
        except Exception as e:
            logger.warning(f"生成智慧标题失败: {e}")
            return self._extract_domain_from_url(url)

    def _extract_source_urls(self, docs: List) -> List[dict]:
        """从文件元数据中提取URL和完整标题信息"""
        sources = []
        seen_urls = set()

        for doc in docs:
            try:
                metadata = getattr(doc, 'metadata', {})
                
                # 方法1: 检查是否有预处理的标题-URL映射
                title_url_mapping = metadata.get('title_url_mapping', {})
                if isinstance(title_url_mapping, str):
                    try:
                        title_url_mapping = json.loads(title_url_mapping)
                    except json.JSONDecodeError:
                        title_url_mapping = {}
                
                # 如果有映射，直接使用
                if title_url_mapping:
                    for title, url in title_url_mapping.items():
                        if url not in seen_urls:
                            sources.append({
                                "title": title,
                                "url": url
                            })
                            seen_urls.add(url)
                    continue
                
                # 方法2: 从文件内容中提取标题和URL
                doc_content = self._get_document_content(doc)
                extracted_links = self._extract_links_from_content(doc_content)
                for link_info in extracted_links:
                    if link_info['url'] not in seen_urls:
                        sources.append(link_info)
                        seen_urls.add(link_info['url'])
                
                # 方法3: 从contained_urls获取URL，然后尝试匹配标题
                url_string = metadata.get('contained_urls', '')
                if url_string and not extracted_links:
                    urls_in_chunk = [url.strip() for url in url_string.split('|') if url.strip()]
                    
                    for url in urls_in_chunk:
                        if url not in seen_urls:
                            # 尝试从文件内容中找到对应的标题
                            title = self._find_title_for_url_in_content(doc_content, url)
                            if not title:
                                # 如果找不到，使用智慧标题提取
                                title = self._generate_smart_title(metadata, url)
                            
                            sources.append({
                                "title": title,
                                "url": url
                            })
                            seen_urls.add(url)

            except Exception as e:
                logger.warning(f"从元数据提取URL时出错: {e}")

        logger.info(f"从元数据中提取到 {len(sources)} 个连接")
        return sources

    def _extract_domain_from_url(self, url: str) -> str:
        """从URL中提取域名作为标题"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            return domain if domain else url
        except:
            return url

    def _filter_duplicate_links(self, sources: List[dict], session_id: str) -> List[dict]:
        """过滤重复连接"""
        if session_id not in self.session_shown_links:
            self.session_shown_links[session_id] = set()
        
        shown_links = self.session_shown_links[session_id]
        filtered_sources = []
        
        for source in sources:
            url = source.get('url', '')
            if url and url not in shown_links:
                filtered_sources.append(source)
                shown_links.add(url)
                logger.info(f"✅ 新连接: {source.get('title', 'unknown')}")
            else:
                logger.info(f"🔄 跳过重复连接: {source.get('title', 'unknown')}")
        
        return filtered_sources

    def _format_source_links(self, sources: List[dict]) -> str:
        """格式化參考連結 - 確保正確換行"""
        if not sources:
            return ""
        
        # 🔧 修正：確保機器人回答和推薦區塊之間有空行
        source_links = "\n\n💡 你可能想知道\n"  # 兩個\n確保空行，最後一個\n讓標題單獨一行
        
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
        
        # 清理标题
        title = title.strip()
        title = re.sub(r'\s+', ' ', title)  # 合并多个空格
        title = re.sub(r'^[^\w\u4e00-\u9fff]+', '', title)  # 移除开头的符号
        title = re.sub(r'[^\w\u4e00-\u9fff]+$', '', title)  # 移除结尾的符号
        
        # 验证标题长度和内容
        if len(title) < 3 or len(title) > 200:
            return ""
        
        # 排除无意义的标题
        meaningless_patterns = [
            r'^(|更多信息|連接|網址|click here|read more|more info|link|url)$',
            r'^\d+$',  # 纯数字
            r'^[^\w\u4e00-\u9fff]+$',  # 只有符号
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, title, re.IGNORECASE):
                return ""
        
        return title


def main():
    parser = argparse.ArgumentParser(description="启动一个独立的聊天机器人实例。")
    parser.add_argument("--bot-name", type=str, required=True, help="要启动的机器人名称")
    args = parser.parse_args()

    try:
        instance = ChatbotInstance(args.bot_name)
        port = instance.config.get("port")
        if not port:
            raise ValueError(f"设定文件 '{args.bot_name}.json' 中未指定端口")
        
        logger.info(f"🤖 机器人 '{instance.bot_name}' 正在 http://localhost:{port} 上启动")
        logger.info(f"📊 对话记录数据库已在实例中配置。")
        logger.info(f"📚 知识库集合：{instance.collection_name}")
        logger.info(f"🔍 搜索模式：{instance.search_mode}")
        if instance.search_mode == "api":
            logger.info(f"🔗 向量API地址：{instance.vector_api_url}")
        logger.info(f"🔧 调试模式已启用，将输出详细的对话记录和推荐问题生成日志")
        
        uvicorn.run(instance.app, host="0.0.0.0", port=int(port))

    except Exception as e:
        logger.error(f"❌ 启动机器人 '{args.bot_name}' 失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()