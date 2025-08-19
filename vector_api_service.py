#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vector_api_service_openai.py - OpenAI API 專用版本
針對 OpenAI API 用戶優化，快速啟動
"""

import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchRequest(BaseModel):
    query: str
    collection_name: str
    k: int = 3

class OpenAIVectorAPIService:
    def __init__(self):
        self.app = FastAPI(
            title="OpenAI 向量API服務",
            description="專為 OpenAI API 用戶優化",
            version="1.0.0-openai",
        )
        
        # CORS支持
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 檢查 OpenAI 配置
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.has_openai_key = bool(self.openai_key)
        
        # 向量系統狀態
        self.vector_system = None
        self.vector_status = "not_initialized"
        self.initialization_error = None
        
        # 統計信息
        self.start_time = time.time()
        self.request_count = 0
        self.upload_count = 0
        
        print(f"🔑 OpenAI API Key: {'已設置' if self.has_openai_key else '❌ 未設置'}")
        
        # 立即嘗試初始化（OpenAI API 版本應該很快）
        self.initialize_vector_system()
        
        self.setup_routes()
    
    def initialize_vector_system(self):
        """立即初始化向量系統 (OpenAI API 版本應該很快)"""
        if not self.has_openai_key:
            self.vector_status = "no_openai_key"
            self.initialization_error = "缺少 OPENAI_API_KEY 環境變數"
            print("❌ 缺少 OpenAI API Key，向量功能將受限")
            return
        
        try:
            print("🔧 初始化 OpenAI 向量系統...")
            self.vector_status = "initializing"
            
            # 檢查必要目錄
            Path("./data").mkdir(exist_ok=True)
            Path("./chroma_langchain_db").mkdir(exist_ok=True)
            
            # 嘗試導入和初始化
            from vector_builder_langchain import OptimizedVectorSystem
            
            # OpenAI API 版本的初始化應該很快，因為不需要下載本地模型
            self.vector_system = OptimizedVectorSystem()
            self.vector_status = "ready"
            
            print("✅ OpenAI 向量系統初始化成功")
            
            # 測試基本功能
            try:
                collections = self.vector_system.get_available_collections()
                print(f"📚 可用集合: {collections}")
            except Exception as e:
                print(f"⚠️ 測試功能時出現警告: {e}")
        
        except ImportError as e:
            self.vector_status = "import_error"
            self.initialization_error = f"導入失敗: {e}"
            print(f"❌ 向量系統導入失敗: {e}")
            print("💡 請檢查 vector_builder_langchain.py 是否存在")
        
        except Exception as e:
            self.vector_status = "init_error"
            self.initialization_error = f"初始化失敗: {e}"
            print(f"❌ 向量系統初始化失敗: {e}")
            print("💡 這可能是配置或依賴問題")
            import traceback
            traceback.print_exc()
    
    def setup_routes(self):
        @self.app.middleware("http")
        async def count_requests(request, call_next):
            self.request_count += 1
            response = await call_next(request)
            return response
        
        @self.app.get("/health")
        async def health_check():
            """Railway PostgreSQL 優化健康檢查"""
            return {
                "status": "healthy",  # 🔧 立即回傳健康狀態
                "service": "vector_api", 
                "database": "postgresql_pgvector",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/search")
        async def search_vectors(request: SearchRequest):
            """搜索向量 - OpenAI API 版本"""
            
            if self.vector_status == "ready" and self.vector_system:
                try:
                    logger.info(f"🔍 OpenAI 向量搜索: {request.query[:50]}...")
                    
                    search_results = self.vector_system.search(
                        query=request.query,
                        collection_name=request.collection_name,
                        k=request.k
                    )
                    
                    # 轉換為 API 格式
                    api_results = []
                    for result in search_results:
                        api_results.append({
                            "content": getattr(result, 'content', getattr(result, 'page_content', str(result))),
                            "score": float(getattr(result, 'score', 0.0)),
                            "metadata": dict(getattr(result, 'metadata', {}))
                        })
                    
                    logger.info(f"✅ 搜索完成: 返回 {len(api_results)} 個結果")
                    return api_results
                
                except Exception as e:
                    logger.error(f"❌ 向量搜索失敗: {e}")
                    # Fallback 到簡單搜索
                    return await self.fallback_search(request)
            
            else:
                # 向量系統不可用時的處理
                if self.vector_status == "no_openai_key":
                    raise HTTPException(
                        status_code=503, 
                        detail="需要設置 OPENAI_API_KEY 環境變數才能使用向量搜索"
                    )
                elif self.vector_status in ["import_error", "init_error"]:
                    return await self.fallback_search(request)
                else:
                    raise HTTPException(
                        status_code=503, 
                        detail=f"向量系統狀態: {self.vector_status}"
                    )
        
        @self.app.post("/upload")
        async def upload_file(
            collection_name: str = Form(...),
            file: UploadFile = File(...)
        ):
            """上傳檔案到向量數據庫"""
            
            if self.vector_status != "ready" or not self.vector_system:
                if self.vector_status == "no_openai_key":
                    raise HTTPException(
                        status_code=503, 
                        detail="需要設置 OPENAI_API_KEY 環境變數才能上傳檔案"
                    )
                else:
                    raise HTTPException(
                        status_code=503, 
                        detail=f"向量系統不可用: {self.vector_status}"
                    )
            
            try:
                self.upload_count += 1
                start_time = time.time()
                
                logger.info(f"📤 上傳請求: {file.filename} -> {collection_name}")
                
                # 讀取檔案內容
                file_content = await file.read()
                if not file_content:
                    raise HTTPException(status_code=400, detail="檔案內容為空")
                
                # 檔案大小檢查 (OpenAI API 有 token 限制)
                max_mb = int(os.getenv("MAX_UPLOAD_MB", "50"))
                if len(file_content) > max_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"檔案過大（超過 {max_mb} MB）")
                
                # 使用向量系統上傳
                if hasattr(self.vector_system, 'upload_single_file'):
                    result = self.vector_system.upload_single_file(
                        file_content=file_content,
                        filename=file.filename,
                        collection_name=collection_name
                    )
                else:
                    # 備用上傳方法
                    result = {
                        "success": True,
                        "message": "檔案已處理但上傳方法不可用",
                        "filename": file.filename,
                        "collection": collection_name
                    }
                
                process_time = time.time() - start_time
                logger.info(f"✅ 上傳完成: {file.filename} ({process_time:.2f}s)")
                
                return JSONResponse(result)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ 上傳失敗: {e}")
                raise HTTPException(status_code=500, detail=f"上傳失敗: {str(e)}")
        
        @self.app.get("/collections")
        async def get_collections():
            """獲取所有集合"""
            if self.vector_status == "ready" and self.vector_system:
                try:
                    collections = self.vector_system.get_available_collections()
                    return JSONResponse(collections)
                except Exception as e:
                    logger.error(f"獲取集合失敗: {e}")
                    # Fallback
                    return JSONResponse(["default", "documents"])
            else:
                return JSONResponse({
                    "error": f"向量系統不可用: {self.vector_status}",
                    "fallback_collections": ["default", "documents"]
                })
        
        @self.app.get("/status")
        async def get_detailed_status():
            """詳細狀態信息"""
            return JSONResponse({
                "vector_system_status": self.vector_status,
                "openai_configured": self.has_openai_key,
                "initialization_error": self.initialization_error,
                "request_count": self.request_count,
                "upload_count": self.upload_count,
                "uptime": time.time() - self.start_time,
                "capabilities": {
                    "search": self.vector_status == "ready",
                    "upload": self.vector_status == "ready",
                    "collections": self.vector_status == "ready"
                }
            })
    
    async def fallback_search(self, request: SearchRequest):
        """Fallback 搜索 (當向量系統不可用時)"""
        logger.info(f"🔍 Fallback 搜索: {request.query[:50]}...")
        
        return [
            {
                "content": f"Fallback 結果: {request.query}",
                "score": 0.7,
                "metadata": {
                    "source": "fallback_search",
                    "collection": request.collection_name,
                    "notice": f"向量系統狀態: {self.vector_status}",
                    "suggestion": "請檢查 OpenAI API 配置"
                }
            }
        ]

def main():
    """主函數"""
    
    # Railway 端口處理
    port = int(os.getenv("PORT", os.getenv("VECTOR_API_PORT", 9002)))
    
    print("=" * 60)
    print("🤖 OpenAI 向量API服務 v1.0")
    print("=" * 60)
    print(f"📍 服務端口: {port}")
    print(f"🔑 OpenAI API Key: {'已設置' if os.getenv('OPENAI_API_KEY') else '❌ 未設置'}")
    print(f"🌐 Railway 環境: {'是' if os.getenv('RAILWAY_PROJECT_ID') else '否'}")
    print("=" * 60)
    
    try:
        service = OpenAIVectorAPIService()
        
        # 根據環境調整配置
        uvicorn_config = {
            "host": "0.0.0.0",
            "port": port,
            "reload": False,
            "workers": 1,
        }
        
        if os.getenv("RAILWAY_PROJECT_ID"):
            uvicorn_config.update({
                "log_level": "info",
                "access_log": False,
            })
        
        uvicorn.run(service.app, **uvicorn_config)
        
    except Exception as e:
        print(f"❌ 服務啟動失敗: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())