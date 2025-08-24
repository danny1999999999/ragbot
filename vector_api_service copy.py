#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vector_api_service.py - 向量数据库API服务 (修正版)
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

# 导入向量系统
try:
    from vector_builder_langchain_huge import OptimizedVectorSystem
    VECTOR_SYSTEM_AVAILABLE = True
    print("✅ 向量系统导入成功")
except ImportError as e:
    VECTOR_SYSTEM_AVAILABLE = False
    print(f"❌ 向量系统导入失败: {e}")
    print("   请检查 vector_builder_langchain.py 是否存在")

# 可选的性能监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil 未安装，性能监控功能受限")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchRequest(BaseModel):
    query: str
    collection_name: str
    k: int = 3

class VectorAPIService:
    def __init__(self):
        self.app = FastAPI(
            title="向量数据库API服务",
            description="集中管理向量数据库，避免并发冲突",
            version="1.0.0",
        )
        
        # CORS支持
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 初始化向量系统
        self.vector_system = None
        if VECTOR_SYSTEM_AVAILABLE:
            try:
                print("🔧 初始化向量系统...")
                
                # 检查和创建必要目录
                data_dir = Path("./data")
                chroma_dir = Path("./chroma_langchain_db")
                
                data_dir.mkdir(exist_ok=True)
                chroma_dir.mkdir(exist_ok=True)
                
                self.vector_system = OptimizedVectorSystem()
                logger.info("✅ 向量系统初始化成功")
                
            except Exception as e:
                logger.error(f"❌ 向量系统初始化失败: {e}")
                print(f"错误详情: {e}")
                self.vector_system = None
        
        # 统计信息
        self.start_time = time.time()
        self.request_count = 0
        self.upload_count = 0  # 避免 /upload 時 self.upload_count += 1 觸發 AttributeError
        self.init_status = "healthy" if self.vector_system else "unavailable"  # 提供 /upload 503 訊息使用
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.middleware("http")
        async def count_requests(request, call_next):
            self.request_count += 1
            response = await call_next(request)
            return response
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            uptime = time.time() - self.start_time
            
            # 检查向量系统状态
            vector_status = "unavailable"
            if self.vector_system:
                try:
                    # 简单测试 - 获取可用集合
                    collections = self.vector_system.get_available_collections()
                    vector_status = "healthy"
                except Exception as e:
                    vector_status = f"error: {str(e)}"
            
            health_data = {
                "status": "healthy" if vector_status == "healthy" else "degraded",
                "service": "vector_api",
                "uptime_seconds": int(uptime),
                "uptime_human": f"{int(uptime//3600)}h {int((uptime%3600)//60)}m",
                "request_count": self.request_count,
                "vector_system": vector_status,
                "timestamp": datetime.now().isoformat()
            }
            
            # 添加内存信息（如果可用）
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    health_data["memory_usage_mb"] = round(process.memory_info().rss / 1024 / 1024, 2)
                except:
                    pass
            
            return health_data
        
        @self.app.post("/search")
        async def search_vectors(request: SearchRequest):
            """搜索向量 - 只读操作"""
            if not self.vector_system:
                raise HTTPException(status_code=503, detail="向量系统不可用")
            
            try:
                logger.info(f"🔍 搜索请求: {request.query[:50]}... (集合: {request.collection_name})")
                
                # 调用向量系统搜索
                search_results = self.vector_system.search(
                    query=request.query,
                    collection_name=request.collection_name,
                    k=request.k
                )
                
                # 转换为统一的API响应格式
                api_results = []
                for i, result in enumerate(search_results):
                    try:
                        # 处理不同类型的搜索结果
                        if hasattr(result, 'content'):
                            content = result.content
                        elif hasattr(result, 'page_content'):
                            content = result.page_content
                        else:
                            content = str(result)
                        
                        score = getattr(result, 'score', 0.0)
                        metadata = getattr(result, 'metadata', {})
                        
                        api_results.append({
                            "content": content,
                            "score": float(score),
                            "metadata": dict(metadata) if metadata else {}
                        })
                        
                    except Exception as item_error:
                        logger.warning(f"处理搜索结果项 {i} 失败: {item_error}")
                        continue
                
                logger.info(f"✅ 搜索完成: 返回 {len(api_results)} 个结果")
                return api_results
                
            except Exception as e:
                logger.error(f"❌ 搜索失败: {e}")
                raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")
        
        @self.app.post("/upload")
        async def upload_file(
            collection_name: str = Form(...),
            file: UploadFile = File(...)
        ):
            """上传文件到向量数据库"""
            if not self.vector_system:
                raise HTTPException(
                    status_code=503, 
                    detail=f"向量系统不可用: {self.init_status}"
                )
            
            try:
                self.upload_count += 1
                start_time = time.time()
                
                logger.info(f"📤 上传请求: {file.filename} -> {collection_name}")
                
                # 基本验证
                if not file.filename or not file.filename.strip():
                    raise HTTPException(status_code=400, detail="文件名为空")
                
                # 讀取內容後以實際長度驗檢
                file_content = await file.read()
                if not file_content:
                    raise HTTPException(status_code=400, detail="文件内容为空")
                
                max_mb = int(os.getenv("VECTOR_API_MAX_UPLOAD_MB", "100"))
                if len(file_content) > max_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"文件过大（超过 {max_mb} MB）")
                
                # 🔧 優先走向量系統的一站式上傳
                if hasattr(self.vector_system, 'upload_single_file'):
                    result = self.vector_system.upload_single_file(
                        file_content=file_content,
                        filename=file.filename,
                        collection_name=collection_name
                    )
                else:
                    # 备用方案：手动处理上传（保持既有流程）
                    result = await self._manual_upload_process(
                        file_content, file.filename, collection_name
                    )
                
                process_time = time.time() - start_time
                logger.info(f"✅ 上传完成: {file.filename} ({process_time:.2f}s)")
                
                return JSONResponse(result)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ 上传失败: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")
        
        @self.app.get("/collections")
        async def get_collections():
            """获取所有集合"""
            if not self.vector_system:
                raise HTTPException(status_code=503, detail="向量系统不可用")
            
            try:
                collections = self.vector_system.get_available_collections()
                return JSONResponse(collections)
            except Exception as e:
                logger.error(f"获取集合失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/collections/{collection_name}/files")
        async def get_collection_files(collection_name: str):
            """获取集合中的文件 - 修正版"""
            if not self.vector_system:
                raise HTTPException(status_code=503, detail="向量系统不可用")
            
            try:
                if hasattr(self.vector_system, 'get_collection_documents'):
                    docs_result = self.vector_system.get_collection_documents(
                        collection_name, page=1, limit=1000
                    )
                    # 修正：在回傳前清理路徑
                    if docs_result and isinstance(docs_result.get('documents'), list):
                        cleaned_docs = []
                        for doc in docs_result['documents']:
                            if isinstance(doc, dict) and 'filename' in doc and isinstance(doc['filename'], str):
                                new_doc = doc.copy()
                                new_doc['filename'] = os.path.basename(doc['filename'])
                                cleaned_docs.append(new_doc)
                            else:
                                cleaned_docs.append(doc) # 保留無法處理的項目
                        docs_result['documents'] = cleaned_docs
                    
                    return JSONResponse(docs_result)
                else:
                    # 备用方案：返回空列表
                    return JSONResponse({
                        "success": True,
                        "documents": [],
                        "total": 0,
                        "message": "此功能在当前版本中不可用"
                    })
            except Exception as e:
                logger.error(f"获取文件列表失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/collections/{collection_name}/files/{filename}")
        async def delete_file(collection_name: str, filename: str):
            """删除文件"""
            if not self.vector_system:
                raise HTTPException(status_code=503, detail="向量系统不可用")
            
            try:
                if hasattr(self.vector_system, 'delete_document'):
                    result = self.vector_system.delete_document(collection_name, filename)
                    return JSONResponse(result)
                else:
                    return JSONResponse({
                        "success": False,
                        "message": "删除功能在当前版本中不可用"
                    })
            except Exception as e:
                logger.error(f"删除文件失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/sync")
        async def sync_all():
            """同步所有集合"""
            if not self.vector_system:
                raise HTTPException(status_code=503, detail="向量系统不可用")
            
            try:
                if hasattr(self.vector_system, 'sync_collections'):
                    changes = self.vector_system.sync_collections()
                    return JSONResponse({
                        "success": True,
                        "changes": changes,
                        "message": f"同步完成，处理了 {changes} 个变更"
                    })
                else:
                    return JSONResponse({
                        "success": False,
                        "message": "同步功能在当前版本中不可用"
                    })
            except Exception as e:
                logger.error(f"同步失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats")
        async def get_stats():
            """获取统计信息"""
            try:
                stats_data = {
                    "service_uptime": time.time() - self.start_time,
                    "requests_handled": self.request_count,
                    "vector_system_available": self.vector_system is not None
                }
                
                if self.vector_system:
                    try:
                        collections = self.vector_system.get_available_collections()
                        stats_data["collections"] = collections
                        stats_data["total_collections"] = len(collections)
                        
                        if hasattr(self.vector_system, 'get_stats'):
                            vector_stats = self.vector_system.get_stats()
                            stats_data["document_stats"] = vector_stats
                    except Exception as e:
                        stats_data["vector_system_error"] = str(e)
                
                return JSONResponse(stats_data)
                
            except Exception as e:
                logger.error(f"获取统计失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

def main():
    """主函数"""
    
    # 端口配置
    port = int(os.getenv("VECTOR_API_PORT", 9002))
    
    print("=" * 60)
    print("🔍 向量数据库API服务 v1.0")
    print("=" * 60)
    print(f"📍 服务端口: {port}")
    print(f"📖 API文档: http://localhost:{port}/docs")
    print(f"🔍 健康检查: http://localhost:{port}/health")
    print(f"🤖 向量系统: {'✅ 可用' if VECTOR_SYSTEM_AVAILABLE else '❌ 不可用'}")
    
    if not VECTOR_SYSTEM_AVAILABLE:
        print("⚠️ 警告: 向量系统不可用，服务将以受限模式运行")
    
    print("=" * 60)
    
    try:
        service = VectorAPIService()
        uvicorn.run(service.app, host="0.0.0.0", port=port, reload=False)
    except Exception as e:
        print(f"❌ 服务启动失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())