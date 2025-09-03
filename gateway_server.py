import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import psycopg
import uvicorn
from fastapi import FastAPI, Request, Response, Header, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import urllib.parse
import tempfile
from bot_config_manager import DatabaseBotManager, initialize_database_bot_configs

# Load environment variables at the very top
load_dotenv()

# --- Project-level Imports ---
from config import app_config
from auth_middleware import AdminAuth, User, auth_response, JWTManager
from user_manager import user_manager, User as ModelUser # Rename to avoid conflict
from bot_service_manager import bot_manager, global_bot_instances
from conversation_logger_simple import create_logger_instance
from vector_builder_langchain import OptimizedVectorSystem

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("gateway")

# --- Global Variables & Paths ---
ROOT_DIR = Path(__file__).resolve().parent
#BOT_CONFIGS_DIR = ROOT_DIR / "bot_configs"
#BOT_CONFIGS_DIR.mkdir(exist_ok=True)
db_bot_manager = None

SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.pdf', '.csv', '.json', '.py', '.js', 
    '.docx', '.doc', '.rst', '.org', '.epub'
}

# --- FastAPI App Initialization ---
app = FastAPI(title="Unified API Gateway", version="3.0")
templates = Jinja2Templates(directory=str(ROOT_DIR))


try:
    db_bot_manager = initialize_database_bot_configs()
    logger.info("資料庫機器人設定系統初始化成功")
except Exception as e:
    logger.error(f"資料庫機器人設定系統初始化失敗: {e}")
    raise

# --- Service Initialization ---
vector_system = OptimizedVectorSystem()
conversation_loggers: Dict[str, object] = {}

def get_conversation_logger(bot_name: str):
    if bot_name not in conversation_loggers:
        db_config = {
            "type": "postgresql",
            "connection_string": os.getenv("DATABASE_URL")
        }
        conversation_loggers[bot_name] = create_logger_instance(db_config)
    return conversation_loggers[bot_name]

# --- HTML Page Routes ---
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("manager_login.html", {"request": request})

@app.get("/manager", response_class=HTMLResponse)
async def manager_page(request: Request, current_user: User = Depends(AdminAuth)):
    return templates.TemplateResponse("manager_ui.html", {"request": request, "user": current_user})

@app.get("/admin/users", response_class=HTMLResponse)
async def user_admin_page(request: Request, current_user: User = Depends(AdminAuth)):
    return templates.TemplateResponse("user_admin.html", {"request": request, "user": current_user})

# --- User Management API ---
@app.get("/api/users")
async def get_users(page: int = 1, limit: int = 15, search: str = "", current_user: User = Depends(AdminAuth)):
    if search:
        users = user_manager.search_users(search_term=search, limit=limit)
        total = len(users) # Simplified total for search
    else:
        users = user_manager.get_users(limit=limit, offset=(page - 1) * limit)
        total = user_manager.get_total_users_count()
    
    return {
        "success": True,
        "users": [u.to_dict() for u in users],
        "total": total,
        "page": page,
        "limit": limit
    }

@app.post("/api/users")
async def create_user(request: Request, current_user: User = Depends(AdminAuth)):
    data = await request.json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role", "user")

    if not all([username, email, password]):
        raise HTTPException(status_code=400, detail="Username, email, and password are required.")

    if user_manager.get_user_by_username(username) or user_manager.get_user_by_email(email):
        raise HTTPException(status_code=400, detail="User or email already exists.")

    hashed_password = user_manager.hash_password(password)
    new_user = ModelUser(
        username=username,
        email=email,
        password_hash=hashed_password,
        role=role
    )
    user_id = user_manager.create_user(new_user)

    if user_id:
        return {"success": True, "message": "User created successfully", "user_id": user_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to create user.")

@app.put("/api/users/{user_id}")
async def update_user(user_id: int, request: Request, current_user: User = Depends(AdminAuth)):
    data = await request.json()
    # Fields that can be updated
    allowed_updates = ["role", "is_active"]
    updates = {key: data[key] for key in allowed_updates if key in data}

    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update.")

    success = user_manager.update_user(user_id, updates)
    if success:
        return {"success": True, "message": "User updated successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to update user.")

@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int, current_user: User = Depends(AdminAuth)):
    # This is a soft delete
    success = user_manager.delete_user(user_id)
    if success:
        return {"success": True, "message": "User deactivated successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to deactivate user.")

# --- Authentication & Management API ---
@app.post("/api/login")
async def handle_login(request: Request):
    data = await request.json()
    username = data.get("username", "").strip()
    password = data.get("password", "")
    success, token_or_msg, user = user_manager.authenticate(username, password)
    if success and user.role in ["admin", "super_admin"]:
        jwt_token = JWTManager.create_access_token(user)
        return auth_response.create_login_response(user, jwt_token)
    elif success:
        return JSONResponse({"success": False, "message": "需要管理員權限"}, status_code=403)
    else:
        return JSONResponse({"success": False, "message": token_or_msg}, status_code=401)

@app.post("/api/logout")
async def handle_logout():
    return auth_response.create_logout_response()

@app.get("/api/bots")
async def get_all_bots(current_user: User = Depends(AdminAuth)):
    return JSONResponse(db_bot_manager.get_all_bots())

@app.post("/api/bots/{bot_name}/start")
async def start_bot(bot_name: str, current_user: User = Depends(AdminAuth)):
    return JSONResponse(bot_manager.start_bot(bot_name, app))

@app.post("/api/bots/{bot_name}/stop")
async def stop_bot(bot_name: str, current_user: User = Depends(AdminAuth)):
    return JSONResponse(bot_manager.stop_bot(bot_name, app))

@app.get("/api/bots/{bot_name}/config")
async def get_bot_config(bot_name: str, current_user: User = Depends(AdminAuth)):
    config = db_bot_manager.get_bot_config(bot_name)
    if config:
        return JSONResponse({"success": True, "config": config})
    return JSONResponse({"success": False, "message": "Config not found"}, status_code=404)

@app.post("/api/bots/create")
async def create_bot(request: Request, current_user: User = Depends(AdminAuth)):
    data = await request.json()
    bot_name = data.get("bot_name")
    display_name = data.get("display_name")
    port = data.get("port")
    system_role = data.get("system_role")

    if not bot_name or not port:
        raise HTTPException(status_code=400, detail="Bot name and port are required.")

    config_data = {
        "bot_name": bot_name,
        "display_name": display_name,
        "port": port,
        "system_role": system_role,
        "created_by": current_user.username
    }

    result = db_bot_manager.create_bot(config_data)
    if result["success"]:
        return JSONResponse(result)
    else:
        raise HTTPException(status_code=400, detail=result["message"])


@app.post("/api/bots/{bot_name}/config")
async def update_bot_config(bot_name: str, request: Request, current_user: User = Depends(AdminAuth)):
    update_data = await request.json()
    result = db_bot_manager.update_bot_config(bot_name, update_data)
    
    if result["success"]:
        return JSONResponse(result)
    else:
        raise HTTPException(status_code=500, detail=result["message"])


@app.post("/api/bots/{bot_name}/knowledge/upload-batch")
async def upload_knowledge_files_batch(
    bot_name: str, 
    files: List[UploadFile] = File(...),  # 支援多檔案
    current_user: User = Depends(AdminAuth)
):
    """批次上傳多個知識庫檔案"""
    try:
        collection_name = f"collection_{bot_name}"
        results = []
        
        # 建立臨時目錄存放檔案
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            file_paths = []
            
            # 儲存所有上傳的檔案
            for file in files:
                if not file.filename:
                    continue
                    
                file_path = temp_dir_path / file.filename
                
                # 檢查檔案格式
                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "message": f"不支援的檔案格式: {file_path.suffix}"
                    })
                    continue
                
                # 寫入檔案
                try:
                    content = await file.read()
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    file_paths.append(file_path)
                    
                except Exception as e:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "message": f"檔案寫入失敗: {str(e)}"
                    })
                    continue
            
            if not file_paths:
                return JSONResponse({
                    "success": False,
                    "message": "沒有有效的檔案可以處理",
                    "results": results
                }, status_code=400)
            
            # 使用向量系統批次處理
            try:
                # 載入所有文檔
                all_documents = []
                for file_path in file_paths:
                    try:
                        documents = vector_system.load_document(file_path)
                        if documents:
                            all_documents.extend(documents)
                            results.append({
                                "filename": file_path.name,
                                "success": True,
                                "message": f"成功載入 {len(documents)} 個分塊",
                                "chunks": len(documents)
                            })
                        else:
                            results.append({
                                "filename": file_path.name,
                                "success": False,
                                "message": "檔案載入後無有效內容"
                            })
                    except Exception as e:
                        results.append({
                            "filename": file_path.name,
                            "success": False,
                            "message": f"載入失敗: {str(e)}"
                        })
                
                if not all_documents:
                    return JSONResponse({
                        "success": False,
                        "message": "所有檔案載入失敗",
                        "results": results
                    }, status_code=400)
                
                # 批次向量化
                vectorstore = vector_system.get_or_create_vectorstore(collection_name)
                
                # 使用批次處理器
                batches = vector_system.batch_processor.create_smart_batches(all_documents)
                success_count = vector_system._process_batches(vectorstore, batches)
                
                # 計算總數據
                total_files = len([r for r in results if r["success"]])
                total_chunks = sum(r.get("chunks", 0) for r in results if r["success"])
                
                return JSONResponse({
                    "success": success_count > 0,
                    "message": f"批次上傳完成: {total_files} 個檔案, {total_chunks} 個分塊, {success_count} 個成功向量化",
                    "results": results,
                    "stats": {
                        "total_files": len(files),
                        "successful_files": total_files,
                        "total_chunks": total_chunks,
                        "vectorized_chunks": success_count
                    }
                })
                
            except Exception as e:
                logger.error(f"批次向量化失敗: {e}")
                return JSONResponse({
                    "success": False,
                    "message": f"批次處理失敗: {str(e)}",
                    "results": results
                }, status_code=500)
                
    except Exception as e:
        logger.error(f"批次上傳失敗: {e}")
        return JSONResponse({
            "success": False,
            "message": f"批次上傳失敗: {str(e)}"
        }, status_code=500)



@app.delete("/api/bots/{bot_name}")
async def delete_bot(bot_name: str, current_user: User = Depends(AdminAuth)):
    """刪除指定的機器人"""
    try:
        # 如果機器人正在運行，先停止它
        if bot_name in global_bot_instances:
            logger.info(f"正在停止運行中的機器人: {bot_name}")
            bot_manager.stop_bot(bot_name, app)

        # 從資料庫刪除配置
        result = db_bot_manager.delete_bot(bot_name)
        
        if result["success"]:
            logger.info(f"機器人 {bot_name} 已成功刪除 (用戶: {current_user.username})")
            return JSONResponse(result)
        else:
            logger.warning(f"刪除機器人失敗: {result['message']}")
            raise HTTPException(status_code=404, detail=result["message"])
            
    except Exception as e:
        logger.error(f"刪除機器人時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"刪除失敗: {str(e)}")



@app.delete("/api/bots/{bot_name}/conversations")
async def delete_conversations_batch(bot_name: str, request: Request, current_user: User = Depends(AdminAuth)):
    data = await request.json()
    conversation_ids = data.get("conversation_ids")
    if not conversation_ids:
        raise HTTPException(status_code=400, detail="No conversation IDs provided.")

    conv_logger = get_conversation_logger(bot_name)
    deleted_count = conv_logger.delete_conversations_batch(conversation_ids)
    
    return {"success": True, "message": f"Deleted {deleted_count} conversations."}


@app.delete("/api/bots/{bot_name}/conversations/{conversation_id}")
async def delete_single_conversation(bot_name: str, conversation_id: int, current_user: User = Depends(AdminAuth)):
    conv_logger = get_conversation_logger(bot_name)
    success = conv_logger.delete_conversation(conversation_id)
    if success:
        return {"success": True, "message": "Conversation deleted."}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found.")


@app.get("/api/bots/{bot_name}/conversations/chunk/{chunk_id}")
async def get_chunk_details(bot_name: str, chunk_id: str, current_user: User = Depends(AdminAuth)):
    # This functionality depends on the vector system, not the conversation logger.
    # Assuming the vector system can fetch a chunk by its ID.
    try:
        # The collection name is needed to get the chunk
        collection_name = f"collection_{bot_name}"
        chunk = vector_system.get_chunk_by_id(collection_name, chunk_id) # This method needs to be implemented
        if chunk:
            return {"success": True, "chunk": chunk}
        else:
            raise HTTPException(status_code=404, detail="Chunk not found.")
    except Exception as e:
        logger.error(f"Failed to get chunk details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bots/{bot_name}/knowledge/upload")
async def upload_knowledge_file(bot_name: str, file: UploadFile = File(...), current_user: User = Depends(AdminAuth)):
    try:
        collection_name = f"collection_{bot_name}"
        file_content = await file.read()
        result = vector_system.upload_single_file(
            file_content=file_content, filename=file.filename, collection_name=collection_name
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)
    
@app.get("/api/bots/{bot_name}/knowledge/files/{filename}/details")
async def get_file_details(bot_name: str, filename: str, current_user: User = Depends(AdminAuth)):
    """獲取檔案詳細資訊"""
    try:
        collection_name = f"collection_{bot_name}"
        chunks = vector_system.get_document_chunks(collection_name, filename)
        
        if not chunks:
            return JSONResponse({"success": False, "message": "檔案不存在"}, status_code=404)
        
        # 計算統計資訊
        total_tokens = sum(chunk.get('token_count', 0) for chunk in chunks)
        total_chars = sum(len(chunk.get('content', '')) for chunk in chunks)
        
        return JSONResponse({
            "success": True,
            "filename": filename,
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "total_characters": total_chars,
            "chunks": chunks[:3]  # 只返回前3個分塊預覽
        })
        
    except Exception as e:
        logger.error(f"獲取檔案詳情失敗: {e}")
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


@app.get("/api/bots/{bot_name}/knowledge/files")
async def get_knowledge_files(
    bot_name: str, 
    page: int = 1,           # 🆕 新增參數，默認值保持相容
    limit: int = 20,         # 🆕 新增參數，默認值保持相容
    search: str = "",        # 🆕 新增參數，默認值保持相容
    current_user: User = Depends(AdminAuth)
):
    """🔧 修改：在現有端點添加分頁參數，保持向後相容性"""
    try:
        # ✅ 參數驗證和清理
        page = max(1, page)
        limit = max(1, min(limit, 100))  # 限制最大值防止性能問題
        search = search.strip() if search else ""
        
        collection_name = f"collection_{bot_name}"
        
        # 🆕 調用新的分頁方法
        docs_result = vector_system.get_collection_documents_paginated(
            collection_name=collection_name,
            page=page, 
            limit=limit,
            search=search
        )
        
        return JSONResponse(docs_result)
        
    except Exception as e:
        logger.error(f"獲取文件清單失敗: {e}")
        return JSONResponse({
            "success": False,
            "message": str(e),
            "documents": [],
            "total": 0,
            "page": page,
            "limit": limit, 
            "total_pages": 0
        }, status_code=500)

@app.get("/api/bots/{bot_name}/conversations")
async def get_conversations(bot_name: str, page: int = 1, limit: int = 20, search: str = "", current_user: User = Depends(AdminAuth)):
    conv_logger = get_conversation_logger(bot_name)
    # 將機器人名稱轉換為 collection 名稱格式
    collection_name = f"collection_{bot_name}"
    
    conversations, total = conv_logger.get_conversations(
        limit=limit, 
        offset=(page - 1) * limit, 
        search=search if search else None,
        collection=collection_name  # 添加這個參數來過濾
    )
    
    logger.info(f"Conversations data for bot '{bot_name}': {conversations}")
    total_pages = (total + limit - 1) // limit if total > 0 else 1
    return JSONResponse({
        "success": True, "conversations": conversations, "total": total,
        "page": page, "limit": limit, "total_pages": total_pages
    })

@app.post("/api/bots/{bot_name}/knowledge/reset")
async def reset_knowledge_collection(
    bot_name: str, 
    request: Request,
    current_user: User = Depends(AdminAuth)
):
    """
    🗑️ 安全重置知識庫集合
    """
    try:
        data = await request.json()
        confirm_token = data.get("confirm_token", "")
        
        # 🛡️ 生成預期的確認令牌
        expected_token = f"RESET_{bot_name}_{int(time.time() // 3600)}"
        
        if confirm_token != expected_token:
            return JSONResponse({
                "success": False,
                "message": "需要有效的確認令牌",
                "required_token": expected_token,
                "warning": "此操作將刪除所有知識庫數據，請謹慎操作"
            }, status_code=400)
        
        collection_name = f"collection_{bot_name}"
        
        # 📊 重置前統計
        logger.info(f"開始重置集合: {collection_name} (用戶: {current_user.username})")
        
        docs_before = vector_system.get_collection_documents(collection_name, limit=1)
        total_before = docs_before.get('total', 0)
        
        if total_before == 0:
            return JSONResponse({
                "success": True,
                "message": "集合已經是空的，無需重置",
                "documents_before": 0,
                "documents_after": 0
            })
        
        # 🗑️ 執行多層次重置
        vectorstore = vector_system.get_or_create_vectorstore(collection_name)
        reset_success = False
        method_used = ""
        
        # 方法1: delete_collection (最徹底)
        try:
            if hasattr(vectorstore, 'delete_collection'):
                vectorstore.delete_collection()
                reset_success = True
                method_used = "delete_collection"
                logger.info("✅ 使用 delete_collection 方法")
            else:
                raise AttributeError("delete_collection 不可用")
                
        except Exception as e1:
            logger.warning(f"delete_collection 失敗: {e1}")
            
            # 方法2: 批量ID刪除
            try:
                all_docs = vectorstore.similarity_search("", k=5000)
                if all_docs:
                    chunk_ids = []
                    for doc in all_docs:
                        chunk_id = doc.metadata.get('chunk_id')
                        if chunk_id:
                            chunk_ids.append(chunk_id)
                    
                    if chunk_ids:
                        vectorstore.delete(ids=chunk_ids)
                        reset_success = True
                        method_used = f"bulk_delete_{len(chunk_ids)}_ids"
                        logger.info(f"✅ 批量刪除 {len(chunk_ids)} 個文檔")
                    
            except Exception as e2:
                logger.warning(f"批量刪除失敗: {e2}")
                
                # 方法3: SQL 直接清理
                try:
                    sql_result = perform_emergency_sql_cleanup(collection_name)
                    if sql_result['success']:
                        reset_success = True
                        method_used = f"sql_cleanup_{sql_result['deleted_rows']}_rows"
                        logger.info(f"✅ SQL清理: {sql_result['deleted_rows']} 條記錄")
                        
                except Exception as e3:
                    logger.error(f"SQL清理失敗: {e3}")
        
        # 🕐 等待操作生效
        time.sleep(3)
        
        # 🔍 驗證重置結果
        docs_after = vector_system.get_collection_documents(collection_name, limit=1)
        total_after = docs_after.get('total', 0)
        
        # 📝 記錄操作
        operation_log = {
            "timestamp": time.time(),
            "user": current_user.username,
            "collection": collection_name,
            "documents_before": total_before,
            "documents_after": total_after,
            "method_used": method_used,
            "success": total_after == 0
        }
        
        logger.info(f"集合重置完成: {operation_log}")
        
        if total_after == 0:
            return JSONResponse({
                "success": True,
                "message": f"集合 {bot_name} 已成功重置",
                "documents_before": total_before,
                "documents_after": total_after,
                "method_used": method_used,
                "operation_log": operation_log
            })
        else:
            return JSONResponse({
                "success": False,
                "message": f"集合重置不完整，還剩 {total_after} 個文檔",
                "documents_before": total_before,
                "documents_after": total_after,
                "method_used": method_used,
                "operation_log": operation_log
            }, status_code=500)
            
    except Exception as e:
        logger.error(f"集合重置失敗: {e}")
        return JSONResponse({
            "success": False,
            "message": f"重置失敗: {str(e)}"
        }, status_code=500)

@app.delete("/api/bots/{bot_name}/knowledge/files/{filename}")
async def delete_file(bot_name: str, filename: str, current_user: User = Depends(AdminAuth)):
    """🗑️ 刪除指定檔案 - 最終穩定版本"""
    try:
        # URL解碼處理中文檔名
        decoded_filename = urllib.parse.unquote(filename, encoding='utf-8')
        logger.info(f"🗑️ 刪除請求: {decoded_filename} from {bot_name}")
        
        collection_name = f"collection_{bot_name}"
        
        # 調用修正後的刪除方法
        result = vector_system.delete_by_file_ids(collection_name, decoded_filename)
        
        # 根據成功/失敗返回適當狀態碼
        if result["success"]:
            return JSONResponse(result, status_code=200)
        else:
            # 如果是文檔不存在，返回404；其他錯誤返回500
            status_code = 404 if "不存在" in result["message"] else 500
            return JSONResponse(result, status_code=status_code)
        
    except Exception as e:
        logger.error(f"刪除API異常: {e}", exc_info=True)
        return JSONResponse({
            "success": False, 
            "message": f"服務器內部錯誤: {str(e)}", 
            "deleted_chunks": 0,
            "filename": filename
        }, status_code=500)

@app.post("/api/search")
async def api_search(request: Request):
    """
    通用搜索 API 端點，用於從特定知識庫集合中搜索。
    """
    try:
        data = await request.json()
        query = data.get("query")
        collection_name = data.get("collection_name")
        k = data.get("k", 5)

        if not query or not collection_name:
            raise HTTPException(status_code=400, detail="需要提供查詢內容 (query) 和集合名稱 (collection_name)。")

        # 調用核心搜索功能
        results = vector_system.search(query=query, collection_name=collection_name, k=k)
        
        return JSONResponse({"success": True, "results": results})
    except Exception as e:
        logger.error(f"API search failed: {e}", exc_info=True)
        return JSONResponse({"success": False, "message": f"搜索時發生內部錯誤: {str(e)}"}, status_code=500)


# --- Health & Debug Routes ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Unified Gateway is running"}

@app.get("/routes")
async def get_all_routes():
    routes = []
    for route in app.routes:
        methods = list(route.methods) if hasattr(route, 'methods') else []
        routes.append({"path": route.path, "name": getattr(route, 'name', 'N/A'), "methods": methods})
    return JSONResponse(routes)


def perform_emergency_sql_cleanup(collection_name: str) -> Dict:
    """
    緊急 SQL 清理方案
    """
    result = {
        'success': False,
        'deleted_rows': 0,
        'tables_processed': [],
        'error': None
    }
    
    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            result['error'] = "DATABASE_URL 未設置"
            return result
        
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        # 查找相關表
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND (table_name LIKE %s OR table_name LIKE 'langchain_%');
        """, (f'%{collection_name}%',))
        
        tables = cursor.fetchall()
        total_deleted = 0
        
        for table_tuple in tables:
            table_name = table_tuple[0]
            
            try:
                # 檢查是否有元數據字段
                cursor.execute(f"""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = '{table_name}' 
                    AND column_name IN ('cmetadata', 'metadata');
                """)
                
                metadata_columns = cursor.fetchall()
                
                if metadata_columns:
                    metadata_col = metadata_columns[0][0]
                    
                    # 刪除匹配的記錄
                    if collection_name in table_name:
                        # 直接清空表
                        delete_query = f"DELETE FROM {table_name};"
                        cursor.execute(delete_query)
                    else:
                        # 根據元數據刪除
                        delete_query = f"""
                            DELETE FROM {table_name} 
                            WHERE {metadata_col}::text LIKE %s;
                        """
                        cursor.execute(delete_query, (f'%{collection_name}%',))
                    
                    deleted_count = cursor.rowcount
                    total_deleted += deleted_count
                    
                    result['tables_processed'].append({
                        'table': table_name,
                        'deleted': deleted_count
                    })
                    
                    logger.info(f"表 {table_name}: 刪除 {deleted_count} 條記錄")
                    
            except Exception as table_error:
                logger.warning(f"處理表 {table_name} 時出錯: {table_error}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        result.update({
            'success': total_deleted > 0,
            'deleted_rows': total_deleted
        })
        
        logger.info(f"SQL 清理完成: 總共刪除 {total_deleted} 條記錄")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"SQL 清理失敗: {e}")
    
    return result

@app.post("/api/emergency/clear-test01")
async def emergency_clear_test01(current_user: User = Depends(AdminAuth)):
    """
    🚨 緊急清空 test_01 集合
    """
    try:
        import psycopg2
        import os
        import time
        
        logger.info(f"緊急清空 test_01 - 用戶: {current_user.username}")
        
        # 連接數據庫
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return JSONResponse({
                "success": False,
                "message": "DATABASE_URL 未設置"
            }, status_code=500)
        
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        results = []
        total_deleted = 0
        
        # 查找相關表
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE '%langchain%';
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        results.append(f"找到 {len(tables)} 個相關表: {', '.join(tables)}")
        
        # 清空每個表中的 test_01 數據
        for table in tables:
            try:
                # 先查看有多少條記錄
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {table} 
                    WHERE cmetadata::text LIKE '%test_01%' 
                    OR cmetadata::text LIKE '%collection_test_01%';
                """)
                
                count_before = cursor.fetchone()[0]
                
                if count_before > 0:
                    # 執行刪除
                    cursor.execute(f"""
                        DELETE FROM {table} 
                        WHERE cmetadata::text LIKE '%test_01%' 
                        OR cmetadata::text LIKE '%collection_test_01%';
                    """)
                    
                    deleted = cursor.rowcount
                    total_deleted += deleted
                    results.append(f"表 {table}: 清空前 {count_before} 條，刪除 {deleted} 條")
                else:
                    results.append(f"表 {table}: 沒有 test_01 相關數據")
                    
            except Exception as e:
                results.append(f"表 {table} 處理失敗: {str(e)}")
        
        # 提交事務
        conn.commit()
        cursor.close()
        conn.close()
        
        # 等待一下讓數據庫操作生效
        time.sleep(2)
        
        # 驗證清空結果
        try:
            from vector_builder_langchain import OptimizedVectorSystem
            system = OptimizedVectorSystem()
            vectorstore = system.get_or_create_vectorstore("collection_test_01")
            remaining_docs = vectorstore.similarity_search("", k=100)
            remaining_count = len(remaining_docs)
            
            results.append(f"驗證結果: 剩餘 {remaining_count} 個文檔")
            
        except Exception as e:
            results.append(f"驗證失敗: {str(e)}")
        
        success = total_deleted > 0
        message = f"清空完成，總計刪除 {total_deleted} 條記錄" if success else "沒有找到需要刪除的數據"
        
        return JSONResponse({
            "success": success,
            "message": message,
            "total_deleted": total_deleted,
            "details": results
        })
        
    except Exception as e:
        logger.error(f"緊急清空失敗: {e}")
        return JSONResponse({
            "success": False,
            "message": f"清空失敗: {str(e)}"
        }, status_code=500)

# 📍 步驟2：添加獲取所有集合狀態的 API
@app.get("/api/emergency/collections-status")
async def get_collections_status(current_user: User = Depends(AdminAuth)):
    """
    📊 獲取所有集合的狀態
    """
    try:
        import psycopg2
        import os
        
        database_url = os.getenv("DATABASE_URL")
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        # 查找相關表
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE '%langchain%';
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        collections_status = {}
        
        for table in tables:
            try:
                # 統計每個集合的文檔數量
                cursor.execute(f"""
                    SELECT 
                        cmetadata::text,
                        COUNT(*) as count
                    FROM {table} 
                    GROUP BY cmetadata::text
                    HAVING COUNT(*) > 0;
                """)
                
                rows = cursor.fetchall()
                table_stats = {}
                
                for row in rows:
                    metadata_text = row[0]
                    count = row[1]
                    
                    # 嘗試提取集合名稱
                    if 'test_01' in metadata_text:
                        table_stats['test_01'] = table_stats.get('test_01', 0) + count
                    elif 'collection_' in metadata_text:
                        # 簡單的正則提取
                        import re
                        match = re.search(r'collection_(\w+)', metadata_text)
                        if match:
                            collection = match.group(1)
                            table_stats[collection] = table_stats.get(collection, 0) + count
                
                if table_stats:
                    collections_status[table] = table_stats
                    
            except Exception as e:
                collections_status[table] = {"error": str(e)}
        
        cursor.close()
        conn.close()
        
        return JSONResponse({
            "success": True,
            "collections_status": collections_status
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"獲取狀態失敗: {str(e)}"
        }, status_code=500)



# --- Entrypoint ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)