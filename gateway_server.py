import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, Request, Response, Header, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# Load environment variables at the very top
load_dotenv()

# --- Project-level Imports ---
from config import app_config
from auth_middleware import AdminAuth, User, auth_response, JWTManager
from user_manager import user_manager
from bot_service_manager import bot_manager, global_bot_instances
from conversation_logger_simple import create_logger_instance
from vector_builder_langchain import OptimizedVectorSystem

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("gateway")

# --- Global Variables & Paths ---
ROOT_DIR = Path(__file__).resolve().parent

# --- FastAPI App Initialization ---
app = FastAPI(title="Unified API Gateway", version="3.0")
templates = Jinja2Templates(directory=str(ROOT_DIR))

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
    return JSONResponse(bot_manager.get_all_bots())

@app.post("/api/bots/{bot_name}/start")
async def start_bot(bot_name: str, current_user: User = Depends(AdminAuth)):
    return JSONResponse(bot_manager.start_bot(bot_name, app))

@app.post("/api/bots/{bot_name}/stop")
async def stop_bot(bot_name: str, current_user: User = Depends(AdminAuth)):
    return JSONResponse(bot_manager.stop_bot(bot_name, app))

@app.get("/api/bots/{bot_name}/config")
async def get_bot_config(bot_name: str, current_user: User = Depends(AdminAuth)):
    config = bot_manager.get_bot_config(bot_name)
    if config:
        return JSONResponse({"success": True, "config": config})
    return JSONResponse({"success": False, "message": "Config not found"}, status_code=404)

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
async def get_knowledge_files(bot_name: str, current_user: User = Depends(AdminAuth)):
    try:
        collection_name = f"collection_{bot_name}"
        docs_result = vector_system.get_collection_documents(collection_name)
        return JSONResponse(docs_result)
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@app.get("/api/bots/{bot_name}/conversations")
async def get_conversations(bot_name: str, page: int = 1, limit: int = 20, search: str = "", current_user: User = Depends(AdminAuth)):
    conv_logger = get_conversation_logger(bot_name)
    conversations, total = conv_logger.get_conversations(limit=limit, offset=(page - 1) * limit, search=search if search else None)
    total_pages = (total + limit - 1) // limit if total > 0 else 1
    return JSONResponse({
        "success": True, "conversations": conversations, "total": total,
        "page": page, "limit": limit, "total_pages": total_pages
    })

@app.get("/api/bots/{bot_name}/knowledge/files/{filename}/details")
async def get_file_details(bot_name: str, filename: str, current_user: User = Depends(AdminAuth)):
    """獲取文件詳細資訊"""
    try:
        collection_name = f"collection_{bot_name}"
        chunks = vector_system.get_document_chunks(collection_name, filename)
        
        if not chunks:
            return JSONResponse({"success": False, "message": "文件不存在"}, status_code=404)
        
        # 計算統計資訊
        total_tokens = sum(chunk.get('token_count', 0) for chunk in chunks)
        
        return JSONResponse({
            "success": True,
            "filename": filename,
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "chunks": chunks
        })
        
    except Exception as e:
        logger.error(f"獲取文件詳情失敗: {e}")
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)
    

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
    """刪除指定檔案及其向量"""
    try:
        collection_name = f"collection_{bot_name}"
        result = vector_system.delete_document(collection_name, filename)
        
        if result["success"]:
            return JSONResponse(result)
        else:
            status_code = 404 if "不存在" in result["message"] else 500
            return JSONResponse(result, status_code=status_code)
            
    except Exception as e:
        logger.error(f"刪除檔案失敗: {e}")
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

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


# --- Entrypoint ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)