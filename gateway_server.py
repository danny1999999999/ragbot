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

# --- Entrypoint ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)