#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API Gateway (FastAPI)
- Public URL: http://localhost:8000/<bot>/...
- Backend:     http://127.0.0.1:<port>/...
- Port lookup: ./bot_configs/<bot>.json 內的 {"port": 9003}
- 動態註冊：/_gateway/register / _gateway/unregister（X-Admin-Token）
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response, Header
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from starlette.background import BackgroundTask


import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from config import app_config  # ⭐ 統一導入

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("gateway")

ROOT_DIR = Path(__file__).resolve().parent
BOT_CONFIGS_DIR = Path(os.getenv("BOT_CONFIGS_DIR", str(ROOT_DIR / "bot_configs")))
GATEWAY_PORT = int(os.getenv("PORT", "8000"))

# hop-by-hop headers 不應被代理轉發
HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade"
}

# 逾時設定（秒）
READ_TIMEOUT = float(os.getenv("GW_READ_TIMEOUT", "180"))
WRITE_TIMEOUT = float(os.getenv("GW_WRITE_TIMEOUT", "180"))
CONNECT_TIMEOUT = float(os.getenv("GW_CONNECT_TIMEOUT", "5"))
POOL_TIMEOUT = float(os.getenv("GW_POOL_TIMEOUT", "5"))

# 管理密鑰（建議正式環境務必設定）
GATEWAY_ADMIN_TOKEN = os.getenv("GATEWAY_ADMIN_TOKEN", "")

# 動態註冊表（管理介面推送）
REGISTRY: Dict[str, int] = {}

# 檔案查詢快取：bot -> (port, mtime)
_FILE_CACHE: Dict[str, Tuple[int, float]] = {}

app = FastAPI(title="Internal API Gateway", version="1.2")


# --------------------------------------------------
# --- 新增：反向代理到管理介面 ---
# --------------------------------------------------
MANAGER_SERVICE_URL = "http://127.0.0.1:9001"
proxy_client = httpx.AsyncClient(base_url=MANAGER_SERVICE_URL, follow_redirects=True)

@app.api_route("/manager/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def reverse_proxy_manager(request: Request, path: str):
    """
    捕獲所有 /manager/ 的請求，並轉發到 9001 port 的管理服務。
    """
    # 修正路徑，確保它以 / 開頭
    if not path.startswith("/"):
        path = "/" + path

    url = httpx.URL(path=path, query=request.url.query.encode("utf-8"))
    
    rp_request = proxy_client.build_request(
        method=request.method,
        url=url,
        headers=request.headers.raw,
        content=await request.body()
    )
    
    try:
        rp_response = await proxy_client.send(rp_request, stream=True)
        
        excluded_headers = ["content-encoding", "content-length", "transfer-encoding", "connection"]
        headers = [(name, value) for (name, value) in rp_response.headers.items() if name.lower() not in excluded_headers]
        
        # 處理重定向路徑
        for i, (name, value) in enumerate(headers):
            if name.lower() == 'location':
                # 如果重定向到根，則加上 /manager 前綴
                if value.startswith('/'):
                    headers[i] = (name, f"/manager{value}")
                logger.info(f"Redirecting to: {headers[i][1]}")

        return StreamingResponse(
            rp_response.aiter_raw(),
            status_code=rp_response.status_code,
            headers=headers,
            background=BackgroundTask(rp_response.aclose)
        )
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"detail": "Manager service is unavailable."})


# ------------------------
# 工具函式
# ------------------------
def get_bot_port(bot_name: str) -> Optional[int]:
    """先查註冊表，再查檔案（含快取）。"""
    # 1) 註冊表命中
    if bot_name in REGISTRY:
        logger.debug(f"[gateway] Bot {bot_name} found in registry: {REGISTRY[bot_name]}")
        return REGISTRY[bot_name]

    # 2) bot_configs 快取
    cfg = BOT_CONFIGS_DIR / f"{bot_name}.json"
    if not cfg.exists():
        logger.debug(f"[gateway] Config file not found: {cfg}")
        return None
    
    try:
        mtime = cfg.stat().st_mtime
        cached = _FILE_CACHE.get(bot_name)
        if cached and cached[1] == mtime:
            logger.debug(f"[gateway] Bot {bot_name} found in cache: {cached[0]}")
            return cached[0]
        
        # 讀取檔案，使用 utf-8-sig 處理 BOM 問題
        try:
            data = json.loads(cfg.read_text(encoding="utf-8-sig"))
        except UnicodeDecodeError:
            # 如果 utf-8-sig 失敗，嘗試 utf-8
            data = json.loads(cfg.read_text(encoding="utf-8"))
        
        port = int(data.get("port")) if data.get("port") else None
        if port:
            _FILE_CACHE[bot_name] = (port, mtime)
            logger.debug(f"[gateway] Bot {bot_name} config loaded: port={port}")
        else:
            logger.warning(f"[gateway] No port found in config for bot {bot_name}")
        
        return port
    except json.JSONDecodeError as e:
        logger.error(f"[gateway] JSON decode error for {cfg}: {e}")
        return None
    except Exception as e:
        logger.warning(f"[gateway] 讀取 {cfg} 失敗: {e}")
        return None


def build_backend_url(port: int, path: str, query: str) -> str:
    """組後端 URL；path 不含 /<bot> 前綴。"""
    if not path.startswith("/"):
        path = "/" + path
    if query:
        return f"http://127.0.0.1:{port}{path}?{query}"
    return f"http://127.0.0.1:{port}{path}"


def _auth_ok(x_admin_token: Optional[str]) -> bool:
    """管理端點的簡易驗證：未設定 token 則放行（開發方便），正式環境務必設置。"""
    if not GATEWAY_ADMIN_TOKEN:
        return True
    return x_admin_token == GATEWAY_ADMIN_TOKEN


# ------------------------
# 健康檢查 & 根頁
# ------------------------
@app.get("/")
async def root():
    return PlainTextResponse("API Gateway is running. Try: /<bot>/health or /manager/ to access the UI.")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "bot_configs_dir": str(BOT_CONFIGS_DIR),
        "registry_size": len(REGISTRY),
        "file_cache_size": len(_FILE_CACHE),
        "available_bots": list(REGISTRY.keys()) + [f.stem for f in BOT_CONFIGS_DIR.glob("*.json")]
    }


# ------------------------
# 管理端點：動態註冊/取消註冊
# ------------------------
@app.post("/_gateway/register")
async def gw_register(payload: dict, x_admin_token: Optional[str] = Header(default=None)):
    if not _auth_ok(x_admin_token):
        return JSONResponse({"success": False, "message": "unauthorized"}, status_code=401)
    bot = (payload.get("bot") or "").strip()
    port = payload.get("port")
    if not bot or not isinstance(port, int):
        return JSONResponse({"success": False, "message": "invalid payload"}, status_code=400)
    REGISTRY[bot] = port
    _FILE_CACHE.pop(bot, None)  # 清掉檔案快取避免新舊衝突
    logger.info(f"[gateway] ✅ registered bot={bot} port={port}")
    return {"success": True, "bot": bot, "port": port}

@app.post("/_gateway/unregister")
async def gw_unregister(payload: dict, x_admin_token: Optional[str] = Header(default=None)):
    if not _auth_ok(x_admin_token):
        return JSONResponse({"success": False, "message": "unauthorized"}, status_code=401)
    bot = (payload.get("bot") or "").strip()
    if not bot:
        return JSONResponse({"success": False, "message": "invalid payload"}, status_code=400)
    REGISTRY.pop(bot, None)
    _FILE_CACHE.pop(bot, None)
    logger.info(f"[gateway] 🔴 unregistered bot={bot}")
    return {"success": True, "bot": bot}


# ------------------------
# 核心代理函式
# ------------------------
async def proxy_to_bot(bot_name: str, request: Request, stripped_path: str = "") -> Response:
    """將 /{bot_name}/<stripped_path> 轉到 127.0.0.1:{port}/<stripped_path>"""
    
    # 特殊處理：如果 bot_name 是常見的 API 端點，嘗試智能路由
    if bot_name in ['chat', 'api', 'stream', 'upload', 'download', 'login']:
        referer = request.headers.get("referer", "")
        logger.info(f"[gateway] Detecting relative path request: /{bot_name}, referer: '{referer}'")
        
        real_bot_name = None
        
        # 方法1: 從 referer 提取（如果有的話）
        if referer:
            match = re.search(r'/(test_\d+)/', referer) # 尋找 /test_01/ 這樣的格式
            if match:
                real_bot_name = match.group(1)
                logger.info(f"[gateway] Extracted bot '{real_bot_name}' from referer: '{referer}'")

        if not real_bot_name:
            # 如果 referer 中沒有，則使用預設或第一個可用的 bot
            available_bots = [b for b in (list(REGISTRY.keys()) + [f.stem for f in BOT_CONFIGS_DIR.glob("*.json")]) if b.startswith('test_')]
            if 'test_01' in available_bots:
                real_bot_name = 'test_01'
            elif available_bots:
                real_bot_name = available_bots[0]
        
        if real_bot_name:
            logger.info(f"[gateway] Redirecting relative path: /{bot_name} -> /{real_bot_name}/{bot_name}")
            new_path = f"{bot_name}/{stripped_path}" if stripped_path else bot_name
            return await proxy_to_bot(real_bot_name, request, stripped_path=new_path)
        
        return JSONResponse({"success": False, "message": f"無法確定目標 bot，請使用完整路徑：/bot_name/{bot_name}"}, status_code=400)
    
    port = get_bot_port(bot_name)
    if port is None:
        logger.warning(f"[gateway] Bot '{bot_name}' not found or not configured")
        return JSONResponse({"success": False, "message": f"Bot '{bot_name}' 不存在或未配置 port"}, status_code=404)

    backend_host = "127.0.0.1"
    
    path = stripped_path
    if not path.startswith("/"):
        path = "/" + path
    query = request.url.query
    
    target_url = f"http://{backend_host}:{port}{path}"
    if query:
        target_url += f"?{query}"
    
    method = request.method
    
    headers: Dict[str, str] = dict(request.headers)
    for h in list(headers.keys()):
        if h.lower() in HOP_BY_HOP_HEADERS:
            headers.pop(h, None)

    headers["x-forwarded-proto"] = request.url.scheme
    headers["x-forwarded-host"] = request.headers.get("host", "")
    headers["x-forwarded-for"] = (request.client.host if request.client else "")
    headers["x-forwarded-prefix"] = f"/{bot_name}"

    body = await request.body()
    content = body if body else None
    
    logger.info(f"[gateway] {method} /{bot_name}/{stripped_path} -> {target_url}")
    
    timeout = httpx.Timeout(connect=CONNECT_TIMEOUT, read=READ_TIMEOUT, write=WRITE_TIMEOUT, pool=POOL_TIMEOUT)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            resp = await client.request(method, target_url, headers=headers, content=content)
            
            response_headers: Dict[str, str] = {}
            for k, v in resp.headers.items():
                if k.lower() not in HOP_BY_HOP_HEADERS:
                    if k.lower() == "location" and v.startswith("/"):
                        v = f"/{bot_name}{v}"
                    response_headers[k] = v
            
            final_content = resp.content
            
            return Response(
                content=final_content,
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get("content-type")
            )

    except httpx.ConnectError as e:
        logger.error(f"[gateway] Connection error to bot {bot_name} on port {port}: {e}")
        return JSONResponse({"success": False, "message": f"Bot '{bot_name}' 未啟動或無法連線 (port: {port})"}, status_code=503)
    except Exception as e:
        logger.exception(f"[gateway] 代理錯誤 for bot {bot_name}: {e}")
        return JSONResponse({"success": False, "message": f"代理錯誤: {e}"}, status_code=502)



# -------------------------
# 管理介面整合 (新增)
# -------------------------
from fastapi.templating import Jinja2Templates
from auth_middleware import AdminAuth, User, auth_response, JWTManager # 匯入認證
from user_manager import user_manager # 匯入使用者管理
from bot_service_manager import bot_manager # 匯入我們改造後的機器人總管

templates = Jinja2Templates(directory=str(ROOT_DIR))

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("manager_login.html", {"request": request})

@app.get("/manager", response_class=HTMLResponse)
async def manager_page(request: Request, current_user: User = Depends(AdminAuth)):
    """管理器主頁面 - 需要管理員權限"""
    return templates.TemplateResponse("manager_ui.html", {"request": request, "user": current_user})

@app.post("/api/login")
async def handle_login(request: Request):
    """處理登入 - 邏輯從 bot_service_manager 搬移至此"""
    try:
        data = await request.json()
        username = data.get("username", "").strip()
        password = data.get("password", "")

        if not username or not password:
            return JSONResponse({"success": False, "message": "請填寫用戶名和密碼"}, status_code=400)

        if not user_manager:
            raise HTTPException(status_code=503, detail="用戶系統未初始化")

        success, token_or_msg, user = user_manager.authenticate(
            username, password,
            ip_address=request.client.host if request.client else "unknown"
        )
        
        if success and user.role in ["admin", "super_admin"]:
            jwt_token = JWTManager.create_access_token(user)
            response = auth_response.create_login_response(user, jwt_token)
            logger.info(f"✅ 用戶 {username} 登入閘道器成功")
            return response
        elif success:
            return JSONResponse({"success": False, "message": "需要管理員權限"}, status_code=403)
        else:
            logger.warning(f"閘道器認證失敗: {username}")
            return JSONResponse({"success": False, "message": token_or_msg}, status_code=401)

    except Exception as e:
        logger.error(f"登入處理異常: {e}", exc_info=True)
        return JSONResponse({"success": False, "message": f"登入系統異常: {str(e)}"}, status_code=500)

@app.post("/api/logout")
async def handle_logout():
    return auth_response.create_logout_response()

@app.get("/api/bots")
async def get_all_bots(current_user: User = Depends(AdminAuth)):
    bots = bot_manager.get_all_bots()
    return JSONResponse(bots)

@app.post("/api/bots/{bot_name}/start")
async def start_bot(bot_name: str, current_user: User = Depends(AdminAuth)):
    result = await bot_manager.start_bot(bot_name, current_user)
    return JSONResponse(result)

@app.post("/api/bots/{bot_name}/stop")
async def stop_bot(bot_name: str, current_user: User = Depends(AdminAuth)):
    result = bot_manager.stop_bot(bot_name, current_user)
    return JSONResponse(result)


# -------------------------
# 通用代理路由（請放在其它固定路由之後）
# -------------------------
@app.api_route("/{bot_name}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def gateway_root(bot_name: str, request: Request):
    return await proxy_to_bot(bot_name, request, stripped_path="")

@app.api_route("/{bot_name}/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def gateway_catchall(bot_name: str, full_path: str, request: Request):
    return await proxy_to_bot(bot_name, request, stripped_path=full_path)


# ------------------------
# Entrypoint
# ------------------------
if __name__ == "__main__":
    print("============================================================")
    print("🌐 API Gateway")
    print("============================================================")
    print(f"🚀 服務端口: {GATEWAY_PORT}")
    print(f"📁 Bot 設定: {BOT_CONFIGS_DIR}")
    if GATEWAY_ADMIN_TOKEN:
        print("🔐 管理端點已啟用權杖保護（X-Admin-Token）")
    else:
        print("⚠️  開發模式未設 GATEWAY_ADMIN_TOKEN（正式環境請務必設定）")
    print("🏥 健康檢查: http://localhost:%d/health" % GATEWAY_PORT)
    print("🔌 使用方式: http://localhost:%d/<bot>/ ... 或 /manager/" % GATEWAY_PORT)
    print("============================================================")
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT, log_level="info")
