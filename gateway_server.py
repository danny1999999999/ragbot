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
from fastapi.responses import JSONResponse, Response, PlainTextResponse

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
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))

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

app = FastAPI(title="Internal API Gateway", version="1.1")


# -------------------------
# 工具函式
# -------------------------
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


# -------------------------
# 健康檢查 & 根頁
# -------------------------
@app.get("/")
async def root():
    return PlainTextResponse("API Gateway is running. Try: /<bot>/health")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "bot_configs_dir": str(BOT_CONFIGS_DIR),
        "registry_size": len(REGISTRY),
        "file_cache_size": len(_FILE_CACHE),
        "available_bots": list(REGISTRY.keys()) + [f.stem for f in BOT_CONFIGS_DIR.glob("*.json")]
    }


# -------------------------
# 管理端點：動態註冊/取消註冊
# -------------------------
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


# -------------------------
# 核心代理函式
# -------------------------
async def proxy_to_bot(bot_name: str, request: Request, stripped_path: str = "") -> Response:
    """將 /{bot_name}/<stripped_path> 轉到 127.0.0.1:{port}/<stripped_path>"""
    
    # 特殊處理：如果 bot_name 是常見的 API 端點，嘗試智能路由
    if bot_name in ['chat', 'api', 'stream', 'upload', 'download']:
        referer = request.headers.get("referer", "")
        logger.info(f"[gateway] Detecting relative path request: /{bot_name}, referer: '{referer}'")
        
        real_bot_name = None
        
        # 方法1: 從 referer 提取（如果有的話）
        if referer:
            match = re.search(r'/([^/]+)/?(?:\?|$)', referer)
            if match:
                candidate = match.group(1)
                logger.debug(f"[gateway] Extracted from referer: '{candidate}'")
                if candidate not in ['chat', 'api', 'stream', 'upload', 'download']:
                    real_bot_name = candidate
        
        # 方法2: 如果 referer 失敗，使用智能默認選擇
        if not real_bot_name:
            # 獲取可用 bot 列表
            available_bots = []
            if REGISTRY:
                available_bots.extend(REGISTRY.keys())
            
            # 也檢查配置文件
            try:
                for cfg_file in BOT_CONFIGS_DIR.glob("*.json"):
                    bot_name_from_file = cfg_file.stem
                    if bot_name_from_file not in available_bots:
                        available_bots.append(bot_name_from_file)
            except Exception:
                pass
            
            # 過濾掉非真實 bot
            real_bots = [b for b in available_bots if b not in ['chat', 'api', 'stream', 'upload', 'download']]
            logger.info(f"[gateway] Real bots available: {real_bots}")
            
            # 智能選擇策略
            if len(real_bots) == 1:
                # 只有一個 bot，直接使用
                real_bot_name = real_bots[0]
                logger.info(f"[gateway] Only one bot available, using: {real_bot_name}")
            elif len(real_bots) > 1:
                # 多個 bot，使用優先級選擇
                priority_bots = ['test_01', 'test_02', 'test_03']  # 優先級順序
                for priority_bot in priority_bots:
                    if priority_bot in real_bots:
                        real_bot_name = priority_bot
                        logger.info(f"[gateway] Multiple bots available, using priority bot: {real_bot_name}")
                        break
                
                # 如果沒有優先級匹配，使用第一個
                if not real_bot_name:
                    real_bot_name = real_bots[0]
                    logger.info(f"[gateway] No priority match, using first available: {real_bot_name}")
        
        # 如果找到了真實的 bot，進行重定向
        if real_bot_name:
            logger.info(f"[gateway] Redirecting relative path: /{bot_name} -> /{real_bot_name}/{bot_name}")
            # 重新構造完整路徑
            if stripped_path:
                new_path = f"{bot_name}/{stripped_path}"
            else:
                new_path = bot_name
            return await proxy_to_bot(real_bot_name, request, stripped_path=new_path)
        
        # 如果都失敗了，返回詳細錯誤
        return JSONResponse({
            "success": False, 
            "message": f"無法確定目標 bot，請使用完整路徑：/bot_name/{bot_name}",
            "available_bots": available_bots,
            "referer": referer,
            "hint": "嘗試直接訪問：http://localhost:8000/test_02/ 而不是通過其他方式"
        }, status_code=400)
    
    port = get_bot_port(bot_name)
    if port is None:
        logger.warning(f"[gateway] Bot '{bot_name}' not found or not configured")
        return JSONResponse({"success": False, "message": f"Bot '{bot_name}' 不存在或未配置 port"}, status_code=404)

    # --- MODIFIED FOR RAILWAY DEPLOYMENT ---
    # Use the bot_name as the hostname for inter-service communication
    # This assumes the service name in Railway matches the bot_name
    backend_host = bot_name
    
    path = stripped_path
    if not path.startswith("/"):
        path = "/" + path
    query = request.url.query
    
    if query:
        target_url = f"http://{backend_host}:{port}{path}?{query}"
    else:
        target_url = f"http://{backend_host}:{port}{path}"
    # --- END MODIFICATION ---
    
    method = request.method
    
    # 檢查是否為 WebSocket 升級請求
    if (request.headers.get("upgrade", "").lower() == "websocket" or 
        request.headers.get("connection", "").lower() == "upgrade"):
        logger.warning(f"[gateway] WebSocket upgrade request detected for /{bot_name}/{stripped_path} - not supported")
        return JSONResponse({"success": False, "message": "WebSocket connections not supported through gateway"}, status_code=501)

    # 準備 headers：去除 hop-by-hop，補上 X-Forwarded-*
    headers: Dict[str, str] = dict(request.headers)
    for h in list(headers.keys()):
        if h.lower() in HOP_BY_HOP_HEADERS:
            headers.pop(h, None)

    headers["x-forwarded-proto"] = request.url.scheme
    headers["x-forwarded-host"] = request.headers.get("host", "")
    headers["x-forwarded-for"] = (request.client.host if request.client else "")
    headers["x-forwarded-prefix"] = f"/{bot_name}"

    # 讀取 request body
    body = await request.body()
    content = body if body else None
    
    # 記錄請求詳情，幫助診斷問題
    logger.info(f"[gateway] {method} /{bot_name}/{stripped_path} -> {target_url}")
    if content:
        logger.debug(f"[gateway] Request body length: {len(content)}, Content-Type: {headers.get('content-type', 'unknown')}")
    
    # 修正：設定完整的 timeout 參數
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT,
        read=READ_TIMEOUT,
        write=WRITE_TIMEOUT,
        pool=POOL_TIMEOUT
    )

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            # 修改：使用普通請求而非 streaming，避免 chunked encoding 問題
            resp = await client.request(method, target_url, headers=headers, content=content)
            logger.info(f"[gateway] ✅ {method} /{bot_name}/{stripped_path} -> {resp.status_code}")
            logger.debug(f"[gateway] Backend response: {resp.status_code}, Content-Type: {resp.headers.get('content-type')}")
            
            # 如果是錯誤狀態，記錄響應內容幫助診斷
            if resp.status_code >= 400:
                logger.error(f"[gateway] Backend error {resp.status_code} for /{bot_name}/{stripped_path}")
                try:
                    error_text = resp.text[:500]  # 只記錄前500個字符
                    logger.error(f"[gateway] Error response: {error_text}")
                except:
                    pass
            
            # 準備回應 headers
            response_headers: Dict[str, str] = {}
            skipped_headers = []
            for k, v in resp.headers.items():
                lk = k.lower()
                # 跳過 hop-by-hop headers
                if lk in HOP_BY_HOP_HEADERS:
                    skipped_headers.append(f"{k} (hop-by-hop)")
                    continue
                # 修正 Location: /xxx → /{bot_name}/xxx（避免 302 把使用者帶回根）
                if lk == "location" and isinstance(v, str) and v.startswith("/"):
                    v = f"/{bot_name}{v}"
                response_headers[k] = v
            
            if skipped_headers:
                logger.debug(f"[gateway] Skipped headers: {', '.join(skipped_headers)}")

            # 處理 HTML 內容中的相對路徑，修正靜態資源路徑
            content = resp.content
            content_type = resp.headers.get("content-type", "").lower()
            
            # 只修改 HTML 內容，不修改 API 響應（如 JSON）
            if content_type.startswith("text/html") and content and resp.status_code == 200:
                try:
                    # 解碼 HTML 內容
                    html_content = content.decode('utf-8')
                    
                    # 修正常見的靜態資源路徑
                    html_content = html_content.replace('href="/', f'href="/{bot_name}/')
                    html_content = html_content.replace("href='/", f"href='/{bot_name}/")
                    html_content = html_content.replace('src="/', f'src="/{bot_name}/')
                    html_content = html_content.replace("src='/", f"src='/{bot_name}/")
                    html_content = html_content.replace('action="/', f'action="/{bot_name}/')
                    html_content = html_content.replace("action='/", f"action='/{bot_name}/")
                    
                    # 修正 fetch() 和其他 JavaScript API 調用 - 絕對路徑
                    html_content = html_content.replace('fetch("/', f'fetch("/{bot_name}/')
                    html_content = html_content.replace("fetch('/", f"fetch('/{bot_name}/")
                    
                    # 特別處理 /api/ 路徑 - 這是遺漏的關鍵！
                    html_content = html_content.replace('"/api/', f'"/{bot_name}/api/')
                    html_content = html_content.replace("'/api/", f"'/{bot_name}/api/")
                    
                    # 修正相對路徑的常見 API 端點
                    common_endpoints = ['chat', 'api', 'stream', 'upload', 'download', 'health']
                    for endpoint in common_endpoints:
                        # fetch('chat') → fetch('/test_02/chat')
                        html_content = html_content.replace(f"fetch('{endpoint}'", f"fetch('/{bot_name}/{endpoint}'")
                        html_content = html_content.replace(f'fetch("{endpoint}"', f'fetch("/{bot_name}/{endpoint}"')
                        
                        # fetch('./chat') → fetch('/test_02/chat')  
                        html_content = html_content.replace(f"fetch('./{endpoint}'", f"fetch('/{bot_name}/{endpoint}'")
                        html_content = html_content.replace(f'fetch("./{endpoint}"', f'fetch("/{bot_name}/{endpoint}"')
                        
                        # 其他 AJAX 調用
                        html_content = html_content.replace(f"url: '{endpoint}'", f"url: '/{bot_name}/{endpoint}'")
                        html_content = html_content.replace(f'url: "{endpoint}"', f'url: "/{bot_name}/{endpoint}"')
                        html_content = html_content.replace(f"url:'{endpoint}'", f"url:'/{bot_name}/{endpoint}'")
                        html_content = html_content.replace(f'url:"{endpoint}"', f'url:"/{bot_name}/{endpoint}"')
                    
                    # 通用的相對路徑模式 (更激進的方法)
                    # 匹配 fetch('單詞') 模式，但跳過已經有 / 的
                    pattern = r"fetch\(['\"]([a-zA-Z][a-zA-Z0-9_-]*)['\"]"
                    def replace_fetch(match):
                        endpoint = match.group(1)
                        if not endpoint.startswith('/') and not endpoint.startswith('http'):
                            return f"fetch('/{bot_name}/{endpoint}'"
                        return match.group(0)
                    html_content = re.sub(pattern, replace_fetch, html_content)
                    
                    # 重新編碼
                    content = html_content.encode('utf-8')
                    
                    # 更新 Content-Length（如果有的話）
                    if "content-length" in response_headers:
                        response_headers["content-length"] = str(len(content))
                        
                    logger.debug(f"[gateway] Modified HTML content for bot {bot_name}, new length: {len(content)}")
                    
                except Exception as e:
                    logger.warning(f"[gateway] Failed to modify HTML content: {e}")
                    # 如果處理失敗，使用原始內容
                    content = resp.content

            # 使用 Response 而非 StreamingResponse，避免 chunked encoding 問題
            final_resp = Response(
                content=content,
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get("content-type")
            )

            # 多重 Set-Cookie 支援
            try:
                for sc in resp.headers.get_list("set-cookie"):
                    final_resp.headers.append("set-cookie", sc)
            except Exception:
                pass

            logger.debug(f"[gateway] ✅ {method} /{bot_name}/{stripped_path} -> {resp.status_code}")
            return final_resp

    except httpx.ConnectError as e:
        logger.error(f"[gateway] Connection error to bot {bot_name} on port {port}: {e}")
        return JSONResponse({"success": False, "message": f"Bot '{bot_name}' 未啟動或無法連線 (port: {port})"}, status_code=503)
    except httpx.TimeoutException as e:
        logger.error(f"[gateway] Timeout error for bot {bot_name}: {e}")
        return JSONResponse({"success": False, "message": "後端回應逾時"}, status_code=504)
    except httpx.ReadError as e:
        logger.error(f"[gateway] Read error for bot {bot_name}: {e}")
        return JSONResponse({"success": False, "message": "讀取後端回應時發生錯誤"}, status_code=502)
    except httpx.WriteError as e:
        logger.error(f"[gateway] Write error for bot {bot_name}: {e}")
        return JSONResponse({"success": False, "message": "向後端發送請求時發生錯誤"}, status_code=502)
    except Exception as e:
        logger.exception(f"[gateway] 代理錯誤 for bot {bot_name}: {e}")
        return JSONResponse({"success": False, "message": f"代理錯誤: {e}"}, status_code=502)


# -------------------------
# 通用代理路由（請放在其它固定路由之後）
# -------------------------
@app.api_route("/{bot_name}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def gateway_root(bot_name: str, request: Request):
    # 導向後端的根路徑 "/"
    return await proxy_to_bot(bot_name, request, stripped_path="")

@app.api_route("/{bot_name}/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def gateway_catchall(bot_name: str, full_path: str, request: Request):
    # 導向後端的相對路徑
    return await proxy_to_bot(bot_name, request, stripped_path=full_path)


# -------------------------
# Entrypoint
# -------------------------
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
    print("🔌 使用方式: http://localhost:%d/<bot>/ ... 例如 http://localhost:%d/test_01/health" % (GATEWAY_PORT, GATEWAY_PORT))
    print("============================================================")
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT, log_level="info")