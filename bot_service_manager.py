import os
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Project-level Imports ---
from auth_middleware import User
from chatbot_instance import ChatbotInstance
# ❗️ 關鍵更動：導入資料庫機器人管理器
from gateway_server import db_bot_manager

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables & Paths ---
# 🗑️ 移除：不再需要從檔案系統讀取設定
# ROOT_DIR = Path(__file__).parent
# BOT_CONFIGS_DIR = ROOT_DIR / "bot_configs"
# BOT_CONFIGS_DIR.mkdir(exist_ok=True)

# --- In-memory State for Bot Instances ---
global_bot_instances: Dict[str, ChatbotInstance] = {}

class BotManager:
    """Manages bot instances in-memory within a single process."""

    def __init__(self):
        logger.info("✅ In-Memory BotManager class initialized (DB-Integrated).")

    # 🗑️ 移除：此功能已由 gateway_server.py 中的 /api/bots 端點處理
    # def get_all_bots(self) -> List[Dict]:
    #     ...

    # 🗑️ 移除：此功能已由 db_bot_manager 取代
    # def get_bot_config(self, bot_name: str) -> Optional[Dict]:
    #     ...

    def start_bot(self, bot_name: str, main_app) -> Dict:
        if bot_name in global_bot_instances:
            return {"success": False, "message": "Bot is already running."}

        # ✨ 關鍵更動：從資料庫獲取設定
        config = db_bot_manager.get_bot_config(bot_name)
        
        if not config:
            logger.error(f"Attempted to start bot '{bot_name}', but config was not found in the database.")
            return {"success": False, "message": "Bot config not found in database."}
        
        try:
            logger.info(f"Starting bot '{bot_name}' in-memory with DB config...")
            # Create the bot instance
            bot_instance = ChatbotInstance(bot_name)
            # Store the instance
            global_bot_instances[bot_name] = bot_instance
            # Mount the bot's FastAPI app onto the main gateway app
            main_app.mount(f"/{bot_name}", bot_instance.app)
            logger.info(f"✅ Bot '{bot_name}' mounted on path /{bot_name}")
            return {"success": True, "message": f"Bot {bot_name} started and mounted."}
        except Exception as e:
            logger.error(f"Failed to start bot '{bot_name}': {e}", exc_info=True)
            return {"success": False, "message": f"Failed to start bot: {e}"}

    def stop_bot(self, bot_name: str, main_app) -> Dict:
        if bot_name not in global_bot_instances:
            return {"success": False, "message": "Bot not running."}

        # Unmount the routes. This is tricky in FastAPI. A simple approach is to just remove the instance.
        # For a real production system, you might need a more robust unmounting mechanism.
        # We will remove it from our dictionary, effectively making it inactive.
        del global_bot_instances[bot_name]
        
        # To truly unmount, we would need to rebuild the app's routes, which is complex.
        # A simpler solution for now is that a stopped bot's routes will still exist but will fail if accessed.
        # Or better, we can add a middleware to check if the bot is active.
        logger.info(f"Bot '{bot_name}' instance removed. Routes are still active but will be inaccessible.")
        # A proper garbage collection would be needed here in a long-running application.
        import gc
        gc.collect()

        # ✨ 新增：從主應用程式中移除掛載點
        # 這是一個比較安全的作法，可以防止停止的機器人路由繼續被訪問
        # 注意：這會修改 app.routes 列表，在某些複雜情境下可能需要更精細的處理
        routes_to_remove = [
            route for route in main_app.routes 
            if hasattr(route, 'path') and route.path.startswith(f"/{bot_name}")
        ]
        for route in routes_to_remove:
            main_app.routes.remove(route)
            logger.info(f"Removed route: {route.path}")

        return {"success": True, "message": f"Bot {bot_name} stopped and unmounted."}

# A single instance that can be imported by other modules
bot_manager = BotManager()
