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

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- In-memory State for Bot Instances ---
global_bot_instances: Dict[str, ChatbotInstance] = {}

class BotManager:
    """Manages bot instances in-memory within a single process."""

    def __init__(self):
        logger.info("✅ In-Memory BotManager class initialized (DB-Integrated).")

    def start_bot(self, bot_name: str, main_app) -> Dict:
        # ✨ 關鍵更動：延遲導入以解決循環依賴
        try:
            from gateway_server import db_bot_manager
        except ImportError:
            logger.error("Could not import db_bot_manager from gateway_server. This indicates a serious issue.")
            return {"success": False, "message": "Internal server error: Cannot access bot configuration."}

        if bot_name in global_bot_instances:
            return {"success": False, "message": "Bot is already running."}

        # 從資料庫獲取設定
        config = db_bot_manager.get_bot_config(bot_name)
        
        if not config:
            logger.error(f"Attempted to start bot '{bot_name}', but config was not found in the database.")
            return {"success": False, "message": "Bot config not found in database."}
        
        try:
            logger.info(f"Starting bot '{bot_name}' in-memory with DB config...")
            
            # ✨ 關鍵更動：將完整的 config 字典傳遞給 ChatbotInstance
            bot_instance = ChatbotInstance(config=config)
            
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

        # Remove the instance from our dictionary, effectively making it inactive.
        del global_bot_instances[bot_name]
        
        logger.info(f"Bot '{bot_name}' instance removed.")
        
        # A proper garbage collection would be needed here in a long-running application.
        import gc
        gc.collect()

        # Remove the mounted routes from the main application
        routes_to_remove = [
            route for route in main_app.routes 
            if hasattr(route, 'path') and route.path.startswith(f"/{bot_name}")
        ]
        if routes_to_remove:
            for route in routes_to_remove:
                main_app.routes.remove(route)
                logger.info(f"Removed route: {getattr(route, 'path', 'N/A')}")
        else:
            logger.warning(f"Could not find routes to unmount for bot '{bot_name}'. They might have been removed already.")

        return {"success": True, "message": f"Bot {bot_name} stopped and unmounted."}

# A single instance that can be imported by other modules
bot_manager = BotManager()
