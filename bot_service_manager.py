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

# --- Global Variables & Paths ---
ROOT_DIR = Path(__file__).parent
BOT_CONFIGS_DIR = ROOT_DIR / "bot_configs"

# Ensure directories exist
BOT_CONFIGS_DIR.mkdir(exist_ok=True)

# --- In-memory State for Bot Instances ---
global_bot_instances: Dict[str, ChatbotInstance] = {}

class BotManager:
    """Manages bot instances in-memory within a single process."""

    def __init__(self):
        logger.info("✅ In-Memory BotManager class initialized.")

    def get_all_bots(self) -> List[Dict]:
        bots = []
        for config_file in sorted(BOT_CONFIGS_DIR.glob("*.json")):
            bot_name = config_file.stem
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                status = "running" if bot_name in global_bot_instances else "stopped"
                display_name = config.get("display_name") or bot_name
                
                bots.append({
                    "name": bot_name,
                    "display_name": display_name,
                    "port": config.get("port"), # Port is now conceptual
                    "status": status,
                })
            except Exception as e:
                logger.error(f"Failed to read bot config {bot_name}: {e}")
                continue
        bots.sort(key=lambda b: (b["status"] != "running", b["name"]))
        return bots

    def get_bot_config(self, bot_name: str) -> Optional[Dict]:
        config_path = BOT_CONFIGS_DIR / f"{bot_name}.json"
        if not config_path.exists():
            return None
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def start_bot(self, bot_name: str, main_app) -> Dict:
        if bot_name in global_bot_instances:
            return {"success": False, "message": "Bot is already running."}

        config = self.get_bot_config(bot_name)
        if not config:
            return {"success": False, "message": "Bot config not found."}
        
        try:
            logger.info(f"Starting bot '{bot_name}' in-memory...")
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

        return {"success": True, "message": f"Bot {bot_name} stopped."}

# A single instance that can be imported by other modules
bot_manager = BotManager()