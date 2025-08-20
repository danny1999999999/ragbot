
import os
import json
import subprocess
import sys
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, IO, List, Optional
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Project-level Imports ---
# Assuming these are in the same directory or accessible via sys.path
from auth_middleware import User

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables & Paths ---
ROOT_DIR = Path(__file__).parent
BOT_CONFIGS_DIR = ROOT_DIR / "bot_configs"
BOT_INSTANCE_SCRIPT = ROOT_DIR / "chatbot_instance.py"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories exist
for d in [BOT_CONFIGS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

# --- In-memory State for Subprocesses ---
global_bot_processes: Dict[str, subprocess.Popen] = {}
global_bot_log_files: Dict[str, IO[str]] = {}

class BotManager:
    """A non-web class to manage chatbot processes and configurations."""

    def __init__(self):
        self.gateway_url = os.getenv("GATEWAY_URL", "http://127.0.0.1:8000")
        self.gateway_admin_token = os.getenv("GATEWAY_ADMIN_TOKEN", "")
        logger.info("✅ BotManager (logic class) initialized.")

    def get_all_bots(self) -> List[Dict]:
        """Gets a list of all available bots and their status."""
        bots = []
        for config_file in sorted(BOT_CONFIGS_DIR.glob("*.json")):
            bot_name = config_file.stem
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)

                process = global_bot_processes.get(bot_name)
                status = "running" if process and process.poll() is None else "stopped"
                display_name = config.get("display_name") or bot_name
                system_role_preview = config.get("system_role", "")[:100] + "..."

                bots.append({
                    "name": bot_name,
                    "display_name": display_name,
                    "port": config.get("port"),
                    "status": status,
                    "system_role": system_role_preview,
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

    def save_bot_config(self, bot_name: str, data: dict, current_user: User) -> bool:
        config_path = BOT_CONFIGS_DIR / f"{bot_name}.json"
        if not config_path.exists():
            return False
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Update only the fields that are passed in the data
        for key, value in data.items():
            config[key] = value

        config["updated_by"] = current_user.username
        config["updated_at"] = datetime.now().isoformat()

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        return True

    async def start_bot(self, bot_name: str, current_user: User) -> Dict:
        if bot_name in global_bot_processes and global_bot_processes[bot_name].poll() is None:
            return {"success": False, "message": "Bot is already running."}

        config = self.get_bot_config(bot_name)
        if not config:
            return {"success": False, "message": "Bot config not found."}
        
        port = config.get("port")
        if not port:
            return {"success": False, "message": "Port not specified in config."}

        command = [sys.executable, str(BOT_INSTANCE_SCRIPT), "--bot-name", bot_name]
        log_path = LOGS_DIR / f"{bot_name}.log"
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)

        child_env = os.environ.copy()
        if os.getenv("DATABASE_URL"):
            child_env["DATABASE_URL"] = os.getenv("DATABASE_URL")

        process = subprocess.Popen(command, stdout=log_file, stderr=log_file, cwd=ROOT_DIR, env=child_env)

        global_bot_processes[bot_name] = process
        global_bot_log_files[bot_name] = log_file

        # The registration logic remains, as the gateway will now host the registration endpoint.
        registration_success = False
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(
                        f"{self.gateway_url}/_gateway/register",
                        headers={"X-Admin-Token": self.gateway_admin_token},
                        json={"bot": bot_name, "port": port}
                    )
                registration_success = True
                logger.info(f"Gateway registration successful for {bot_name} (attempt {attempt+1})")
                break
            except Exception as e:
                logger.warning(f"Gateway registration failed for {bot_name} (attempt {attempt+1}): {e}")
                await asyncio.sleep(2)

        if not registration_success:
            logger.error(f"Final Gateway registration failed for {bot_name}.")

        logger.info(f"Bot {bot_name} started by {current_user.username}.")
        return {"success": True, "message": f"Bot {bot_name} started."}

    def stop_bot(self, bot_name: str, current_user: User) -> Dict:
        process = global_bot_processes.get(bot_name)
        if not process or process.poll() is not None:
            return {"success": False, "message": "Bot not running."}

        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        finally:
            if bot_name in global_bot_processes:
                del global_bot_processes[bot_name]
            log_file = global_bot_log_files.pop(bot_name, None)
            if log_file:
                log_file.close()

        logger.info(f"Bot {bot_name} stopped by {current_user.username}.")
        return {"success": True, "message": f"Bot {bot_name} stopped."}

    def delete_bot(self, bot_name: str, current_user: User) -> Dict:
        self.stop_bot(bot_name, current_user) # Stop if running
        config_path = BOT_CONFIGS_DIR / f"{bot_name}.json"
        if config_path.exists():
            config_path.unlink()
        # Note: Deleting data directories can be risky, handled separately if needed.
        logger.info(f"Bot {bot_name} config deleted by {current_user.username}.")
        return {"success": True, "message": f"Bot {bot_name} deleted."}

# A single instance that can be imported by other modules
bot_manager = BotManager()
