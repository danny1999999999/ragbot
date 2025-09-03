# bot_config_manager.py - 修正版
import psycopg
import json
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 資料庫表結構
CREATE_BOT_CONFIGS_TABLE = """
CREATE TABLE IF NOT EXISTS bot_configs (
    id SERIAL PRIMARY KEY,
    bot_name VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(200),
    port INTEGER NOT NULL,
    system_role TEXT,
    temperature REAL DEFAULT 0.7,
    max_tokens INTEGER DEFAULT 2000,
    dynamic_recommendations_enabled BOOLEAN DEFAULT FALSE,
    dynamic_recommendations_count INTEGER DEFAULT 0,
    cite_sources_enabled BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    config_version INTEGER DEFAULT 1,
    additional_config JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_bot_configs_bot_name ON bot_configs(bot_name);
CREATE INDEX IF NOT EXISTS idx_bot_configs_active ON bot_configs(is_active);
"""

class BotConfigManager:
    """機器人設定資料庫管理器"""
    
    def __init__(self, database_url: str):
        if not database_url:
            raise ValueError("DATABASE_URL 不能為空")
        self.database_url = database_url
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """確保資料表存在"""
        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    # 分別執行每個 SQL 語句
                    statements = CREATE_BOT_CONFIGS_TABLE.split(';')
                    for statement in statements:
                        statement = statement.strip()
                        if statement:
                            cur.execute(statement)
                    conn.commit()
                    logger.info("機器人設定表已就緒")
        except Exception as e:
            logger.error(f"建立設定表失敗: {e}")
            raise
    
    def create_bot_config(self, config_data: Dict) -> bool:
        """建立新的機器人設定"""
        # 必填欄位驗證
        required_fields = ['bot_name', 'port']
        for field in required_fields:
            if not config_data.get(field):
                raise ValueError(f"必填欄位 {field} 不能為空")
        
        # 端口範圍驗證
        port = config_data.get('port')
        if not isinstance(port, int) or port < 1025 or port > 65535:
            raise ValueError("端口必須在 1025-65535 範圍內")
        
        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    # ✅ 統一使用位置參數 %s
                    # 檢查端口是否已被使用
                    cur.execute("""
                        SELECT bot_name FROM bot_configs 
                        WHERE port = %s AND is_active = TRUE AND bot_name != %s
                    """, (port, config_data.get('bot_name')))
                    
                    if cur.fetchone():
                        raise ValueError(f"端口 {port} 已被其他機器人使用")
                    
                    # ✅ 修正：統一使用位置參數
                    cur.execute("""
                        INSERT INTO bot_configs (
                            bot_name, display_name, port, system_role, 
                            temperature, max_tokens, dynamic_recommendations_enabled,
                            dynamic_recommendations_count, cite_sources_enabled,
                            created_by, additional_config
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """, (
                        config_data.get('bot_name'),
                        config_data.get('display_name'),
                        int(config_data.get('port')),
                        config_data.get('system_role', ''),
                        float(config_data.get('temperature', 0.7)),
                        int(config_data.get('max_tokens', 2000)),
                        bool(config_data.get('dynamic_recommendations_enabled', False)),
                        int(config_data.get('dynamic_recommendations_count', 0)),
                        bool(config_data.get('cite_sources_enabled', False)),
                        config_data.get('created_by', 'system'),
                        json.dumps(config_data.get('additional_config', {}))
                    ))
                    conn.commit()
                    return True
        except psycopg.IntegrityError as e:
            if 'bot_name' in str(e):
                raise ValueError(f"機器人名稱 '{config_data.get('bot_name')}' 已存在")
            else:
                raise ValueError(f"資料完整性錯誤: {e}")
        except Exception as e:
            logger.error(f"建立設定失敗: {e}")
            raise
    
    def get_bot_config(self, bot_name: str) -> Optional[Dict]:
        """取得機器人設定"""
        if not bot_name:
            return None
            
        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT bot_name, display_name, port, system_role, temperature,
                               max_tokens, dynamic_recommendations_enabled, 
                               dynamic_recommendations_count, cite_sources_enabled,
                               created_by, created_at, updated_at, additional_config,
                               config_version
                        FROM bot_configs 
                        WHERE bot_name = %s AND is_active = TRUE
                    """, (bot_name,))
                    
                    row = cur.fetchone()
                    if not row:
                        return None
                    
                    # ✅ 修正：安全的資料類型轉換
                    config = {
                        'bot_name': row[0],
                        'display_name': row[1],
                        'port': int(row[2]) if row[2] is not None else 8000,
                        'system_role': row[3] or '',
                        'temperature': float(row[4]) if row[4] is not None else 0.7,
                        'max_tokens': int(row[5]) if row[5] is not None else 2000,
                        'dynamic_recommendations_enabled': bool(row[6]) if row[6] is not None else False,
                        'dynamic_recommendations_count': int(row[7]) if row[7] is not None else 0,
                        'cite_sources_enabled': bool(row[8]) if row[8] is not None else False,
                        'created_by': row[9] or 'system',
                        'created_at': row[10].isoformat() if row[10] else None,
                        'updated_at': row[11].isoformat() if row[11] else None,
                        'config_version': int(row[13]) if row[13] is not None else 1
                    }
                    
                    # 處理額外設定
                    if row[12]:  # additional_config
                        try:
                            additional = row[12] if isinstance(row[12], dict) else json.loads(row[12])
                            if isinstance(additional, dict):
                                config.update(additional)
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"解析額外設定失敗: {e}")
                    
                    return config
                    
        except Exception as e:
            logger.error(f"取得設定失敗: {e}")
            return None
    
    def update_bot_config(self, bot_name: str, updates: Dict) -> bool:
        """更新機器人設定"""
        if not bot_name or not updates:
            return False
        
        # 端口衝突檢查
        if 'port' in updates:
            port = updates['port']
            try:
                port = int(port)
                if port < 1025 or port > 65535:
                    raise ValueError("端口必須在 1025-65535 範圍內")
                updates['port'] = port
            except (ValueError, TypeError):
                raise ValueError("端口必須是有效的整數")
        
        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    # 端口衝突檢查
                    if 'port' in updates:
                        cur.execute("""
                            SELECT bot_name FROM bot_configs 
                            WHERE port = %s AND is_active = TRUE AND bot_name != %s
                        """, (updates['port'], bot_name))
                        
                        if cur.fetchone():
                            raise ValueError(f"端口 {updates['port']} 已被其他機器人使用")
                    
                    # ✅ 修正：建構動態 UPDATE 語句使用位置參數
                    set_clauses = []
                    params = []
                    
                    allowed_fields = [
                        'display_name', 'port', 'system_role', 'temperature', 
                        'max_tokens', 'dynamic_recommendations_enabled',
                        'dynamic_recommendations_count', 'cite_sources_enabled'
                    ]
                    
                    for field in allowed_fields:
                        if field in updates:
                            set_clauses.append(f"{field} = %s")
                            # ✅ 資料類型轉換
                            value = updates[field]
                            if field == 'temperature' and value is not None:
                                params.append(float(value))
                            elif field in ['port', 'max_tokens', 'dynamic_recommendations_count'] and value is not None:
                                params.append(int(value))
                            elif field in ['dynamic_recommendations_enabled', 'cite_sources_enabled']:
                                params.append(bool(value))
                            else:
                                params.append(value)
                    
                    # 處理額外設定
                    if 'additional_config' in updates:
                        set_clauses.append("additional_config = %s")
                        params.append(json.dumps(updates['additional_config']))
                    
                    if not set_clauses:
                        return True  # 沒有要更新的欄位
                    
                    # 版本號自動遞增和更新時間
                    set_clauses.extend(["updated_at = CURRENT_TIMESTAMP", "config_version = config_version + 1"])
                    params.append(bot_name)  # WHERE 條件的參數
                    
                    sql = f"""
                        UPDATE bot_configs 
                        SET {', '.join(set_clauses)}
                        WHERE bot_name = %s AND is_active = TRUE
                    """
                    
                    cur.execute(sql, params)
                    conn.commit()
                    return cur.rowcount > 0
                    
        except Exception as e:
            logger.error(f"更新設定失敗: {e}")
            raise
    
    def get_all_bot_configs(self) -> List[Dict]:
        """取得所有機器人設定"""
        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT bot_name, display_name, port, system_role, 
                               created_by, created_at, updated_at, is_active,
                               config_version
                        FROM bot_configs 
                        WHERE is_active = TRUE
                        ORDER BY created_at DESC
                    """)
                    
                    configs = []
                    for row in cur.fetchall():
                        config = {
                            'name': row[0],  # ✅ 保持與原有 API 相容
                            'bot_name': row[0],
                            'display_name': row[1],
                            'port': int(row[2]) if row[2] is not None else 8000,
                            'system_role': row[3] or '',
                            'created_by': row[4] or 'system',
                            'created_at': row[5].isoformat() if row[5] else None,
                            'updated_at': row[6].isoformat() if row[6] else None,
                            'is_active': bool(row[7]),
                            'config_version': int(row[8]) if row[8] is not None else 1
                        }
                        configs.append(config)
                    
                    return configs
                    
        except Exception as e:
            logger.error(f"取得所有設定失敗: {e}")
            return []
    
    def delete_bot_config(self, bot_name: str) -> bool:
        """刪除機器人設定（軟刪除）"""
        if not bot_name:
            return False
            
        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE bot_configs 
                        SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                        WHERE bot_name = %s AND is_active = TRUE
                    """, (bot_name,))
                    conn.commit()
                    return cur.rowcount > 0
                    
        except Exception as e:
            logger.error(f"刪除設定失敗: {e}")
            return False

    def migrate_from_json_files(self, json_dir: str) -> Dict:
        """從 JSON 檔案遷移設定到資料庫"""
        results = {
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        json_path = Path(json_dir)
        if not json_path.exists():
            return results
        
        for json_file in json_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # ✅ 改善錯誤處理
                try:
                    if self.create_bot_config(config_data):
                        results['success'] += 1
                        logger.info(f"遷移成功: {json_file.stem}")
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"創建失敗: {json_file.stem}")
                except ValueError as ve:
                    results['failed'] += 1
                    results['errors'].append(f"{json_file.stem}: {str(ve)}")
                    
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"{json_file.stem}: {str(e)}")
        
        return results

class DatabaseBotManager:
    """資料庫版本的機器人管理器 - 修正循環導入"""
    
    def __init__(self, database_url: str):
        self.config_manager = BotConfigManager(database_url)
        self._global_bot_instances = None  # ✅ 延遲導入
    
    def _get_global_bot_instances(self):
        """✅ 延遲導入避免循環依賴"""
        if self._global_bot_instances is None:
            try:
                # 動態導入，避免循環導入
                import sys
                if 'gateway_server' in sys.modules:
                    gateway_module = sys.modules['gateway_server']
                    self._global_bot_instances = getattr(gateway_module, 'global_bot_instances', {})
                else:
                    self._global_bot_instances = {}
            except Exception as e:
                logger.warning(f"無法獲取運行狀態: {e}")
                self._global_bot_instances = {}
        return self._global_bot_instances
    
    def get_all_bots(self) -> List[Dict]:
        """取得所有機器人（含運行狀態）"""
        configs = self.config_manager.get_all_bot_configs()
        global_instances = self._get_global_bot_instances()
        
        for config in configs:
            bot_name = config['name']
            config['status'] = 'running' if bot_name in global_instances else 'stopped'
        
        return configs
    
    def get_bot_config(self, bot_name: str) -> Optional[Dict]:
        """取得機器人設定"""
        return self.config_manager.get_bot_config(bot_name)
    
    def create_bot(self, config_data: Dict) -> Dict:
        """建立新機器人"""
        try:
            if self.config_manager.create_bot_config(config_data):
                return {
                    "success": True, 
                    "message": f"機器人 {config_data['bot_name']} 建立成功"
                }
            else:
                return {
                    "success": False, 
                    "message": "機器人建立失敗"
                }
        except ValueError as e:
            return {
                "success": False, 
                "message": str(e)
            }
        except Exception as e:
            logger.error(f"建立機器人時發生未預期錯誤: {e}")
            return {
                "success": False, 
                "message": "系統錯誤，請稍後再試"
            }
    
    def update_bot_config(self, bot_name: str, updates: Dict) -> Dict:
        """更新機器人設定"""
        try:
            if self.config_manager.update_bot_config(bot_name, updates):
                return {
                    "success": True, 
                    "message": "設定更新成功"
                }
            else:
                return {
                    "success": False, 
                    "message": "找不到指定的機器人或無更新內容"
                }
        except ValueError as e:
            return {
                "success": False, 
                "message": str(e)
            }
        except Exception as e:
            logger.error(f"更新機器人設定時發生錯誤: {e}")
            return {
                "success": False, 
                "message": "設定更新失敗"
            }
    
    def delete_bot(self, bot_name: str) -> Dict:
        """刪除機器人"""
        try:
            if self.config_manager.delete_bot_config(bot_name):
                return {
                    "success": True, 
                    "message": f"機器人 {bot_name} 已刪除"
                }
            else:
                return {
                    "success": False, 
                    "message": "找不到指定的機器人"
                }
        except Exception as e:
            logger.error(f"刪除機器人時發生錯誤: {e}")
            return {
                "success": False, 
                "message": "刪除失敗"
            }

def initialize_database_bot_configs():
    """初始化資料庫機器人設定系統"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL 環境變數未設置")
    
    try:
        # ✅ 先測試資料庫連接
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        
        config_manager = BotConfigManager(database_url)
        
        # 檢查是否需要從 JSON 遷移
        json_dir = Path(__file__).resolve().parent / "bot_configs"
        if json_dir.exists() and any(json_dir.glob("*.json")):
            logger.info("發現 JSON 設定檔，開始遷移...")
            results = config_manager.migrate_from_json_files(str(json_dir))
            logger.info(f"遷移結果: 成功 {results['success']}, 失敗 {results['failed']}")
            
            if results['errors']:
                logger.warning("遷移錯誤:")
                for error in results['errors']:
                    logger.warning(f"  • {error}")
        
        return DatabaseBotManager(database_url)
        
    except Exception as e:
        logger.error(f"初始化資料庫機器人設定系統失敗: {e}")
        raise