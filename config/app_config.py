import os
import logging

class AppConfig:
    def __init__(self):
        self.is_railway = bool(os.getenv('RAILWAY_PROJECT_ID'))
        self.base_port = int(os.getenv('PORT', 8000)) if self.is_railway else 8000
        self.environment = 'railway' if self.is_railway else 'local'
        
        # 設置簡單日誌
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f'配置初始化: {self.environment}')
    
    def get_vector_api_url(self):
        if self.is_railway:
            return f'http://localhost:{self.base_port}'
        else:
            return 'http://localhost:9002'
    
    def get_conversation_db_path(self, bot_name):
        if self.is_railway:
            return f'/tmp/{bot_name}_conversations.db'
        else:
            return f'{bot_name}_conversations.db'
    
    def get_service_port(self, service_name):
        ports = {
            'vector_api': 9002 if not self.is_railway else self.base_port,
            'gateway': 8000 if not self.is_railway else self.base_port + 1,
            'manager': 9001 if not self.is_railway else self.base_port + 2,
        }
        return ports.get(service_name, self.base_port)
    
    def print_config_summary(self):
        print('=' * 40)
        print('配置摘要')
        print('=' * 40)
        print(f'環境: {self.environment}')
        print(f'基礎端口: {self.base_port}')
        print(f'向量API: {self.get_vector_api_url()}')
        print('=' * 40)

app_config = AppConfig()