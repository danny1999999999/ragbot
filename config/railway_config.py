# ============================================
# config/railway_config.py - 修正版本 ✅
# ============================================

#!/usr/bin/env python3
"""
Railway 環境配置管理器 - 修正版本
"""

import os
import logging
from typing import Optional, List, Dict

class RailwayConfig:
    """Railway 環境配置管理器"""
    
    def __init__(self):
        # 環境檢測
        self.is_railway = bool(os.getenv("RAILWAY_PROJECT_ID"))
        self.is_production = os.getenv("RAILWAY_ENVIRONMENT_NAME") == "production"  # 🔧 修正：使用正確的環境變數
        self.railway_env = os.getenv("RAILWAY_ENVIRONMENT_NAME", "development")    # 🔧 修正：預設值
        
        # Railway 專案信息
        self.project_id = os.getenv("RAILWAY_PROJECT_ID")
        self.region = os.getenv("RAILWAY_REGION", "unknown")
        
        # 設置日誌
        self._setup_logging()
        
        # 🆕 新增：配置驗證
        self._validate_basic_config()
    
    def _setup_logging(self):
        """設置日誌配置"""
        log_level = self.get_log_level()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        if self.is_railway:
            self.logger.info(f"🚂 Railway 環境初始化 - {self.railway_env}")
        else:
            self.logger.info("💻 本地開發環境初始化")
    
    def _validate_basic_config(self):
        """🆕 驗證基本配置"""
        if self.is_railway:
            required_vars = ["DATABASE_URL", "OPENAI_API_KEY"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                self.logger.warning(f"⚠️ Railway 環境缺少變數: {missing_vars}")
        
        self.logger.info("✅ 基本配置檢查完成")
    
    def get_vector_api_url(self) -> str:
        """獲取向量API的正確URL"""
        
        if self.is_railway:
            # 🔧 修正：更好的 Railway 服務發現邏輯
            
            # 方法1: 使用 Railway 提供的服務變數
            service_domain = os.getenv("VECTOR_API_RAILWAY_PRIVATE_DOMAIN")
            service_port = os.getenv("VECTOR_API_PORT")
            
            if service_domain and service_port:
                return f"http://{service_domain}:{service_port}"
            
            # 方法2: 使用公共域名 (如果設置了)
            public_domain = os.getenv("VECTOR_API_RAILWAY_PUBLIC_DOMAIN")
            if public_domain:
                return f"https://{public_domain}"
            
            # 方法3: 使用 Railway 模板變數格式
            vector_service_url = os.getenv("VECTOR_API_URL")  # 🆕 新增：直接使用配置的 URL
            if vector_service_url:
                return vector_service_url
            
            # 方法4: 最後備用 - Railway 內部網路
            service_name = os.getenv("VECTOR_SERVICE_NAME", "vector-api")
            return f"http://{service_name}.railway.internal:8080"
            
        else:
            # 本地開發環境
            return os.getenv("VECTOR_API_LOCAL_URL", "http://localhost:9002")
    
    def get_database_url(self) -> str:
        """獲取資料庫連接字符串"""
        
        if self.is_railway:
            # Railway 上使用 PostgreSQL
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("Railway 環境缺少 DATABASE_URL")
            return db_url
        else:
            # 本地開發環境
            local_db = os.getenv("DATABASE_URL")
            if local_db:
                return local_db
            
            # 🔧 修正：更靈活的本地資料庫配置
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            user = os.getenv("DB_USER", "postgres")
            password = os.getenv("DB_PASSWORD", "password")
            database = os.getenv("DB_NAME", "chatbot_system")
            
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    def get_service_port(self) -> int:
        """獲取服務端口"""
        
        if self.is_railway:
            # Railway 自動分配端口
            return int(os.getenv("PORT", 8080))
        else:
            # 🔧 修正：本地端口應該可配置
            return int(os.getenv("LOCAL_PORT", os.getenv("PORT", 9002)))
    
    def get_cors_origins(self) -> List[str]:  # 🔧 修正：返回類型註解
        """獲取 CORS 允許的來源"""
        
        if self.is_railway and self.is_production:
            # 生產環境：只允許指定域名
            origins = []
            
            # 🔧 修正：更靈活的域名配置
            railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN")
            if railway_domain:
                origins.append(f"https://{railway_domain}")
            
            # 🆕 新增：支持多個自定義域名
            custom_origins = os.getenv("CORS_ORIGINS", "").split(",")
            for origin in custom_origins:
                origin = origin.strip()
                if origin:
                    origins.append(origin)
            
            # 🆕 新增：如果沒有設置任何域名，允許所有 (開發時)
            return origins if origins else ["*"]
        else:
            # 開發環境：允許所有來源
            return ["*"]
    
    def get_log_level(self) -> str:
        """獲取日誌級別"""
        
        if self.is_railway:
            return os.getenv("LOG_LEVEL", "INFO")
        else:
            return os.getenv("LOG_LEVEL", "DEBUG")
    
    # 🆕 新增方法
    def get_gateway_url(self) -> str:
        """獲取 Gateway URL"""
        if self.is_railway:
            gateway_domain = os.getenv("GATEWAY_RAILWAY_PUBLIC_DOMAIN")
            if gateway_domain:
                return f"https://{gateway_domain}"
            return f"https://{os.getenv('RAILWAY_PUBLIC_DOMAIN', 'unknown')}"
        else:
            return os.getenv("GATEWAY_LOCAL_URL", "http://localhost:8000")
    
    def get_manager_url(self) -> str:
        """獲取管理器 URL"""
        if self.is_railway:
            manager_domain = os.getenv("MANAGER_RAILWAY_PRIVATE_DOMAIN")
            manager_port = os.getenv("MANAGER_PORT")
            if manager_domain and manager_port:
                return f"http://{manager_domain}:{manager_port}"
            return "http://manager.railway.internal:8080"
        else:
            return os.getenv("MANAGER_LOCAL_URL", "http://localhost:9001")
    
    def should_use_postgresql(self) -> bool:
        """是否使用 PostgreSQL"""
        if os.getenv("FORCE_CHROMA_ONLY", "false").lower() == "true":
            return False
        return self.is_railway or os.getenv("USE_POSTGRESQL", "false").lower() == "true"
    
    def get_openai_api_key(self) -> str:
        """獲取 OpenAI API Key"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("缺少 OPENAI_API_KEY 環境變數")
        return api_key
    
    def get_gateway_token(self) -> str:
        """獲取 Gateway Token"""
        token = os.getenv("GATEWAY_TOKEN")
        if not token:
            self.logger.warning("⚠️ 未設置 GATEWAY_TOKEN")
            return "default_insecure_token"
        return token
    
    # 🆕 新增：統一的服務 URL 獲取方法
    def get_service_urls(self) -> Dict[str, str]:
        """獲取所有服務的URL"""
        return {
            "vector_api": self.get_vector_api_url(),
            "gateway": self.get_gateway_url(),
            "manager": self.get_manager_url()
        }
    
    # 🆕 新增：配置摘要
    def get_environment_info(self) -> Dict:
        """獲取環境資訊摘要"""
        return {
            "is_railway": self.is_railway,
            "is_production": self.is_production,
            "environment_name": self.railway_env,
            "project_id": self.project_id,
            "region": self.region,
            "service_port": self.get_service_port(),
            "log_level": self.get_log_level(),
            "database_type": "postgresql" if self.should_use_postgresql() else "chroma",
            "service_urls": self.get_service_urls()
        }
    
    def print_config_summary(self):
        """打印配置摘要"""
        info = self.get_environment_info()
        
        print("=" * 60)
        print("🔧 Railway 配置摘要")
        print("=" * 60)
        print(f"🌍 環境: {'Railway' if info['is_railway'] else '本地'} ({info['environment_name']})")
        print(f"🌐 端口: {info['service_port']}")
        print(f"🗄️ 資料庫: {info['database_type']}")
        print(f"📊 日誌級別: {info['log_level']}")
        print("🔗 服務 URL:")
        for service, url in info['service_urls'].items():
            print(f"   {service}: {url}")
        print("=" * 60)

# 🔧 修正：移到類外面的函數需要正確處理
def setup_railway_environment() -> bool:
    """設置 Railway 環境變數驗證"""
    
    # 檢查必要的環境變數
    required_vars = [
        "OPENAI_API_KEY",
        "DATABASE_URL", 
        "GATEWAY_TOKEN"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ 缺少必要環境變數: {missing_vars}")
        return False
    
    # 檢查 Railway 特定變數
    if not os.getenv("RAILWAY_PROJECT_ID"):
        print("⚠️ 不在 Railway 環境中")
        return False
    
    print("✅ Railway 環境檢查通過")
    return True

# 創建全局配置實例
config = RailwayConfig()

# 🔧 修正：將原本的功能移到模塊級別函數
def get_service_urls() -> Dict[str, str]:
    """獲取所有服務的URL - 向後兼容函數"""
    return config.get_service_urls()

# ============================================
# 主程式 - 用於測試配置
# ============================================

def main():
    """主函數 - 測試配置"""
    
    if config.is_railway:
        print("🚂 Railway 部署模式")
        if not setup_railway_environment():
            return 1
    else:
        print("💻 本地開發模式")
    
    # 打印配置摘要
    config.print_config_summary()
    
    # 測試各種配置
    try:
        print(f"\n🔑 OpenAI API Key: {config.get_openai_api_key()[:8]}...")
        print(f"🎫 Gateway Token: {config.get_gateway_token()[:8]}...")
        print(f"🗄️ 資料庫 URL: {config.get_database_url()[:30]}...")
    except Exception as e:
        print(f"⚠️ 配置測試發現問題: {e}")
    
    return 0

if __name__ == "__main__":
    exit(main())