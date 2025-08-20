# ============================================
# config/__init__.py
# ============================================

"""
配置模塊
提供統一的環境配置管理
"""

from .railway_config import config, RailwayConfig, setup_railway_environment, get_service_urls

# 導出主要的配置對象和類
__all__ = [
    'config',                    # 全局配置實例 ⭐ 最常用
    'RailwayConfig',            # 配置類 (如果需要創建自定義實例)
    'setup_railway_environment', # 環境驗證函數
    'get_service_urls'          # 服務 URL 獲取函數 (向後兼容)
]

# 版本信息
__version__ = "1.0.0"

# 🔧 可選：添加配置驗證
def validate_config():
    """驗證配置完整性"""
    try:
        # 測試基本配置
        config.get_service_port()
        config.get_vector_api_url()
        return True
    except Exception as e:
        print(f"❌ 配置驗證失敗: {e}")
        return False