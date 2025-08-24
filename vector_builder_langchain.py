#!/usr/bin/env python3
"""
📦 主入口檔案 vector_builder_langchain.py - Refactored

🎯 設計原則：
   - 直接依賴 4A+4B 結構 (vector_operations.py + management_api.py)
   - 移除複雜的回退和智能導入機制，簡化依賴鏈
"""

import os

# =============================================================================
# 🎯 直接導入 - 依賴清晰的 4A+4B 結構
# =============================================================================

try:
    # 4A: 向量操作核心層 (已包含所有配置和文本處理工具)
    from vector_operations import (
        VectorOperationsCore,
        SYSTEM_CONFIG, TOKEN_LIMITS, SMART_TEXT_CONFIG, PERFORMANCE_CONFIG,
        SUPPORTED_EXTENSIONS, FileInfo, TextAnalysis, ChunkInfo, SearchResult,
        detect_railway_environment, check_openai_api_key,
        SmartTextAnalyzer, OptimizedTextSplitter, ChineseTextNormalizer, 
        AdaptiveBatchProcessor, PGVECTOR_AVAILABLE, OPENAI_EMBEDDINGS_AVAILABLE,
        DOCX2TXT_AVAILABLE, EPUB_AVAILABLE, OPENCC_AVAILABLE
    )
    
    # 4B: 管理API層
    from management_api import OptimizedVectorSystem
    
    print("✅ 4A+4B 結構導入成功")
    USING_4AB_STRUCTURE = True

except ImportError as e:
    print(f"❌ 關鍵模塊導入失敗: {e}")
    print("   請確保 vector_operations.py 和 management_api.py 文件存在且無誤")
    # 拋出一個更直接的錯誤，而不是自定義的模糊錯誤
    raise ImportError(
        "無法加載核心向量系統模塊。請檢查 vector_operations.py 和 management_api.py 的完整性。"
    ) from e

# =============================================================================
# 🎯 主程式函數 - 保持完全向後兼容
# =============================================================================

def main():
    """主程式 - Refactored for 4A+4B structure"""
    print("🇨🇳 === 優化版 LangChain 中文向量系統 (Refactored) === 🇨🇳")
    
    print("📦 當前架構: 4A+4B 模塊化結構")
    print("   📁 4A: vector_operations.py (核心層)")
    print("   📁 4B: management_api.py (管理層)")
    
    # 系統狀態檢查
    print(f"🤖 OpenAI Embeddings: {'✅ 已啟用' if OPENAI_EMBEDDINGS_AVAILABLE else '❌ 未啟用'}")
    print(f"🐘 PostgreSQL: {'✅ 已啟用' if detect_railway_environment() else '❌ 未啟用'}")  
    print(f"📊 PGVector: {'✅ 已啟用' if PGVECTOR_AVAILABLE else '❌ 未啟用'}")
    print(f"📄 DOCX 支援: {'✅ 已啟用' if DOCX2TXT_AVAILABLE else '❌ 未啟用'}")
    print(f"📚 EPUB 支援: {'✅ 已啟用' if EPUB_AVAILABLE else '❌ 未啟用'}")
    print(f"🔤 繁簡轉換: {'✅ 已啟用' if OPENCC_AVAILABLE else '❌ 未啟用'}")
    
    try:
        system = OptimizedVectorSystem()
        
        print("\n" + "="*60)
        if hasattr(system, 'diagnose_system'):
            system.diagnose_system()
        print("="*60 + "\n")
        
        return system
        
    except Exception as e:
        print(f"\n❌ 系統初始化失敗: {e}")
        return None

# =============================================================================
# 🔒 向後兼容性導出 - 確保所有現有import語句正常工作
# =============================================================================

__all__ = [
    # 主要系統類別
    'OptimizedVectorSystem',
    
    # 核心配置
    'SYSTEM_CONFIG', 'TOKEN_LIMITS', 'SMART_TEXT_CONFIG', 'PERFORMANCE_CONFIG',
    'SUPPORTED_EXTENSIONS',
    
    # 資料類別
    'FileInfo', 'TextAnalysis', 'ChunkInfo', 'SearchResult',
    
    # 處理類別
    'SmartTextAnalyzer', 'OptimizedTextSplitter', 'ChineseTextNormalizer', 
    'AdaptiveBatchProcessor',
    
    # 工具函數
    'detect_railway_environment', 'check_openai_api_key',
    
    # 主程式
    'main'
]

if __name__ == "__main__":
    main()
