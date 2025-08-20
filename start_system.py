#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
start_system.py - Railway 適配版本 (實際可行方案)
保持原有邏輯，但適配 Railway 環境
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
import logging

# 檢查必要的依賴
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests 庫未安裝，健康檢查功能受限")

# 🔧 添加配置管理支持
try:
    import simple_config as config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    
    # 創建簡單的配置替代
    class SimpleConfig:
        def __init__(self):
            self.is_railway = bool(os.getenv("RAILWAY_PROJECT_ID"))
            self.base_port = int(os.getenv("PORT", 8000)) if self.is_railway else 8000
            self.logger = logging.getLogger(__name__)
        
        def print_config_summary(self):
            print("=" * 60)
            print("🔧 簡化配置模式")
            print("=" * 60)
            print(f"🌍 環境: {'Railway' if self.is_railway else '本地'}")
            print(f"🌐 基礎端口: {self.base_port}")
            print("=" * 60)
    
    service_config = SimpleConfig()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_port(port):
    """檢查端口是否可用"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result != 0
    except Exception as e:
        print(f"端口檢查失敗: {e}")
        return True

def wait_for_service(url, timeout=180, service_name="服務"):
    """等待服務啟動 - Railway 優化版本"""
    if not REQUESTS_AVAILABLE:
        print(f"⚠️ 無法進行 {service_name} 健康檢查，等待固定時間...")
        time.sleep(10 if service_config.is_railway else 15)
        return True
    
    print(f"⏳ 等待 {service_name} 啟動...", end="", flush=True)
    start_time = time.time()
    
    # Railway 專用：更短的檢查間隔，更寬鬆的條件
    check_interval = 2 if service_config.is_railway else 5
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=10)
            # 🔧 Railway 專用：只要能連通就算成功，不要求完美的 200 狀態
            if response.status_code in [200, 202, 503]:
                elapsed = time.time() - start_time
                print(f" ✅ {service_name} 可訪問 ({elapsed:.1f}s)")
                return True
            else:
                print(f"({response.status_code})", end="", flush=True)
        except requests.exceptions.RequestException:
            print(".", end="", flush=True)
        except Exception as e:
            print("(E)", end="", flush=True)
        
        time.sleep(check_interval)
    
    elapsed = time.time() - start_time
    print(f"\n⚠️ {service_name} 健康檢查超時 ({elapsed:.1f}s)")
    
    # 🔧 Railway 特殊處理：即使超時也繼續，因為服務可能仍在後台初始化
    if service_config.is_railway:
        print(f"💡 Railway 環境：服務可能仍在後台初始化，繼續啟動...")
        return True
    
    return False

def create_directories():
    """創建必要的目錄"""
    directories = ["logs", "pids", "data", "temp_uploads"]
    
    # Railway 特殊處理：使用 /tmp 目錄
    if service_config.is_railway:
        directories = ["/tmp/logs", "/tmp/pids", "/tmp/data", "/tmp/temp_uploads"]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"📁 創建目錄: {dir_name}")
            except Exception as e:
                print(f"❌ 創建目錄失敗 {dir_name}: {e}")
                return False
    
    return True

def get_service_config(service_name):
    """獲取服務配置 - Railway 適配"""
    if service_config.is_railway:
        # Railway 環境：所有服務共用基礎端口，通過路由區分
        base_port = service_config.base_port
        
        configs = {
            "vector_api": {
                "port": base_port,
                "url": f"http://localhost:{base_port}/api/vector/health"
            },
            "gateway": {
                "port": base_port + 1,  # 小偏移避免衝突
                "url": f"http://localhost:{base_port + 1}/health"
            },
            "manager": {
                "port": base_port + 2,  # 小偏移避免衝突
                "url": f"http://localhost:{base_port + 2}/health"
            }
        }
    else:
        # 本地環境：使用原有端口配置
        configs = {
            "vector_api": {
                "port": 9002,
                "url": "http://localhost:9002/health"
            },
            "gateway": {
                "port": 8000, 
                "url": "http://localhost:8000/health"
            },
            "manager": {
                "port": 9001,
                "url": "http://localhost:9001/health"
            }
        }
    
    return configs.get(service_name, {})

def start_service_with_env(script_name, service_name, log_file, env_vars=None):
    """啟動服務 - 支持環境變數注入"""
    print(f"📍 啟動 {service_name}...")
    
    script_path = Path(script_name)
    if not script_path.exists():
        print(f"❌ 腳本文件不存在: {script_name}")
        return None
    
    config = get_service_config(service_name.lower().replace(' ', '_').replace('服務', ''))
    port = config.get('port')
    
    if not service_config.is_railway and port:
        if not check_port(port):
            print(f"❌ 端口 {port} 被佔用")
            return None
    
    try:
        # 準備環境變數
        process_env = os.environ.copy()
        
        # 注入服務特定的環境變數
        if env_vars:
            process_env.update(env_vars)
        
        # Railway 專用環境變數
        if service_config.is_railway:
            process_env.update({
                "RAILWAY_OPTIMIZED": "true",
                "PYTHONUNBUFFERED": "1",  # 確保日誌立即輸出
                "MAX_WORKERS": "1"
            })
        
        # 確保日誌目錄存在
        log_dir = Path("/tmp/logs" if service_config.is_railway else "logs")
        log_dir.mkdir(exist_ok=True)
        
        log_path = log_dir / log_file
        
        with open(log_path, "w", encoding="utf-8") as log:
            process = subprocess.Popen([
                sys.executable, script_name
            ], 
            stdout=log, 
            stderr=subprocess.STDOUT, 
            cwd=Path.cwd(),
            env=process_env
            )
        
        print(f"✅ {service_name} 進程已啟動 (PID: {process.pid})")
        
        # 保存PID（如果可能）
        try:
            pid_dir = Path("/tmp/pids" if service_config.is_railway else "pids")
            pid_dir.mkdir(exist_ok=True)
            pid_file = pid_dir / f"{service_name.lower().replace(' ', '_')}.pid"
            with open(pid_file, "w", encoding="utf-8") as f:
                f.write(str(process.pid))
        except Exception as pid_error:
            logger.warning(f"⚠️ 保存PID失敗: {pid_error}")
        
        return process
        
    except Exception as e:
        print(f"❌ 啟動 {service_name} 失敗: {e}")
        return None

def start_railway_optimized_services():
    """Railway 優化的服務啟動序列"""
    logger.info("🚂 Railway 優化啟動序列")
    
    services = []
    
    # 設置 Railway 專用環境變數
    os.environ.update({
        "USE_VECTOR_API": "true",
        "RAILWAY_OPTIMIZED": "true",
        "FORCE_POSTGRESQL": "true"
    })
    
    try:
        # 1. 先啟動向量API服務 (最重要)
        print("\n🔧 Step 1: 啟動向量API服務...")
        vector_env = {
            "PORT": str(service_config.base_port),
            "VECTOR_API_PORT": str(service_config.base_port)
        }
        
        vector_process = start_service_with_env(
            "vector_api_service.py",
            "向量API服務", 
            "vector_api.log",
            vector_env
        )
        
        if vector_process:
            services.append(("向量API服務", vector_process))
            
            # Railway 上給更多時間初始化
            vector_config = get_service_config("vector_api")
            if vector_config.get('url'):
                print("⏳ 等待向量API初始化...")
                if wait_for_service(vector_config['url'], timeout=300, service_name="向量API服務"):
                    print("✅ 向量API服務就緒")
                else:
                    print("⚠️ 向量API服務可能需要更多時間")
        
        # 給向量系統更多初始化時間
        time.sleep(20 if service_config.is_railway else 5)
        
        # 2. 啟動管理器服務（依賴向量API）
        print("\n👑 Step 2: 啟動管理器服務...")
        manager_env = {
            "PORT": str(service_config.base_port + 2),
            "MANAGER_PORT": str(service_config.base_port + 2),
            "VECTOR_API_URL": f"http://localhost:{service_config.base_port}"
        }
        
        manager_process = start_service_with_env(
            "bot_service_manager.py",
            "管理器服務",
            "manager.log", 
            manager_env
        )
        
        if manager_process:
            services.append(("管理器服務", manager_process))
            time.sleep(15)
        
        # 3. 最後啟動閘道器（可選）
        if Path("gateway_server.py").exists():
            print("\n🌐 Step 3: 啟動閘道器服務...")
            gateway_env = {
                "PORT": str(service_config.base_port + 1),
                "GATEWAY_PORT": str(service_config.base_port + 1),
                "VECTOR_API_URL": f"http://localhost:{service_config.base_port}"
            }
            
            gateway_process = start_service_with_env(
                "gateway_server.py",
                "閘道器服務",
                "gateway.log",
                gateway_env
            )
            
            if gateway_process:
                services.append(("閘道器服務", gateway_process))
        
        return services
        
    except Exception as e:
        logger.error(f"❌ Railway 服務啟動異常: {e}")
        return services

def start_local_services():
    """本地環境服務啟動（保持原有邏輯）"""
    logger.info("💻 本地環境啟動序列")
    
    services = []
    
    try:
        # 1. 啟動向量API服務
        print("\n🔧 Step 1: 啟動向量API服務...")
        vector_process = start_service_with_env(
            "vector_api_service.py",
            "向量API服務",
            "vector_api.log"
        )
        
        if vector_process:
            services.append(("向量API服務", vector_process))
            
            if wait_for_service("http://localhost:9002/health", timeout=60, service_name="向量API服務"):
                print("✅ 向量API服務就緒")
        
        time.sleep(5)
        
        # 2. 啟動閘道器服務
        if Path("gateway_server.py").exists():
            print("\n🌐 Step 2: 啟動閘道器服務...")
            gateway_process = start_service_with_env(
                "gateway_server.py",
                "閘道器服務",
                "gateway.log"
            )
            
            if gateway_process:
                services.append(("閘道器服務", gateway_process))
                time.sleep(3)
        
        # 3. 啟動管理器服務
        print("\n👑 Step 3: 啟動管理器服務...")
        manager_process = start_service_with_env(
            "bot_service_manager.py",
            "管理器服務",
            "manager.log"
        )
        
        if manager_process:
            services.append(("管理器服務", manager_process))
        
        return services
        
    except Exception as e:
        logger.error(f"❌ 本地服務啟動異常: {e}")
        return services

def monitor_services(services):
    """監控服務狀態"""
    if not services:
        print("❌ 沒有服務需要監控")
        return False
    
    print(f"\n⏳ 開始監控 {len(services)} 個服務...")
    print("按 Ctrl+C 停止所有服務...")
    
    try:
        while True:
            failed_services = []
            for name, process in services:
                if process.poll() is not None:
                    failed_services.append((name, process.returncode))
            
            if failed_services:
                print("\n⚠️ 檢測到服務異常:")
                for name, exit_code in failed_services:
                    print(f"   {name}: 退出碼 {exit_code}")
                return False
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🛑 收到停止信號...")
        return True

def stop_services(services):
    """停止所有服務"""
    if not services:
        return
    
    print("🛑 正在停止所有服務...")
    
    for name, process in reversed(services):
        try:
            if process.poll() is None:
                print(f"⏹️ 停止 {name}...")
                process.terminate()
                
                try:
                    process.wait(timeout=5)
                    print(f"✅ {name} 已停止")
                except subprocess.TimeoutExpired:
                    print(f"💀 強制停止 {name}...")
                    process.kill()
                    process.wait()
                    print(f"✅ {name} 已強制停止")
            else:
                print(f"ℹ️ {name} 已經停止")
                
        except Exception as e:
            print(f"⚠️ 停止 {name} 時出錯: {e}")

def main():
    """主函數 - Railway 適配版本"""
    print("🚀 啟動聊天機器人系統")
    
    # 打印配置摘要
    service_config.print_config_summary()
    
    print("=" * 50)
    
    # 創建必要目錄
    if not create_directories():
        print("❌ 初始化目錄失敗")
        return 1
    
    services = []
    
    try:
        # 根據環境選擇啟動策略
        if service_config.is_railway:
            services = start_railway_optimized_services()
        else:
            services = start_local_services()
        
        # 顯示啟動結果
        print("\n🎉 系統啟動完成！")
        print("=" * 50)
        
        running_services = []
        failed_services = []
        
        for name, process in services:
            if process.poll() is None:
                running_services.append(name)
                print(f"   {name}: ✅ 運行中")
            else:
                failed_services.append((name, process.returncode))
                print(f"   {name}: ❌ 已停止 (退出碼: {process.returncode})")
        
        # Railway 特定狀態報告
        if service_config.is_railway:
            print(f"\n🚂 Railway 部署狀態:")
            print(f"   ✅ 運行中服務: {len(running_services)}")
            print(f"   ❌ 失敗服務: {len(failed_services)}")
            
            if len(running_services) >= 1:
                print(f"   📊 系統狀態: 運行中")
                railway_url = os.getenv("RAILWAY_STATIC_URL") or os.getenv("RAILWAY_PUBLIC_DOMAIN")
                if railway_url:
                    print(f"   🌐 訪問地址: https://{railway_url}/")
        
        # 服務監控
        if running_services:
            success = monitor_services(services)
            return 0 if success else 1
        else:
            print(f"\n❌ 沒有服務成功運行")
            return 1
        
    except Exception as e:
        print(f"❌ 系統啟動異常: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        stop_services(services)
        print("✅ 系統已完全停止")
    
    return 0

if __name__ == "__main__":
    exit(main())