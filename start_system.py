#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
start_system.py - 聊天機器人系統啟動腳本 (驗證修正版)
修正了語法和邏輯問題
"""

import os
import sys
import time
import json  # 🔧 移到頂層導入
import subprocess
from pathlib import Path

# 檢查必要的依賴
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests 庫未安裝，健康檢查功能受限")

def check_port(port):
    """檢查端口是否可用"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result != 0  # 返回True表示端口可用
    except Exception as e:
        print(f"端口檢查失敗: {e}")
        return True  # 假設可用

def wait_for_service(url, timeout=180, service_name="服務"):  # 🔧 大幅增加超時時間
    """等待服務啟動 - Railway 專用版本"""
    if not REQUESTS_AVAILABLE:
        print(f"⚠️ 無法進行 {service_name} 健康檢查，等待固定時間...")
        time.sleep(15)  # 增加等待時間
        return True
    
    print(f"⏳ 等待 {service_name} 啟動 (最多 {timeout}s)...", end="", flush=True)
    start_time = time.time()
    
    # Railway 特定的檢查邏輯
    check_interval = 5  # 每5秒檢查一次
    last_status_time = start_time
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=20)  # 增加請求超時
            if response.status_code == 200:
                elapsed = time.time() - start_time
                print(f" ✅ {service_name} 健康檢查通過 ({elapsed:.1f}s)")
                return True
            else:
                print(f"({response.status_code})", end="", flush=True)
        except requests.exceptions.RequestException:
            print(".", end="", flush=True)
        except Exception as e:
            print(f"(E)", end="", flush=True)
        
        # 每30秒顯示一次狀態
        current_time = time.time()
        if current_time - last_status_time >= 30:
            elapsed = current_time - start_time
            remaining = timeout - elapsed
            print(f"\n   ⏳ {service_name} 仍在啟動中... ({elapsed:.0f}s/{timeout}s, 剩餘 {remaining:.0f}s)", end="", flush=True)
            last_status_time = current_time
        
        time.sleep(check_interval)
    
    elapsed = time.time() - start_time
    print(f"\n⚠️ {service_name} 健康檢查超時 ({elapsed:.1f}s)")
    return False

def create_directories():
    """創建必要的目錄"""
    directories = ["logs", "pids", "data", "chroma_langchain_db", "bot_configs"]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            try:
                dir_path.mkdir(exist_ok=True)
                print(f"📁 創建目錄: {dir_name}")
            except Exception as e:
                print(f"❌ 創建目錄失敗 {dir_name}: {e}")
                return False
    
    return True

def start_service(script_name, service_name, port, log_file):
    """啟動單個服務"""
    print(f"📍 啟動 {service_name}...")
    
    # 檢查腳本是否存在
    script_path = Path(script_name)
    if not script_path.exists():
        print(f"❌ 腳本文件不存在: {script_name}")
        return None
    
    # 檢查端口
    if not check_port(port):
        print(f"❌ 端口 {port} 被佔用")
        return None
    
    try:
        # 確保 logs 目錄存在
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 啟動進程
        log_path = log_dir / log_file
        with open(log_path, "w", encoding="utf-8") as log:
            process = subprocess.Popen([
                sys.executable, script_name
            ], stdout=log, stderr=subprocess.STDOUT, cwd=Path.cwd())
        
        print(f"✅ {service_name} 進程已啟動 (PID: {process.pid})")
        
        # 保存PID
        pid_dir = Path("pids")
        pid_dir.mkdir(exist_ok=True)
        pid_file = pid_dir / f"{service_name.lower().replace(' ', '_').replace('服務', '_service')}.pid"
        with open(pid_file, "w", encoding="utf-8") as f:
            f.write(str(process.pid))
        
        return process
        
    except Exception as e:
        print(f"❌ 啟動 {service_name} 失敗: {e}")
        return None

def check_bot_configs():
    """檢查機器人配置並檢測端口衝突"""
    bot_configs_dir = Path("bot_configs")
    
    if not bot_configs_dir.exists():
        print("📭 未找到 bot_configs 目錄")
        return
    
    config_files = list(bot_configs_dir.glob("*.json"))
    if not config_files:
        print("📭 未找到機器人配置文件")
        return
    
    print(f"📄 找到 {len(config_files)} 個機器人配置")
    
    # 檢查是否有端口衝突
    system_ports = {8000, 9001, 9002}
    conflict_found = False
    
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            bot_name = config.get('bot_name', config_file.stem)
            bot_port = config.get('port')
            
            if bot_port in system_ports:
                print(f"⚠️ 機器人 {bot_name} 的端口 {bot_port} 與系統端口衝突")
                conflict_found = True
            else:
                print(f"   ✓ {bot_name}: 端口 {bot_port}")
                
        except Exception as e:
            print(f"⚠️ 檢查配置文件 {config_file} 失敗: {e}")
    
    if conflict_found:
        print("🚨 發現端口衝突，請修改機器人配置中的端口號")
    
    print("💡 請通過管理界面啟動具體的機器人實例")

def monitor_services(services):
    """監控服務狀態"""
    print("\n按 Ctrl+C 停止所有服務...")
    
    try:
        while True:
            # 檢查服務狀態
            failed_services = []
            for name, process in services:
                if process.poll() is not None:
                    failed_services.append((name, process.returncode))
            
            if failed_services:
                print("\n⚠️ 檢測到服務異常:")
                for name, exit_code in failed_services:
                    print(f"   {name}: 退出碼 {exit_code}")
                print("準備停止系統...")
                return False
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🛑 收到停止信號...")
        return True

def stop_services(services):
    """停止所有服務"""
    print("🛑 正在停止所有服務...")
    
    for name, process in reversed(services):  # 反向停止
        try:
            if process.poll() is None:  # 如果還在運行
                print(f"⏹️ 停止 {name}...")
                process.terminate()
                
                # 等待優雅關閉
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
    
    # 清理PID文件
    try:
        pids_dir = Path("pids")
        if pids_dir.exists():
            for pid_file in pids_dir.glob("*.pid"):
                pid_file.unlink()
            print("🧹 已清理PID文件")
    except Exception as e:
        print(f"⚠️ 清理PID文件失敗: {e}")

def main():
    """主函數 - Railway 快速修復版"""
    print("🚀 啟動聊天機器人系統 (Railway 優化版)...")
    
    # 檢測 Railway 環境
    is_railway = bool(os.getenv("RAILWAY_PROJECT_ID"))
    if is_railway:
        print("🚂 Railway 環境檢測到 - 使用擴展超時設置")
    
    print("=" * 50)
    
    # 創建必要目錄
    if not create_directories():
        print("❌ 初始化目錄失敗")
        return 1
    
    services = []
    
    try:
        # 1. 啟動向量API服務
        print("\n🔧 Step 1: 啟動向量API服務...")
        vector_process = start_service(
            "vector_api_service.py",
            "向量API服務",
            9002,
            "vector_api.log"
        )
        
        if vector_process:
            services.append(("向量API服務", vector_process))
            
            # Railway 環境使用更長的等待時間
            vector_timeout = 300 if is_railway else 60  # 5分鐘 vs 1分鐘
            
            print(f"⏳ 向量API服務初始化中... (這可能需要 {vector_timeout//60} 分鐘)")
            
            if wait_for_service("http://localhost:9002/health", timeout=vector_timeout, service_name="向量API服務"):
                print("✅ 向量API服務就緒")
            else:
                print("⚠️ 向量API服務健康檢查超時")
                print("💡 服務可能仍在後台初始化，繼續啟動其他服務...")
        else:
            print("❌ 向量API服務啟動失敗")
            print("💡 將繼續嘗試啟動其他服務...")
        
        # 給向量系統更多初始化時間
        if is_railway:
            print("⏳ Railway 環境：額外等待向量系統初始化...")
            time.sleep(30)  # Railway 需要更多時間
        else:
            time.sleep(5)
        
        # 2. 🆕 啟動 Gateway 服務
        print("\n🌐 Step 2: 啟動 Gateway 服務...")
        gateway_process = start_service(
            "gateway_server.py",
            "Gateway服務",
            8000,
            "gateway.log"
        )
        
        if gateway_process:
            services.append(("Gateway服務", gateway_process))
            
            # Gateway 服務通常啟動較快
            gateway_timeout = 120 if is_railway else 60
            
            if wait_for_service("http://localhost:8000/health", timeout=gateway_timeout, service_name="Gateway服務"):
                print("✅ Gateway 服務就緒")
            else:
                print("⚠️ Gateway 服務可能啟動異常")
        else:
            print("❌ Gateway 服務啟動失敗")
        
        # 3. 啟動管理器服務
        print("\n👑 Step 3: 啟動管理器服務...")
        manager_process = start_service(
            "bot_service_manager.py",
            "管理器服務",
            9001,
            "manager.log"
        )
        
        if manager_process:
            services.append(("管理器服務", manager_process))
            
            # 管理器服務依賴向量API，需要最長時間
            manager_timeout = 400 if is_railway else 90  # 約7分鐘 vs 1.5分鐘
            
            print(f"⏳ 管理器服務啟動中... (依賴向量API，最多等待 {manager_timeout//60} 分鐘)")
            
            if wait_for_service("http://localhost:9001/health", timeout=manager_timeout, service_name="管理器服務"):
                print("✅ 管理器服務就緒")
            else:
                print("⚠️ 管理器服務健康檢查超時")
                print("💡 這通常是因為向量API服務仍在初始化中")
        else:
            print("⚠️ 管理器服務啟動失敗")
        
        # 4. 檢查機器人配置
        print("\n🤖 Step 4: 檢查機器人配置...")
        check_bot_configs()
        
        # 顯示啟動完成信息
        print("\n🎉 系統核心服務啟動完成！")
        print("=" * 50)
        
        # 詳細的服務狀態
        running_services = []
        failed_services = []
        
        for name, process in services:
            if process.poll() is None:
                running_services.append(name)
                print(f"   {name}: ✅ 運行中")
            else:
                failed_services.append((name, process.returncode))
                print(f"   {name}: ❌ 已停止 (退出碼: {process.returncode})")
        
        # Railway 特定的狀態報告
        if is_railway:
            print(f"\n🚂 Railway 部署狀態:")
            print(f"   ✅ 運行中服務: {len(running_services)}")
            print(f"   ❌ 失敗服務: {len(failed_services)}")
            
            if len(running_services) >= 1:  # 至少有一個服務運行
                print(f"   📊 系統狀態: 部分功能可用")
                railway_url = os.getenv("RAILWAY_STATIC_URL") or os.getenv("RAILWAY_PUBLIC_DOMAIN")
                if railway_url:
                    print(f"   🌐 訪問地址: https://{railway_url}/")
            else:
                print(f"   📊 系統狀態: 所有服務均失敗")
        
        # 服務監控
        if running_services:
            print(f"\n⏳ 開始監控運行中的服務...")
            monitor_services(services)
        else:
            print(f"\n❌ 沒有服務成功運行，退出...")
            return 1
        
    except Exception as e:
        print(f"❌ 系統啟動異常: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 停止所有服務
        stop_services(services)
        print("✅ 系統已完全停止")
    
    return 0

if __name__ == "__main__":
    exit(main())