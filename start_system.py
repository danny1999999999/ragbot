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

def wait_for_service(url, timeout=30, service_name="服務"):
    """等待服務啟動"""
    if not REQUESTS_AVAILABLE:
        print(f"⚠️ 無法進行 {service_name} 健康檢查，等待固定時間...")
        time.sleep(5)
        return True
    
    print(f"⏳ 等待 {service_name} 啟動...", end="", flush=True)
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f" ✅ {service_name} 健康檢查通過")
                return True
        except requests.exceptions.RequestException:
            pass
        except Exception as e:
            print(f"\n健康檢查異常: {e}")
        
        time.sleep(1)
        print(".", end="", flush=True)
    
    print(f"\n⚠️ {service_name} 健康檢查超時")
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
    """主函數"""
    print("🚀 啟動聊天機器人系統...")
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
            
            # 等待服務啟動
            if wait_for_service("http://localhost:9002/health", service_name="向量API服務"):
                print("✅ 向量API服務就緒")
            else:
                print("⚠️ 向量API服務可能啟動異常，但繼續啟動其他服務")
        else:
            print("❌ 向量API服務啟動失敗")
            return 1
        
        # 給向量API一些額外時間完全初始化
        print("⏳ 等待向量系統完全初始化...")
        time.sleep(3)
        
        # 2. 🆕 啟動 Gateway 服務 (最重要的修復)
        print("\n🌐 Step 2: 啟動 Gateway 服務...")
        gateway_process = start_service(
            "gateway_server.py",
            "Gateway服務",
            8000,
            "gateway.log"
        )
        
        if gateway_process:
            services.append(("Gateway服務", gateway_process))
            
            # 等待服務啟動
            if wait_for_service("http://localhost:8000/health", service_name="Gateway服務"):
                print("✅ Gateway 服務就緒")
            else:
                print("⚠️ Gateway 服務可能啟動異常")
        else:
            print("❌ Gateway 服務啟動失敗，機器人訪問將會有問題")
            # 不直接返回，讓用戶決定是否繼續
        
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
            
            # 等待服務啟動
            if wait_for_service("http://localhost:9001/health", service_name="管理器服務"):
                print("✅ 管理器服務就緒")
            else:
                print("⚠️ 管理器服務可能啟動異常")
        else:
            print("⚠️ 管理器服務啟動失敗")
        
        # 4. 檢查機器人配置
        print("\n🤖 Step 4: 檢查機器人配置...")
        check_bot_configs()
        
        # 顯示啟動完成信息
        print("\n🎉 系統核心服務啟動完成！")
        print("=" * 50)
        
        # 顯示訪問地址
        if gateway_process and gateway_process.poll() is None:
            print("🌐 Gateway 路由: http://localhost:8000/")
            print("   機器人訪問格式: http://localhost:8000/<機器人名稱>/")
        if manager_process and manager_process.poll() is None:
            print("📊 管理界面: http://localhost:9001/manager")
            print("🔐 登錄頁面: http://localhost:9001/login")
        if vector_process and vector_process.poll() is None:
            print("🔍 向量API文檔: http://localhost:9002/docs")
            print("🔍 向量API健康檢查: http://localhost:9002/health")
        
        print("\n📋 服務運行狀態:")
        for name, process in services:
            status = "✅ 運行中" if process.poll() is None else "❌ 已停止"
            print(f"   {name}: {status}")
        
        print("\n🔧 端口分配:")
        print("   8000 - Gateway (機器人路由)")
        print("   9001 - 管理器 (bot_service_manager)")
        print("   9002 - 向量API (vector_api_service)")
        print("   8003+ - 機器人實例 (通過管理界面啟動)")
        
        # 監控循環
        monitor_services(services)
        
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