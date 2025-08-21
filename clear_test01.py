# 1. 創建清空腳本

import os
import psycopg2

def clear_test01():
    try:
        database_url = os.getenv("DATABASE_URL")
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        # 查找表
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE '%langchain%';
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        total = 0
        
        for table in tables:
            cursor.execute(f"""
                DELETE FROM {table} 
                WHERE cmetadata::text LIKE '%test_01%';
            """)
            deleted = cursor.rowcount
            total += deleted
            print(f"表 {table}: 刪除 {deleted} 條")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"總計刪除: {total} 條記錄")
        return total > 0
        
    except Exception as e:
        print(f"清空失敗: {e}")
        return False

if __name__ == "__main__":
    success = clear_test01()
    print("✅ 清空完成!" if success else "❌ 清空失敗!")
EOF

# 2. 執行清空
python clear_test01.py