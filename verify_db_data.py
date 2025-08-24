import os
import psycopg2
from dotenv import load_dotenv

def list_all_tables():
    """
    Connects to the PostgreSQL database and lists all tables in the public schema.
    """
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        print("❌ 錯誤：DATABASE_URL 未在 .env 文件中設定。")
        return

    print(f"🔍 正在連線到資料庫...")
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        print("✅ 資料庫連線成功！")

        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """
        
        print("Executing query to list all tables in public schema...")
        cursor.execute(query)
        rows = cursor.fetchall()

        print(f"\n--- 📋 'public' schema 中的所有資料表 ---")
        if not rows:
            print("No tables found in the public schema.")
        else:
            for row in rows:
                print(f"  - {row[0]}")

    except Exception as e:
        print(f"❌ 執行資料庫查詢時發生錯誤: {e}")

    finally:
        if 'conn' in locals() and conn:
            cursor.close()
            conn.close()
            print("\n✅ 資料庫連線已關閉。")

if __name__ == "__main__":
    list_all_tables()