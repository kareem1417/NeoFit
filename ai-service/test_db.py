import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        database="ringside",
        user="postgres",
        password="rootpassword",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    print("✅ Connection Successful & pgvector is ready!")
    cur.close()
    conn.close()
except Exception as e:
    print(f"❌ Error: {e}")
