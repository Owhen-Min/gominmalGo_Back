import sqlite3

# SQLite 데이터베이스 경로
db_path = './db/ssafy_ai.db'

# 데이터베이스 연결
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 데이터 확인 쿼리
cursor.execute('SELECT * FROM emotion_corpus LIMIT 5')
for row in cursor.fetchall():
    print(row)

# 연결 종료
conn.close()
