import sqlite3

db_path = './db/ssafy_ai.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 사용자 입력을 포함 검색으로 변환
search_text = f"%{input('검색할 텍스트를 입력하세요: ').strip()}%"

# 포함 검색 쿼리 실행
cursor.execute('''
        SELECT * 
        FROM wellness
        WHERE dialogue like ?
        limit 1
''', (search_text,))

# 결과 출력
rows = cursor.fetchall()
print(rows)

# 데이터베이스 연결 종료
conn.commit()
conn.close()
