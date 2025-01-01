import os
import sqlite3
import json
from tqdm import tqdm

# JSON 파일 경로
json_file_path = './data/감성대화말뭉치(최종데이터)_Validation.json'

# SQLite 데이터베이스 경로
db_path = './db/ssafy_ai.db'

# 디렉토리 확인 및 생성
db_dir = os.path.dirname(db_path)
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

# SQLite 데이터베이스 연결
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS emotion_corpus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    human_text TEXT NOT NULL,
    emotion_id TEXT NOT NULL,
    computer_response TEXT NOT NULL
)
''')

# JSON 파일 읽기
try:
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Error: JSON file not found at {json_file_path}")
    conn.close()
    exit()

# 데이터 삽입
try:
    for item in tqdm(data, desc="Inserting data"):
        human_text = item["talk"]["content"]["HS01"]
        emotion_id = item["profile"]["emotion"]["emotion-id"]
        computer_response = item["talk"]["content"]["SS01"]

        cursor.execute('''
            INSERT INTO emotion_corpus (human_text, emotion_id, computer_response)
            VALUES (?, ?, ?)
        ''', (human_text, emotion_id, computer_response))
    
    # 변경 사항 커밋
    conn.commit()
    print(f"Data successfully inserted into {db_path}")

except Exception as e:
    print(f"Error during data insertion: {e}")
    conn.rollback()

# 데이터베이스 연결 종료
finally:
    conn.close()
