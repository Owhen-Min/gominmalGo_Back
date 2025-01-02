import sqlite3
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

def init_pinecone():
    """
    Pinecone 초기화
    """
    with open('./env/pinecone_key', 'r') as file:
        api_key = file.read().strip()

    # Pinecone 클래스 인스턴스 생성
    pc = Pinecone(api_key=api_key)
    return pc

def search_emotion(query):
    pc = init_pinecone()
    index_name = "emotion-corpus-jhgan"
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    index = pc.Index(index_name)  # Pinecone 객체 사용해 Index 초기화

    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=1, include_metadata=True)  # 키워드 인자로 수정
    return results

def search_wellness(query):
    pc = init_pinecone()
    index_name = "wellness-corpus"
    index = pc.Index(index_name)  # Pinecone 객체 사용해 Index 초기화
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=1, include_metadata=True)  # 키워드 인자로 수정
    return results

def fetch_emotion_from_db(human_text):
    """
    SQLite 데이터베이스에서 emotion_corpus 테이블을 조회하여
    human_text와 일치하는 행을 반환하는 함수.

    :param human_text: 검색할 텍스트
    :return: 일치하는 데이터베이스 행의 리스트
    """
    db_path = './db/ssafy_ai.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 데이터베이스 쿼리 실행
    cursor.execute('''
        SELECT * 
        FROM emotion_corpus
        WHERE human_text = ?
        limit 1
    ''', (human_text,))
    rows = cursor.fetchall()

    # 연결 종료
    conn.close()
    return rows

def fetch_wellness_from_db(text):
    """
    SQLite 데이터베이스에서 wellness 테이블을 조회하여
    dialogue와 일치하는 행을 반환하는 함수.

    :param dialogue: 검색할 텍스트
    :return: 일치하는 데이터베이스 행의 리스트
    """
    db_path = './db/ssafy_ai.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 데이터베이스 쿼리 실행
    cursor.execute('''
        SELECT * 
        FROM wellness
        WHERE dialogue = ?
        limit 1
    ''', (text,))
    rows = cursor.fetchall()

    # 연결 종료
    conn.close()
    return rows