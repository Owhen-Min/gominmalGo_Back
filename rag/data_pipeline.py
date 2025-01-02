import sqlite3
import os
import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import re
from datetime import datetime

# ==================
# 1. 감정증상 데이터 생성
# ==================

def create_database():
    """
    감정증상 데이터 기반 SQLite 데이터베이스 생성 및 데이터 삽입
    """
    db_path = './db/ssafy_ai.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

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

    # 데이터 삽입
    json_file_path = './data/감성대화말뭉치(최종데이터)_Training.json'
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in tqdm(data, desc="Inserting Emotion Data"):
        cursor.execute('''
        INSERT INTO emotion_corpus (human_text, emotion_id, computer_response)
        VALUES (?, ?, ?)
        ''', (item["talk"]["content"]["HS01"], item["profile"]["emotion"]["emotion-id"], item["talk"]["content"]["SS01"]))

    conn.commit()
    conn.close()
    print("감정증상 데이터 기반 SQLite 데이터베이스 생성 완료!")

# ==================
# 2. 감정증상 Pinecone 모델 생성
# ==================

def init_pinecone():
    """
    Pinecone 초기화
    """
    with open('./env/pinecone_key', 'r') as file:
        api_key = file.read().strip()

    # Pinecone 클래스 인스턴스 생성
    pc = Pinecone(api_key=api_key)
    
    return pc

def create_emotion_model():
    """
    감정증상 데이터 기반 Pinecone 모델 생성
    """
    pc = init_pinecone()

    # Pinecone 인덱스 이름 생성
    index_name = "emotion-corpus-jhgan"
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    # 인덱스 존재 여부 확인 및 생성
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    # 기존 인덱스에 접근
    index = pc.Index(index_name)

    # 데이터 업로드
    json_file_path = './data/감성대화말뭉치(최종데이터)_Training.json'
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in tqdm(data, desc="Uploading Emotion Data"):
        embedding = model.encode(item["talk"]["content"]["HS01"]).tolist()
        index.upsert([("id_" + str(hash(item["talk"]["content"]["HS01"])), embedding, {"text": item["talk"]["content"]["HS01"]})])

    print(f"감정증상 Pinecone 모델 생성 완료: {index_name}")



# ==================
# 3. 웰니스 데이터 생성
# ==================

def create_wellness_database():
    """
    Wellness 데이터 기반 SQLite 데이터베이스 생성 및 데이터 삽입
    """
    db_path = './db/ssafy_ai.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 테이블 생성
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS wellness (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        dialogue TEXT NOT NULL
    )
    ''')

    # 데이터 삽입
    csv_file_path = './data/Wellness_final.csv'
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Wellness 데이터 파일을 찾을 수 없습니다: {csv_file_path}")

    data = pd.read_csv(csv_file_path)

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Inserting Wellness Data"):
        cursor.execute('''
        INSERT INTO wellness (category, dialogue)
        VALUES (?, ?)
        ''', (row['구분'], row['증상']))

    conn.commit()
    conn.close()
    print("Wellness 데이터 기반 SQLite 데이터베이스 생성 완료!")

# ==================
# 4. 웰니스 Pinecone 모델 생성
# ==================

def create_wellness_model():
    """
    Wellness 데이터 기반 Pinecone 모델 생성
    """
    # Pinecone 초기화
    with open('./env/pinecone_key', 'r') as file:
        api_key = file.read().strip()
    pc = Pinecone(api_key=api_key)

    # Pinecone 인덱스 이름 생성
    index_name = "wellness-corpus"
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    # 인덱스 존재 여부 확인 및 생성
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # 기존 인덱스 접근
    index = pc.Index(index_name)

    # 데이터 업로드
    csv_file_path = './data/Wellness_final.csv'
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Wellness 데이터 파일을 찾을 수 없습니다: {csv_file_path}")

    data = pd.read_csv(csv_file_path)

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Uploading Wellness Data"):
        embedding = model.encode(row['증상']).tolist()
        metadata = {"category": row['구분'], "text": row['증상']}
        index.upsert([(f"id_{_}", embedding, metadata)])

    print("Wellness Pinecone 모델 생성 완료!")
