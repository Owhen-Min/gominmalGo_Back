import pinecone
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import os

# 데이터 전처리 함수
def preprocess_text(text):
    """
    텍스트 전처리:
    1. 불필요한 키워드 제거
    2. 구어체 처리
    3. 소문자 변환 (영어 포함 시)
    """
    remove_keywords = ['아내', '남편', '친구']  # 필요에 따라 수정 가능
    for keyword in remove_keywords:
        text = text.replace(keyword, '')

    text = re.sub(r'\s+', ' ', text)  # 중복 공백 제거
    text = re.sub(r'[?!]', '.', text)  # 감탄사 변환
    text = text.lower()  # 영어 포함 시 소문자 변환
    return text.strip()

# Pinecone 초기화
pinecone.init(api_key="your-pinecone-api-key", environment="us-east-1")

# 데이터베이스 연결
index_name = "user_state_analysis"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384, metric="cosine")
index = pinecone.Index(index_name)

# 모델 로드
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

# CSV 데이터 로드
data_path = './data/tmp.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

data = pd.read_csv(data_path)

# 데이터 전처리 및 Pinecone 업로드
for idx, row in data.iterrows():
    preprocessed_text = preprocess_text(row['담화'])  # 전처리
    embedding = model.encode(preprocessed_text).tolist()  # 벡터화
    metadata = {"구분": row['구분'], "원문": row['담화']}
    index.upsert([(str(idx), embedding, metadata)])

print("Data inserted into Pinecone successfully!")
