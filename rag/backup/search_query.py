from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

# Pinecone API 키 로드
with open('./env/pinecone_key', 'r') as file:
    key = file.read().strip()

# Pinecone 초기화
pc = Pinecone(api_key=key)

# ./model/에서 가장 최신 모델 디렉토리 찾기
model_dir = "./model"
latest_model = max(
    [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))],
    key=lambda x: x  # 디렉토리 이름 기준 정렬 (문자열 비교)
)

# Pinecone 인덱스 이름 및 모델 경로
index_name = latest_model
index = pc.Index(index_name)

# 저장된 모델 로드
model = SentenceTransformer(os.path.join(model_dir, latest_model))

# 검색 쿼리 처리
query = "내가 실수를 해서 너무 미안해."
query_vector = model.encode(query).tolist()

# Pinecone에서 유사한 벡터 검색
results = index.query(vector=query_vector, top_k=1, include_metadata=True)

# 결과 출력
for result in results["matches"]:
    print(f"Emotion ID: {result['id']}")
    print(f"Original Text: {result['metadata']['text']}")
'''
    {
    "matches": [
        {
        "id": "12345",
        "score": 0.987,
        "metadata": {
            "text": "내가 실수를 해서 너무 미안해."
        }
        }
    ],
    "namespace": "default"
    }
'''