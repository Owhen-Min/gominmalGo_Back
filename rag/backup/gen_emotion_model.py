from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
import json


with open('./env/pinecone_key', 'r') as file:
    key = file.read().strip()  # 파일에서 읽은 내용의 공백 제거
pc = Pinecone(api_key=key)

index_name = f"emotion_corpus_{datetime.now().strftime('%m%d_%H%M')}"
pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

index = pc.Index(index_name)
model = SentenceTransformer('all-MiniLM-L6-v2')

with open('./data/감성대화말뭉치(최종데이터)_Training.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    
# Pinecone에 데이터 저장
for item in tqdm(data):
    human_text = item["talk"]["content"]["HS01"]
    
    # 텍스트를 벡터로 변환
    embedding = model.encode(human_text).tolist()
    
    # Pinecone에 업로드
    index.upsert([("id_" + str(hash(human_text)), embedding, {"text": human_text})])

# 모델 저장 (인덱스 이름과 동일하게)
model.save(f'./model/{index_name}')

print(f"모델과 인덱스 이름: {index_name}")
