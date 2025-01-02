import pinecone
from sentence_transformers import SentenceTransformer
from insert_data import preprocess_text  # 전처리 함수 재사용

# Pinecone 초기화
pinecone.init(api_key="your-pinecone-api-key", environment="us-east-1")
index_name = "user_state_analysis"
index = pinecone.Index(index_name)

# 모델 로드
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

# 검색 및 판별 함수
def find_user_state(query, threshold=0.7):
    """
    입력 텍스트를 전처리하고 Pinecone 데이터와 비교하여
    가장 유사한 구분을 반환하거나 비슷한 데이터가 없음을 반환합니다.
    """
    # 입력 텍스트 전처리 및 벡터화
    query = preprocess_text(query)
    query_vector = model.encode(query).tolist()

    # Pinecone에서 유사한 데이터 검색
    results = index.query(query_vector, top_k=5, include_metadata=True)

    # 가장 유사한 결과 분석
    for result in results['matches']:
        if result['score'] >= threshold:
            return f"유사한 구분: {result['metadata']['구분']}, 유사도: {result['score']:.2f}"

    return "비슷한 구분을 찾을 수 없습니다."

# 테스트
if __name__ == "__main__":
    query_text = "내가 요즘 배우자와의 관계가 좋지 않아 고민 중입니다."
    print(find_user_state(query_text))
