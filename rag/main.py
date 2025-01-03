import mysql.connector
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from collections import Counter
from openai import OpenAI
import os
import warnings

# Disable specific warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
openai_key = os.getenv("OPENAI_API_KEY")


def connect_mysql():
    """
    MySQL 데이터베이스 연결
    """
    return mysql.connector.connect(
        host="localhost",        # MySQL 서버 주소
        user="your_username",    # 사용자 이름
        password="your_password", # 비밀번호
        database="ssafy_ai"      # 데이터베이스 이름
    )


def fetch_emotion_from_db(human_text):
    """
    MySQL에서 감정 데이터를 검색
    """
    conn = connect_mysql()
    cursor = conn.cursor()

    query = '''
        SELECT * 
        FROM emotion_corpus
        WHERE human_text = %s
        LIMIT 1
    '''
    cursor.execute(query, (human_text,))
    rows = cursor.fetchall()

    conn.close()
    return rows


def fetch_wellness_from_db(text):
    """
    MySQL에서 웰니스 데이터를 검색
    """
    conn = connect_mysql()
    cursor = conn.cursor()

    query = '''
        SELECT * 
        FROM wellness
        WHERE dialogue = %s
        LIMIT 1
    '''
    cursor.execute(query, (text,))
    rows = cursor.fetchall()

    conn.close()
    return rows


def init_pinecone():
    """
    Pinecone 초기화
    """
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_key)
    return pc


def search_emotion(query):
    pc = init_pinecone()
    index_name = "emotion-corpus-jhgan"
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    index = pc.Index(index_name)

    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=1, include_metadata=True)
    return results


def search_wellness(query):
    pc = init_pinecone()
    index_name = "wellness-corpus"
    index = pc.Index(index_name)
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=1, include_metadata=True)
    return results


def split_into_sentences(input_text):
    """
    사용자 입력을 문장별로 나누는 함수
    """
    client = OpenAI(api_key=openai_key)
    prompt = """
    구어체로 이루어진 문단의 경우 문장으로 나누어 파이썬 리스트 형식으로 반환하세요.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": input_text
            }
        ],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content


def summarize_input(input_text):
    """
    사용자 입력을 요약하여 5개의 문장으로 만드는 함수
    """
    client = OpenAI(api_key=openai_key)
    prompt = """
    대화 내용을 간단히 요약하여 증상 문장 리스트로 반환하세요.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": input_text
            }
        ],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content


def main():
    accumulated_input = ""  # 누적 입력 저장
    wellness_categories = []
    while True:
        user_input = input("입력: ").strip()
        accumulated_input += f" {user_input}"  # 입력을 누적

        print("\n[진행 상황] 입력을 문장별로 나누는 중...")
        sentences = split_into_sentences(accumulated_input)
        print(f"[결과] 문장 리스트: {sentences}")

        print("\n[진행 상황] 감정 검색 시작...")
        for sentence in eval(sentences):
            results = search_emotion(sentence)
            if results['matches']:
                match = results["matches"][0]
                score = match['score']
                text = match['metadata']['text'].lstrip().rstrip()
                db_result = fetch_emotion_from_db(text)
                print("\n감정 검색 결과:")
                print('score:', score, 'text:', text)
                for row in db_result:
                    print(row)

        print("\n[진행 상황] 입력 요약 시작...")
        summarized_sentences = summarize_input(accumulated_input)
        print(f"[결과] 요약된 문장 리스트: {summarized_sentences}")

        print("\n[진행 상황] 웰니스 검색 시작...")
        for sentence in eval(summarized_sentences):
            results = search_wellness(sentence)
            print('sentence:', sentence)
            if results['matches']:
                for match in results["matches"][:3]:
                    score = match['score']
                    text = match['metadata']['text'].lstrip().rstrip()
                    db_result = fetch_wellness_from_db(text)
                    if db_result:
                        print("\n웰니스 검색 결과:")
                        print('score:', score, 'text:', text)
                        print(db_result)
                        category = db_result[0][1]
                        new_cate = '/'.join(category.split('/')[:2])
                        wellness_categories.append(new_cate)  # category 추가

        print("\n[진행 상황] 카테고리 빈도수 계산 중...")
        category_count = Counter(wellness_categories)
        print(category_count.items())
        for category, count in category_count.items():
            if count >= 3:
                print(f"\n[알림] 카테고리 '{category}'의 빈도가 {count}로 3개 이상입니다. 프로그램을 종료합니다.")
                return

        print("\n[알림] 조건을 만족하지 않아 다시 입력을 기다립니다.")


if __name__ == "__main__":
    main()
