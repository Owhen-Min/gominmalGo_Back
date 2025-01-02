import sqlite3
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from collections import Counter
from openai import OpenAI
import os
import warnings

# Disable specific warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow logs
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Tokenizers parallelism warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")


def init_pinecone():
    """
    Pinecone 초기화
    """
    with open('./env/pinecone_key', 'r') as file:
        api_key = file.read().strip()

    pc = Pinecone(api_key=api_key)
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

def fetch_emotion_from_db(human_text):
    db_path = './db/ssafy_ai.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * 
        FROM emotion_corpus
        WHERE human_text = ?
        LIMIT 1
    ''', (human_text,))
    rows = cursor.fetchall()

    conn.close()
    return rows

def fetch_wellness_from_db(text):
    db_path = './db/ssafy_ai.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * 
        FROM wellness
        WHERE dialogue = ?
        LIMIT 1
    ''', (text,))
    rows = cursor.fetchall()

    conn.close()
    return rows

def split_into_sentences(input_text):
    """
    사용자 입력을 문장별로 나누는 함수.
    """
    with open('./env/open_key', 'r') as file:
        api_key = file.read().strip()
    client = OpenAI(api_key=api_key)
    prompt = """
    구어체로 이루어진 문단의 경우 일일이 코딩을 하여 문장으로 나누기 어렵다.
    다음 구어체 형태의 텍스트의 내용을 파악하여 충분한 크기의 1개 이상의 문어체 문장들로 재구성할 것.
    원본 텍스트의 내용을 변형하지 않아야 한다.
    출력형태는 파이썬 문법의 리스트형식으로 출력한다.
    출력형태는 다음과 같다.
    ['나의 고민은 과식을 하는 것이다.','기분이 풀릴 때까지 먹는다.']
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
    사용자 입력을 요약하여 5개의 문장으로 만드는 함수.
    """
    with open('./env/open_key', 'r') as file:
        api_key = file.read().strip()
    client = OpenAI(api_key=api_key)
    prompt = """
    다음 대화 내용을 하나의 증상을 나타내는 문어체 문장으로 변환하고, 
    같은 증상의 경우 같은 문장으로 통일해 주세요. 
    의사가 환자의 증상을 적듯이, 간단하고 명료하게 작성할 것.
    증상의 경우 파이썬 문법의 리스트 형식으로 제공된다.
    파이썬 문법의 리스트 형식으로 변환된 문장을 리스트에 담아 출력할 것.
    출력형태는 다음과 같다.

    
    ['감정조절이 어렵다.','의심이 많아졌다.']
    
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
                print('score:',score, 'text',text)
                for row in db_result:
                    print(row)

        print("\n[진행 상황] 입력 요약 시작...")
        summarized_sentences = summarize_input(accumulated_input)
        print(f"[결과] 요약된 문장 리스트: {summarized_sentences}")

        print("\n[진행 상황] 웰니스 검색 시작...")
        for sentence in eval(summarized_sentences):
            results = search_wellness(sentence)
            print('sentence',sentence)
            if results['matches']:
                for match in results["matches"][:3]:
                    score = match['score']
                    text = match['metadata']['text'].lstrip().rstrip()
                    db_result = fetch_wellness_from_db(text)
                    if db_result:
                        print("\n웰니스 검색 결과:")
                        print('score:',score, 'text',text)
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
