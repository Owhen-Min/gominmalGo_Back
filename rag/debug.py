from data_pipeline import (
    create_database,
    init_pinecone,
    create_emotion_model,
    create_wellness_database,
    create_wellness_model,
)
from search_pipeline import search_emotion, search_wellness, fetch_emotion_from_db, fetch_wellness_from_db
import warnings
# 모든 경고 무시
warnings.filterwarnings('ignore')

print("==== 프로젝트 관리 시스템 ====")
print("1. 감정대화 데이터 생성")
print("2. 웰니스 데이터 생성")
print("3. 감성대화 검색")
print("4. 웰니스 검색")
print("5. 종료")
print("=============================")

while True:
    choice = input("수행할 작업 번호를 입력하세요 (1-5): ").strip()

    if choice == "1":
        print("\n[1] 감정대화 데이터 생성 시작...")
        create_database()  # SQLite 데이터 생성
        init_pinecone()  # Pinecone 초기화/
        create_emotion_model()  # 감성대화 Pinecone 모델 생성
        print("[1] 감정대화 데이터 생성 완료!")

    elif choice == "2":
        print("\n[2] 웰니스 데이터 생성 시작...")
        create_wellness_database()
        create_wellness_model()  # 웰니스 Pinecone 모델 생성
        print("[2] 웰니스 데이터 생성 완료!")

    elif choice == "3":
        print("\n[3] 감성대화 검색 시작...")
        query = input("검색할 텍스트를 입력하세요: ").strip()
        
        results = search_emotion(query)
        print("\n검색 결과:")
        match = results["matches"][0]
        score = match['score']
        text = match['metadata']['text'].lstrip().rstrip()
        print(f"ID: {match['id']}, 유사도: {score:.2f}, 텍스트: {text}")
    
        rows = fetch_emotion_from_db(text)
        if rows:
            print("\n=== 데이터베이스 검색 결과 ===")
            for row in rows:
                print(f"ID: {row[0]}, Human Text: {row[1]}, Emotion ID: {row[2]}, Computer Response: {row[3]}")
        else:
            print("\n[알림] 데이터베이스에서 일치하는 항목을 찾을 수 없습니다.")

    elif choice == "4":
        print("\n[4] 웰니스 검색 시작...")
        query = input("검색할 텍스트를 입력하세요: ").strip()
        results = search_wellness(query)
        print("\n검색 결과:")
        match = results["matches"][0]
        text = match['metadata']['text'].lstrip().rstrip()
        print(f"ID: {match['id']}, 유사도: {match['score']:.2f}, 증상:{match['metadata']['text']} ,구분: {match['metadata']['category']}")
        
        rows = fetch_wellness_from_db(text)
        if rows:
            print("\n=== 데이터베이스 검색 결과 ===")
            for row in rows:
                print(f"ID: {row[0]}, category: {row[1]}, dialogue: {row[2]}")
        else:
            print("\n[알림] 데이터베이스에서 일치하는 항목을 찾을 수 없습니다.")

        
    elif choice == "5":
        print("\n프로그램을 종료합니다.")
        break

    else:
        print("\n잘못된 입력입니다. 1-5 사이의 번호를 입력하세요.\n")

