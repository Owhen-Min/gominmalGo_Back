# pip install fastapi uvicorn openai python-dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import json
from rag.main import split_into_sentences, search_emotion, search_wellness, fetch_emotion_from_db, fetch_wellness_from_db, summarize_input
from collections import Counter


load_dotenv()  # Load .env file if present
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


app = FastAPI()

assistant = OpenAI().beta.assistants.create(
    name="Counsel_bot",
    instructions='''
    As a professional counselor, you will evaluate both the `user_input` and the provided `RAG Analysis` to determine the appropriate response. 

    The RAG Analysis contains:
    - emotion_analysis: Emotional context from similar cases
    - wellness_categories: Frequency of wellness categories related to the input

    Response Guidelines:
    - If the `user_input` is unrelated to typical counseling topics, return `type` as 0 and `context` as null.
    - If the `user_input` is relevant but lacks sufficient information, return `type` as 1 and provide a suggestion in Korean using insights from the emotion_analysis.
    - If the `user_input` contains enough information and wellness_categories shows recurring patterns, return `type` as 2 with a response that incorporates both emotional understanding and specific counseling suggestions.
    - If the `user_input` is relevant but minor, return `type` as 3 and provide a soothing response using similar cases from the emotion_analysis.

    # Output Format
    {
        "type": integer (0, 1, 2, or 3),
        "context": string or null
    }
    ''',
    model="gpt-4o-mini",
)

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gominmalgo.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)


class MessageRequest(BaseModel):
    message: str


@app.post("/assistant/")
async def assistant_endpoint(req: MessageRequest):
    # RAG 처리
    user_input = req.message
    
    # 1. 문장 분리
    sentences = eval(split_into_sentences(user_input))
    
    # 2. 감정 분석
    emotion_results = []
    for sentence in sentences:
        results = search_emotion(sentence)
        if results['matches']:
            match = results["matches"][0]
            text = match['metadata']['text'].strip()
            db_result = fetch_emotion_from_db(text)
            if db_result:
                emotion_results.append(db_result)
    
    # 3. 웰니스 분석
    wellness_categories = []
    summarized = eval(summarize_input(user_input))
    for sentence in summarized:
        results = search_wellness(sentence)
        if results['matches']:
            match = results["matches"][0]
            text = match['metadata']['text'].strip()
            db_result = fetch_wellness_from_db(text)
            if db_result:
                category = db_result[0][1]
                wellness_categories.append('/'.join(category.split('/')[:2]))
    
    # RAG 결과를 포함한 컨텍스트 생성
    rag_context = str({
        "emotion_analysis": emotion_results,
        "wellness_categories": dict(Counter(wellness_categories))
    })
    
    # Assistant에 RAG 결과와 함께 요청 전송
    thread = await openai.beta.threads.create(
        messages=[{
            "role": "user", 
            "content": f"User Input: {req.message}\nRAG Analysis: {rag_context}"
        }]
    )
    
    # Create a run and poll until completion using the helper method
    run = await openai.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant.id
    )

    # Get messages for this specific run
    messages = list(
        await openai.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
    )
    message_content = messages[0][1][0].content[0]
    
    message_json = json.loads(message_content.text.value)

    return message_json


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
