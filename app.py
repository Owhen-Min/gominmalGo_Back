# pip install fastapi uvicorn openai python-dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import json


load_dotenv()  # Load .env file if present
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


app = FastAPI()

assistant = OpenAI().beta.assistants.create(
    name="Counsel_bot",
    instructions='''
    As a professional counselor, you will evaluate the `user_input` and determine its relevance and the appropriate response. 

    - If the `user_input` is unrelated to typical counseling topics, return `type` as 0 and `context` as null.
    - If the `user_input` is relevant but lacks sufficient information for an assessment, especially when it is only statement of feeling, return `type` as 1 and provide a suggestion in Korean about what additional details would be helpful in the `context`.
    - If the `user_input` contains enough information for counseling, return `type` as 2 and include in `context` a response acknowledging the user's suffering, mentioning that others have had visited psychiatry, and suggesting a counseling session may help improve their mood: "~~한 배경에서 ~~한 상황을 맞닥뜨려서 어떤 감정을 느끼셨겠어요. 정말 힘들었겠어요. 이와 비슷한 사유로 상담을 방문한 사람들이 있어요. 한번 상담을 받아보면서 기분을 풀어보는건 어떨까요?"
    - If the `user_input` is relevant but too minor to warrant counseling, return `type` as 3 and provide a soothing response in the `context`.
    
    # Output Format
    Your output should be a JSON object structured as follows:
    - `type`: integer (0, 1, 2, or 3)
    - `context`: string or null
    ''',
    model="gpt-4o-mini",
)

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    message: str


@app.post("/assistant")
async def assistant_endpoint(req: MessageRequest):
    thread = await openai.beta.threads.create(
        messages=[{"role": "user", "content": req.message}]
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
