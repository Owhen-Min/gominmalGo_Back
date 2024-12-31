# pip install fastapi uvicorn openai python-dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import json

load_dotenv()  # Load .env file if present

openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    message: str


@app.post("/assistant")
async def assistant_endpoint(req: MessageRequest):
    assistant = await openai.beta.assistants.retrieve("asst_6NHvOWoaG3KHd4LxliJfvGXS")

    # Create a new thread with the user's message
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
    
    # 메시지 타입과 컨텍스트를 분리하여 처리
    message_json = json.loads(message_content.text.value)

    return message_json


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
