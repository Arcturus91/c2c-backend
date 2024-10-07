# main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import os
import dotenv
import uuid
from fastapi.security import APIKeyHeader
from fastapi import Security
import logging

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from constants import SYS_MSG_SEO_ARTICLE_GENERATOR, SYS_MSG_SEO_OUTLINER, MODELS

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# API key security
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

class Message(BaseModel):
    role: str
    content: str

class ChatSession(BaseModel):
    id: str
    messages: List[Message] = []
    model: str
    system_prompt: str  # Add this line

# This will act as our simple database
chat_sessions = {}


@app.post("/chat_sessions/SYS_MSG_SEO_OUTLINER", response_model=ChatSession)
async def create_chat_session_outliner(
    model: str = "anthropic/claude-3-5-sonnet-20240620",
    api_key: str = Depends(get_api_key)
):
    session_id = str(uuid.uuid4())
    chat_session = ChatSession(
        id=session_id,
        model=model,
        system_prompt=SYS_MSG_SEO_OUTLINER
    )
    chat_sessions[session_id] = chat_session
    return chat_session

@app.post("/chat_sessions/SYS_MSG_SEO_ARTICLE_GENERATOR", response_model=ChatSession)
async def create_chat_session_article_generator(
    model: str = "anthropic/claude-3-5-sonnet-20240620",
    api_key: str = Depends(get_api_key)
):
    session_id = str(uuid.uuid4())
    chat_session = ChatSession(
        id=session_id,
        model=model,
        system_prompt=SYS_MSG_SEO_ARTICLE_GENERATOR
    )
    chat_sessions[session_id] = chat_session
    return chat_session

@app.get("/chat_sessions/{session_id}", response_model=ChatSession)
async def get_chat_session(session_id: str, api_key: str = Depends(get_api_key)):
    session = chat_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session

@app.post("/chat_sessions/{session_id}/messages", response_model=Message)
async def add_message(session_id: str, message: Message, api_key: str = Depends(get_api_key)):
    session = chat_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    session.messages.append(message)
    return message

@app.post("/chat_sessions/{session_id}/ai_response")
async def get_ai_response(session_id: str, api_key: str = Depends(get_api_key)):
    logger.debug(f"Generating AI response for session: {session_id}")
    session = chat_sessions.get(session_id)
    if session is None:
        logger.warning(f"Chat session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Prepare messages for the AI model
    messages = [
        SystemMessage(content=session.system_prompt)  # Use the session's system prompt
    ]
    for msg in session.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))

    try:
        # Select the appropriate model
        model_provider, model_name = session.model.split('/')
        if model_provider == "openai":
            llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        elif model_provider == "anthropic":
            llm = ChatAnthropic(model=model_name, temperature=0.7)
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

        # Call the LLM
        response = llm(messages)

        # Extract the AI's response
        ai_message = Message(role="assistant", content=response.content)
        
        # Add the AI's response to the session
        session.messages.append(ai_message)

        return ai_message
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/chat_sessions/{session_id}/model")
async def update_model(session_id: str, model: str, api_key: str = Depends(get_api_key)):
    session = chat_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    if model not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid model")
    session.model = model
    return {"message": "Model updated successfully"}