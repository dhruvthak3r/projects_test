import os
import asyncio
import json
import uvicorn
from uuid import uuid4
from datetime import datetime
import redis.asyncio as redis
from starlette.middleware.sessions import SessionMiddleware
from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from models import QueryRequest
from Services import (
    load_faiss_index,
    load_scheme_data,
    query_faiss_index,
    retrieve_documents,
    generate_response,
    QueryResponse
)

load_dotenv()


# Session Manager
class SessionManager:
    def __init__(self, redis_client):
        self.redis_client = redis_client

    async def create_session(self, session_data: dict) -> str:
        session_id = str(uuid4())
        await self.redis_client.set(session_id, json.dumps(session_data), ex=7200)
        return session_id

    async def get_session(self, session_id: str) -> dict:
        try:
            data = await self.redis_client.get(session_id)
            return json.loads(data) if data else {}
        except (redis.RedisError, json.JSONDecodeError):
            return {}

    async def update_session(self, session_id: str, session_data: dict) -> None:
        await self.redis_client.setex(session_id, 3600, json.dumps(session_data))

# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index = await asyncio.to_thread(load_faiss_index, "faiss_index.bin")
    app.state.scheme_data = await asyncio.to_thread(load_scheme_data, "scheme_details.json")
    app.state.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    app.state.llm_model = "gemini-1.5-pro-002"
    app.state.project_id = os.getenv("GOOGLE_PROJECT_ID")

    app.state.redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True
    )

    yield  #

    # Cleanup resources
    await app.state.redis_client.close()

# Initialize FastAPI App
app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY"))

# Middleware for Session Management
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    session_manager = request.app.state.session_manager
    session_id = request.cookies.get("session_id")

    if session_id:
        session_data = await session_manager.get_session(session_id) or {"chat_history": []}
    else:
        session_data = {"chat_history": []}
        session_id = await session_manager.create_session(session_data)

    request.state.session = session_data
    response = await call_next(request)
    await session_manager.update_session(session_id, request.state.session)

    if "session_id" not in request.cookies:
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=3600,
            httponly=True,
            secure=True,
            samesite="Lax"
        )

    return response

# Custom Exception Class
class QueryProcessingError(Exception):
    pass

# Query Handling Endpoint
@app.post("/query/", response_model=QueryResponse)
async def handle_query(query_request: QueryRequest, request: Request):
    try:
        session = request.state.session  # Access session from request.state
        chat_history = session.get("chat_history", [])

        # Process query through the pipeline
        indices = await asyncio.to_thread(
            query_faiss_index,
            request.app.state.index,
            query_request.query_text,
            request.app.state.embedding_model
        )
        
        retrieved_docs = await asyncio.to_thread(
            retrieve_documents,
            indices,
            request.app.state.scheme_data
        )
        
        response = await asyncio.to_thread(
            generate_response,
            chat_history,
            retrieved_docs,
            request.app.state.llm_model,
            query_request.query_text,
            request.app.state.project_id
        )

        if not isinstance(response, QueryResponse):
            raise QueryProcessingError("Invalid response format")

        # Update chat history in session
        chat_history.append({
            "query": query_request.query_text,
            "response": response.content,
            "timestamp": datetime.now()
        })

        return response
    
    except QueryProcessingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Run Application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
