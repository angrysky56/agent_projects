"""
FastAPI server for COMPASS Web UI.

Provides REST and WebSocket endpoints for chat, configuration, and MCP integration.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import logging
from contextlib import asynccontextmanager

from .llm_providers import create_provider, ProviderType, Message
from .mcp_client import initialize_mcp, shutdown_mcp
from .compass_api import get_compass_api, ProcessingUpdate, COMPASSResult

from .conversation_manager import ConversationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize conversation manager
conversation_manager = ConversationManager()


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("Starting COMPASS Web UI backend...")
    try:
        await initialize_mcp()
        logger.info("MCP client initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize MCP: {e}")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await shutdown_mcp()


app = FastAPI(title="COMPASS Web UI API", description="Backend API for the COMPASS cognitive framework with multi-provider LLM support", version="0.1.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models
# ============================================================================


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    provider: Optional[str] = "ollama"
    model: Optional[str] = None
    stream: bool = True
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    use_compass: bool = True
    context: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None  # Track conversation for history


class ProviderInfo(BaseModel):
    type: str
    available: bool
    models: List[str] = []


class MCPToolInfo(BaseModel):
    name: str
    description: str
    server_name: str


class MCPToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    server_name: Optional[str] = None


class MCPServerAdd(BaseModel):
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None


# ============================================================================
# Provider Management
# ============================================================================


@app.get("/api/providers")
async def list_providers() -> Dict[str, ProviderInfo]:
    """List all available LLM providers and their status."""
    providers_dict = {}

    # Check each provider type
    for provider_type in ProviderType:
        try:
            provider = create_provider(provider_type)
            async with provider:
                available = await provider.is_available()
                models = await provider.list_models() if available else []

            providers_dict[provider_type.value] = ProviderInfo(type=provider_type.value, available=available, models=models)
        except Exception as e:
            logger.error(f"Error checking provider {provider_type.value}: {e}")
            providers_dict[provider_type.value] = ProviderInfo(type=provider_type.value, available=False, models=[])

    return providers_dict


# ============================================================================
# Chat Endpoints
# ============================================================================


@app.post("/api/chat")
async def chat_completion(request: ChatRequest):
    """
    Generate chat completion with optional COMPASS integration.

    Supports streaming and non-streaming responses with conversation history.
    """
    try:
        # Convert to Message objects
        messages = [Message(role=msg.role, content=msg.content) for msg in request.messages]

        # Handle conversation persistence
        conversation_id = request.conversation_id
        if not conversation_id:
            # Create new conversation
            first_msg_preview = messages[0].content[:50] if messages else "New Chat"
            conversation_id = conversation_manager.create_conversation(title=first_msg_preview)
            logger.info(f"Created new conversation: {conversation_id}")

        # Save user message
        if messages:
            conversation_manager.add_message(conversation_id, {"role": messages[-1].role, "content": messages[-1].content})

        # Get provider
        provider_type = ProviderType(request.provider)
        provider = create_provider(provider_type)

        # Check availability first
        async with provider:
            if not await provider.is_available():
                raise HTTPException(status_code=503, detail=f"Provider {request.provider} not available")

        # Set model if provided
        if request.model:
            provider.config.model = request.model

        if request.use_compass and len(messages) > 0:
            # Use COMPASS for enhanced reasoning
            compass_api = get_compass_api()
            last_message = messages[-1].content

            # Build conversation context from history
            conversation_context = request.context or {}
            if len(messages) > 1:
                conversation_context["conversation_history"] = [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages[:-1]  # All except current message
                ]
                conversation_context["conversation_id"] = conversation_id

            if request.stream:

                async def compass_stream():
                    # Re-open provider for streaming session
                    async with provider:
                        compass_result = None
                        thinking_steps = []  # Collect thinking steps

                        async for update in compass_api.process_task(
                            last_message,
                            conversation_context,  # Pass full context
                            llm_provider=provider,
                        ):
                            if isinstance(update, ProcessingUpdate):
                                # Collect thinking steps
                                if update.stage == "thinking":
                                    thinking_steps.append({"timestamp": update.data.get("thinking_step") if update.data else None, "content": update.message})
                                # Still emit all updates for progress tracking
                                yield f"data: {json.dumps({'type': 'update', 'data': update.__dict__})}\n\n"
                            elif isinstance(update, COMPASSResult):
                                compass_result = update
                                # Attach thinking steps to result
                                compass_result.thinking = thinking_steps
                                yield f"data: {json.dumps({'type': 'result', 'data': update.__dict__})}\n\n"

                                # Emit thinking block immediately after result
                                if thinking_steps:
                                    yield f"data: {json.dumps({'type': 'thinking', 'steps': thinking_steps})}\n\n"

                        # Generate natural language response using LLM
                        if compass_result:
                            # Create context-aware prompt with conversation history
                            system_prompt = "You are COMPASS, an advanced cognitive system. You have just analyzed the user's request using your internal frameworks. Use the following reasoning trace and solution to formulate a helpful, natural language response. Explain your reasoning clearly."

                            # Format context safely
                            solution_str = json.dumps(compass_result.solution, default=str)

                            compass_context = f"Original Task: {last_message}\n\nInternal Solution/Decision: {solution_str}\n"

                            # Add conversation history to LLM context
                            if len(messages) > 1:
                                history_str = "\n".join([f"{m.role}: {m.content}" for m in messages[:-1]])
                                compass_context = f"Conversation History:\n{history_str}\n\n{compass_context}"

                            # Create messages for LLM
                            llm_messages = [Message(role="system", content=system_prompt), Message(role="user", content=compass_context)]

                            # Stream LLM response
                            assistant_response = ""
                            async for chunk in provider.chat_completion(llm_messages, stream=True, temperature=0.7):
                                assistant_response += chunk
                                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

                            # Save assistant response to conversation
                            conversation_manager.add_message(conversation_id, {"role": "assistant", "content": assistant_response, "reasoning": compass_result.__dict__})

                        yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': conversation_id})}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(compass_stream(), media_type="text/event-stream")
            else:
                # Non-streaming COMPASS
                result = None
                async for update in compass_api.process_task(
                    last_message,
                    conversation_context,  # Pass full context
                    llm_provider=provider,
                ):
                    if isinstance(update, COMPASSResult):
                        result = update

                # Save result
                if result:
                    conversation_manager.add_message(conversation_id, {"role": "assistant", "content": str(result.solution), "reasoning": result.__dict__})

                return {"type": "compass_result", "data": result.__dict__ if result else {}, "conversation_id": conversation_id}

        else:
            # Direct LLM completion without COMPASS
            if request.stream:

                async def llm_stream():
                    async with provider:
                        async for chunk in provider.chat_completion(messages, stream=True, temperature=request.temperature, max_tokens=request.max_tokens):
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(llm_stream(), media_type="text/event-stream")
            else:
                # Non-streaming
                async with provider:
                    content = ""
                    async for chunk in provider.chat_completion(messages, stream=False, temperature=request.temperature, max_tokens=request.max_tokens):
                        content += chunk
                    return {"type": "content", "content": content}

    except Exception as e:
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Chat
# ============================================================================


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            # Parse request
            messages = [Message(role=msg["role"], content=msg["content"]) for msg in data.get("messages", [])]
            provider_type = ProviderType(data.get("provider", "ollama"))
            use_compass = data.get("use_compass", True)

            provider = create_provider(provider_type)

            async with provider:
                if use_compass and len(messages) > 0:
                    # COMPASS processing
                    compass_api = get_compass_api()
                    async for update in compass_api.process_task(messages[-1].content, data.get("context"), llm_provider=provider):
                        if isinstance(update, ProcessingUpdate):
                            await websocket.send_json({"type": "update", "data": update.__dict__})
                        elif isinstance(update, COMPASSResult):
                            await websocket.send_json({"type": "result", "data": update.__dict__})
                else:
                    # Direct LLM
                    async for chunk in provider.chat_completion(messages, stream=True):
                        await websocket.send_json({"type": "content", "content": chunk})

                await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": str(e)})


# ============================================================================
# COMPASS Endpoints
# ============================================================================


@app.get("/api/compass/status")
async def compass_status():
    """Get COMPASS framework status."""
    compass_api = get_compass_api()
    return await compass_api.get_status()


@app.get("/api/compass/trace")
async def compass_trace():
    """Export reasoning trace from last execution."""
    compass_api = get_compass_api()
    return await compass_api.export_reasoning_trace()


# ============================================================================
# MCP Endpoints (Placeholder - for future MCP server expansion)
# ============================================================================


@app.get("/api/mcp/servers")
async def list_mcp_servers():
    """List all connected MCP servers."""
    return {"connected": [], "all_servers": []}


@app.get("/api/mcp/tools")
async def list_mcp_tools(server_name: Optional[str] = None):
    """List MCP tools (all or from specific server)."""
    return {"tools": []}


# ============================================================================
# Conversation Management Endpoints
# ============================================================================


@app.get("/api/conversations")
async def list_conversations(limit: int = 50, offset: int = 0):
    """List all conversations."""
    return conversation_manager.list_conversations(limit=limit, offset=offset)


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all messages."""
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations")
async def create_conversation(title: str = "New Conversation"):
    """Create a new conversation."""
    conv_id = conversation_manager.create_conversation(title=title)
    return {"conversation_id": conv_id, "title": title}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    deleted = conversation_manager.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "conversation_id": conversation_id}


@app.patch("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, title: str):
    """Update conversation title."""
    updated = conversation_manager.update_conversation_title(conversation_id, title)
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "updated", "conversation_id": conversation_id, "title": title}


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api_server:app", host="0.0.0.0", port=8000, reload=True)
