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
from .mcp_client import initialize_mcp, shutdown_mcp, get_mcp_client
from .compass_api import get_compass_api, ProcessingUpdate, COMPASSResult
from .tool_permissions import get_permission_manager, classify_tool_risk

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
    role: str  # "system", "user", "assistant", "tool"
    content: str
    reasoning: Optional[Dict[str, Any]] = None  # For COMPASS reasoning trace
    tool_calls: Optional[List[Dict[str, Any]]] = None  # For assistant messages with tool requests
    tool_name: Optional[str] = None  # For tool response messages


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
        # Convert request messages to internal Message objects
        messages = [Message(role=msg.role, content=msg.content, reasoning=msg.reasoning, tool_calls=msg.tool_calls) for msg in request.messages]

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

        # Fetch available MCP tools for function calling (always fetch)
        available_tools = []
        from .mcp_tool_adapter import get_available_tools_for_llm

        mcp_client = await get_mcp_client()
        if mcp_client:
            available_tools = await get_available_tools_for_llm(mcp_client)
            if available_tools:
                logger.info(f"🔧 Loaded {len(available_tools)} MCP tools")
            else:
                logger.warning("⚠️ No MCP tools available")
        else:
            logger.warning("⚠️ MCP client not initialized")

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
                            mcp_client=await get_mcp_client(),
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
                            # Create context-aware prompt with full COMPASS reasoning
                            system_prompt = """You are COMPASS, an advanced cognitive system that uses multiple AI frameworks for deep reasoning.

You have just analyzed the user's request through your internal cognitive pipeline including:
- SHAPE: Input processing and concept extraction
- SLAP: Semantic logic and advancement planning
- SMART: Objective setting and feasibility analysis
- Integrated Intelligence: Multi-modal reasoning and decision making
- Constraint Governor: System trace and validation

Your task is to provide a clear, helpful response to the user. You should be transparent about your reasoning process and refer to your internal system traces (critiques, objectives) if they are relevant to the answer."""

                            # Build comprehensive reasoning context
                            reasoning_parts = []

                            # Add trajectory information if available
                            if hasattr(compass_result, "trajectory") and compass_result.trajectory:
                                traj_dict = compass_result.trajectory if isinstance(compass_result.trajectory, dict) else compass_result.trajectory

                                # Add Constraint Violations (System Trace)
                                if "constraint_violations" in traj_dict:
                                    violations = traj_dict["constraint_violations"]
                                    if violations and violations.get("total_violations", 0) > 0:
                                        reasoning_parts.append(f"System Trace / Critiques: {json.dumps(violations)}")

                                # Extract SLAP planning information
                                if "steps" in traj_dict and traj_dict["steps"]:
                                    for step_pair in traj_dict["steps"]:
                                        if isinstance(step_pair, list) and len(step_pair) > 0:
                                            slap_plan = step_pair[0] if isinstance(step_pair[0], dict) else {}

                                            # Add advancement score and semantic reasoning
                                            if "advancement" in slap_plan:
                                                reasoning_parts.append(f"Reasoning Quality Score: {slap_plan['advancement']:.2f}")

                                            # Add conceptualization
                                            if "conceptualization" in slap_plan:
                                                concept = slap_plan["conceptualization"]
                                                if isinstance(concept, dict) and "primary_concept" in concept:
                                                    reasoning_parts.append(f"Primary Concept: {concept['primary_concept']}")

                                            # Add scrutiny findings
                                            if "scrutiny" in slap_plan and isinstance(slap_plan["scrutiny"], dict):
                                                scrutiny = slap_plan["scrutiny"]
                                                if "gaps" in scrutiny and scrutiny["gaps"]:
                                                    reasoning_parts.append(f"Identified Gaps: {', '.join(scrutiny['gaps'])}")

                                        # Extract decision reasoning from second element in step
                                        if isinstance(step_pair, list) and len(step_pair) > 1:
                                            decision = step_pair[1] if isinstance(step_pair[1], dict) else {}
                                            if "reasoning" in decision:
                                                reasoning_parts.append(f"Decision Basis: {decision['reasoning']}")
                                            if "intelligence_breakdown" in decision:
                                                breakdown = decision["intelligence_breakdown"]
                                                if isinstance(breakdown, dict):
                                                    top_scores = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:3]
                                                    scores_str = ", ".join([f"{k}: {v:.2f}" for k, v in top_scores])
                                                    reasoning_parts.append(f"Top Intelligence Factors: {scores_str}")

                            # Build the context string
                            compass_context = f"User's Question: {last_message}\n\n"

                            if reasoning_parts:
                                compass_context += "My Internal Analysis:\n" + "\n".join([f"- {part}" for part in reasoning_parts]) + "\n\n"

                            # Add solution/decision
                            solution_str = str(compass_result.solution) if compass_result.solution else "Analysis complete"
                            compass_context += f"Recommended Action: {solution_str}\n\n"

                            # Add conversation history to LLM context
                            if len(messages) > 1:
                                history_parts = []
                                for m in messages[:-1]:
                                    msg_str = f"{m.role}: {m.content}"
                                    # Add reasoning summary if available
                                    if m.role == "assistant" and m.reasoning:
                                        if isinstance(m.reasoning, dict):
                                            # Extract key reasoning points
                                            r_summary = []
                                            if "trajectory" in m.reasoning:
                                                traj = m.reasoning["trajectory"]
                                                if isinstance(traj, dict) and "steps" in traj:
                                                    r_summary.append(f"[Reasoning Steps: {len(traj['steps'])}]")
                                            if "solution" in m.reasoning:
                                                r_summary.append(f"[Solution: {str(m.reasoning['solution'])[:100]}...]")

                                            if r_summary:
                                                msg_str += f"\n(Internal Thought Trace: {' '.join(r_summary)})"

                                    history_parts.append(msg_str)

                                history_str = "\n".join(history_parts)
                                compass_context = f"Previous Conversation:\n{history_str}\n\n{compass_context}"

                            compass_context += "Please provide a helpful, natural response to the user based on this analysis. Focus on answering their question clearly and explaining your reasoning."

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
                async with provider:
                    result = None
                    async for update in compass_api.process_task(
                        last_message,
                        conversation_context,  # Pass full context
                        llm_provider=provider,
                        mcp_client=await get_mcp_client(),
                    ):
                        if isinstance(update, COMPASSResult):
                            result = update

                    # Generate natural language response using LLM (same logic as streaming)
                    if result:
                        system_prompt = """You are COMPASS, an advanced cognitive system that uses multiple AI frameworks for deep reasoning.

You have just analyzed the user's request through your internal cognitive pipeline including:
- SHAPE: Input processing and concept extraction
- SLAP: Semantic logic and advancement planning
- SMART: Objective setting and feasibility analysis
- Integrated Intelligence: Multi-modal reasoning and decision making

Your task is to provide a clear, helpful response to the user based on your analysis. Explain your reasoning naturally without mentioning the internal framework names."""

                        # Build comprehensive reasoning context
                        reasoning_parts = []

                        # Add trajectory information if available
                        if hasattr(result, "trajectory") and result.trajectory:
                            traj_dict = result.trajectory if isinstance(result.trajectory, dict) else result.trajectory

                            # Extract SLAP planning information
                            if "steps" in traj_dict and traj_dict["steps"]:
                                for step_pair in traj_dict["steps"]:
                                    if isinstance(step_pair, list) and len(step_pair) > 0:
                                        slap_plan = step_pair[0] if isinstance(step_pair[0], dict) else {}

                                        # Add advancement score and semantic reasoning
                                        if "advancement" in slap_plan:
                                            reasoning_parts.append(f"Reasoning Quality Score: {slap_plan['advancement']:.2f}")

                                        # Add conceptualization
                                        if "conceptualization" in slap_plan:
                                            concept = slap_plan["conceptualization"]
                                            if isinstance(concept, dict) and "primary_concept" in concept:
                                                reasoning_parts.append(f"Primary Concept: {concept['primary_concept']}")

                                        # Add scrutiny findings
                                        if "scrutiny" in slap_plan and isinstance(slap_plan["scrutiny"], dict):
                                            scrutiny = slap_plan["scrutiny"]
                                            if "gaps" in scrutiny and scrutiny["gaps"]:
                                                reasoning_parts.append(f"Identified Gaps: {', '.join(scrutiny['gaps'])}")

                                    # Extract decision reasoning from second element in step
                                    if isinstance(step_pair, list) and len(step_pair) > 1:
                                        decision = step_pair[1] if isinstance(step_pair[1], dict) else {}
                                        if "reasoning" in decision:
                                            reasoning_parts.append(f"Decision Basis: {decision['reasoning']}")
                                        if "intelligence_breakdown" in decision:
                                            breakdown = decision["intelligence_breakdown"]
                                            if isinstance(breakdown, dict):
                                                top_scores = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:3]
                                                scores_str = ", ".join([f"{k}: {v:.2f}" for k, v in top_scores])
                                                reasoning_parts.append(f"Top Intelligence Factors: {scores_str}")

                        # Build the context string
                        compass_context = f"User's Question: {last_message}\n\n"

                        if reasoning_parts:
                            compass_context += "My Internal Analysis:\n" + "\n".join([f"- {part}" for part in reasoning_parts]) + "\n\n"

                        # Add solution/decision
                        solution_str = str(result.solution) if result.solution else "Analysis complete"
                        compass_context += f"Recommended Action: {solution_str}\n\n"

                        # Add conversation history to LLM context
                        if len(messages) > 1:
                            history_str = "\n".join([f"{m.role}: {m.content}" for m in messages[:-1]])
                            compass_context = f"Previous Conversation:\n{history_str}\n\n{compass_context}"

                        compass_context += "Please provide a helpful, natural response to the user based on this analysis. Focus on answering their question clearly and explaining your reasoning."

                        # Create messages for LLM
                        llm_messages = [Message(role="system", content=system_prompt), Message(role="user", content=compass_context)]

                        # Get LLM response
                        assistant_response = ""
                        async for chunk in provider.chat_completion(llm_messages, stream=False, temperature=0.7):
                            assistant_response += chunk

                        # Save assistant response
                        conversation_manager.add_message(conversation_id, {"role": "assistant", "content": assistant_response, "reasoning": result.__dict__})

                        return {"type": "compass_result", "content": assistant_response, "data": result.__dict__, "conversation_id": conversation_id}

                    return {"type": "compass_result", "data": {}, "conversation_id": conversation_id}

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
                # Non-streaming with tool calling support
                async with provider:
                    from .mcp_tool_adapter import format_tool_call_for_mcp, format_tool_result_for_llm
                    import json as json_module

                    # Initial LLM call with tools
                    content = ""
                    tool_calls_detected = None

                    async for chunk in provider.chat_completion(messages, stream=False, temperature=request.temperature, max_tokens=request.max_tokens, tools=available_tools if available_tools else None):
                        content += chunk

                    # Check if response contains tool calls
                    try:
                        if "{" in content and "tool_calls" in content:
                            parsed = json_module.loads(content)
                            if "tool_calls" in parsed:
                                tool_calls_detected = parsed["tool_calls"]
                    except json_module.JSONDecodeError:
                        pass  # Normal text response

                    # If tool calls detected, execute them and get final response
                    if tool_calls_detected:
                        logger.info(f"LLM requested {len(tool_calls_detected)} tool call(s)")

                        # Build messages history with assistant's tool call request
                        tool_messages = messages + [Message(role="assistant", content="")]

                        # Execute each tool and add results
                        mcp_client = await get_mcp_client()
                        for tool_call in tool_calls_detected:
                            tool_name, arguments = format_tool_call_for_mcp(tool_call)
                            logger.info(f"Executing tool: {tool_name} with args: {arguments}")

                            # Save assistant's tool call to history
                            conversation_manager.add_message(conversation_id, {"role": "assistant", "content": "", "tool_calls": [tool_call]})

                            try:
                                # Execute tool via MCP
                                result = await mcp_client.call_tool(tool_name, arguments)
                                tool_result_msg = format_tool_result_for_llm(tool_name, result)

                                # Add tool result to conversation history
                                tool_messages.append(Message(role=tool_result_msg["role"], content=tool_result_msg["content"]))

                                # Save tool result to DB
                                conversation_manager.add_message(conversation_id, {"role": tool_result_msg["role"], "content": tool_result_msg["content"], "name": tool_name})

                            except Exception as e:
                                logger.error(f"Error executing tool {tool_name}: {e}")
                                error_msg = f"Error executing {tool_name}: {str(e)}"
                                tool_messages.append(Message(role="tool", content=error_msg))
                                conversation_manager.add_message(conversation_id, {"role": "tool", "content": error_msg, "name": tool_name})

                        # Get final response from LLM with tool results
                        final_content = ""
                        async for chunk in provider.chat_completion(tool_messages, stream=False, temperature=request.temperature, max_tokens=request.max_tokens, tools=available_tools if available_tools else None):
                            final_content += chunk

                        # Save final response
                        conversation_manager.add_message(conversation_id, {"role": "assistant", "content": final_content})

                        return {"type": "content", "content": final_content, "conversation_id": conversation_id}
                    else:
                        # No tool calls, return direct response
                        conversation_manager.add_message(conversation_id, {"role": "assistant", "content": content})
                        return {"type": "content", "content": content, "conversation_id": conversation_id}

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
            messages = [Message(role=msg["role"], content=msg["content"], reasoning=msg.get("reasoning"), tool_calls=msg.get("tool_calls")) for msg in data.get("messages", [])]
            provider_type = ProviderType(data.get("provider", "ollama"))
            use_compass = data.get("use_compass", True)

            provider = create_provider(provider_type)

            async with provider:
                if use_compass and len(messages) > 0:
                    # COMPASS processing
                    compass_api = get_compass_api()
                    async for update in compass_api.process_task(messages[-1].content, data.get("context"), llm_provider=provider, mcp_client=await get_mcp_client()):
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
    # In this simplified single-client version, we just check if the global client is connected
    client = await get_mcp_client()
    connected = list(client.sessions.keys()) if client else []
    return {"connected": connected, "all_servers": connected}


@app.get("/api/mcp/tools")
async def list_mcp_tools(server_name: Optional[str] = None):
    """List MCP tools (all or from specific server)."""
    try:
        client = await get_mcp_client()
        if not client:
            return {"tools": []}

        tools = await client.list_tools()
        # Do NOT overwrite server_name, it's already set by mcp_client.list_tools()

        return {"tools": tools}
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        return {"tools": []}


@app.post("/api/mcp/call-tool")
async def call_mcp_tool(request: MCPToolCall):
    """
    Execute an MCP tool.

    Args:
        request: MCPToolCall containing tool_name, arguments, and optional server_name

    Returns:
        The result from the tool execution
    """
    try:
        client = await get_mcp_client()
        if not client:
            raise HTTPException(status_code=503, detail="MCP client not initialized")

        if not client.sessions:
            raise HTTPException(status_code=503, detail="No MCP servers connected")

        # Call the tool
        result = await client.call_tool(name=request.tool_name, arguments=request.arguments, server_name=request.server_name)

        # The result from MCP SDK is a CallToolResult object
        # We need to convert it to a dict for JSON serialization
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump()
        elif hasattr(result, "__dict__"):
            result_dict = result.__dict__
        else:
            result_dict = {"result": str(result)}

        return {"success": True, "result": result_dict}

    except RuntimeError as e:
        # Tool not found or server not connected
        logger.error(f"Runtime error calling tool {request.tool_name}: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error calling tool {request.tool_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


# ============================================================================
# Tool Permission Management
# ============================================================================


class ToolPermissionRequest(BaseModel):
    tool_name: str
    server_name: str
    action: str  # "ALLOW_ALWAYS", "DENY_ALWAYS", "ASK"
    risk_level: str  # "SAFE", "MODERATE", "DANGEROUS"


@app.get("/api/tool-permissions")
async def list_tool_permissions():
    """List all stored tool permissions."""
    try:
        permission_manager = get_permission_manager()
        permissions = permission_manager.list_permissions()
        return {"permissions": [p.to_dict() for p in permissions]}
    except Exception as e:
        logger.error(f"Error listing permissions: {e}", exc_info=True)
        return {"permissions": []}


@app.post("/api/tool-permissions")
async def save_tool_permission(request: ToolPermissionRequest):
    """Save a tool permission decision."""
    try:
        permission_manager = get_permission_manager()

        permission_manager.set_permission(
            tool_name=request.tool_name,
            server_name=request.server_name,
            action=request.action,  # type: ignore
            risk_level=request.risk_level,  # type: ignore
        )

        return {"success": True, "message": "Permission saved"}
    except Exception as e:
        logger.error(f"Error saving permission: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/tool-permissions/{server_name}/{tool_name}")
async def remove_tool_permission(server_name: str, tool_name: str):
    """Remove a stored tool permission."""
    try:
        permission_manager = get_permission_manager()

        removed = permission_manager.remove_permission(tool_name, server_name)

        if removed:
            return {"success": True, "message": "Permission removed"}
        else:
            raise HTTPException(status_code=404, detail="Permission not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing permission: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/tool-permissions")
async def clear_all_permissions():
    """Clear all stored tool permissions."""
    try:
        permission_manager = get_permission_manager()
        permission_manager.clear_all()
        return {"success": True, "message": "All permissions cleared"}
    except Exception as e:
        logger.error(f"Error clearing permissions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tool-risk/{tool_name}")
async def get_tool_risk_level(tool_name: str, description: str = ""):
    """Classify a tool's risk level."""
    try:
        risk_level = classify_tool_risk(tool_name, description)
        return {"tool_name": tool_name, "risk_level": risk_level}
    except Exception as e:
        logger.error(f"Error classifying tool risk: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
