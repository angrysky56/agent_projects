"""
Unified LLM Provider abstraction layer.

Supports multiple providers: Ollama (local + cloud), LM Studio, OpenAI, Anthropic.
Ollama is the default provider with auto-detection and cloud fallback.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import httpx
import os
import logging

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types."""

    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class Message:
    """Represents a chat message."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    reasoning: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    provider_type: ProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 900


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    @abstractmethod
    async def chat_completion(self, messages: List[Message], stream: bool = False, temperature: float = 0.7, max_tokens: Optional[int] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> AsyncIterator[str]:
        """Generate chat completion (streaming or non-streaming).

        Args:
            messages: List of Message objects
            stream: Whether to stream the response
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional list of OpenAI-format function definitions for function calling
            **kwargs: Additional provider-specific parameters
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available/reachable."""
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models from this provider."""
        pass


class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider with local and cloud support.

    Auto-detects local Ollama instance, falls back to cloud.
    Default provider for the COMPASS system.
    """

    LOCAL_URL = "http://localhost:11434"
    CLOUD_URL = "https://api.ollama.com"

    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.OLLAMA,
                api_key=os.getenv("OLLAMA_API_KEY"),
                base_url=None,  # Will auto-detect
                model="kimi-k2-thinking:cloud",
            )
        super().__init__(config)
        self.use_cloud = False
        self.effective_url = self.LOCAL_URL

    async def _detect_mode(self):
        """Detect whether to use local or cloud Ollama."""
        # Try local first
        try:
            response = await self.client.get(f"{self.LOCAL_URL}/api/tags", timeout=2.0)
            if response.status_code == 200:
                self.effective_url = self.LOCAL_URL
                self.use_cloud = False
                logger.info("Using local Ollama instance")
                return
        except Exception as e:
            logger.debug(f"Local Ollama not available: {e}")

        # Fall back to cloud
        if self.config.api_key:
            self.effective_url = self.CLOUD_URL
            self.use_cloud = True
            logger.info("Using Ollama Cloud")
        else:
            logger.warning("Neither local Ollama nor cloud API key available")

    async def is_available(self) -> bool:
        """Check if Ollama is available (local or cloud)."""
        await self._detect_mode()
        try:
            headers = {}
            if self.use_cloud and self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            response = await self.client.get(f"{self.effective_url}/api/tags", headers=headers)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False

    async def list_models(self) -> List[str]:
        """List available Ollama models."""
        await self._detect_mode()

        headers = {}
        if self.use_cloud and self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            response = await self.client.get(f"{self.effective_url}/api/tags", headers=headers)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def chat_completion(self, messages: List[Message], stream: bool = True, temperature: float = 0.7, max_tokens: Optional[int] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> AsyncIterator[str]:
        """Generate Ollama chat completion with streaming support and tool calling."""
        await self._detect_mode()

        # Convert messages to Ollama format
        ollama_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        headers = {"Content-Type": "application/json"}
        if self.use_cloud and self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model or "llama3.2:latest",
            "messages": ollama_messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        # Add tools if provided (Ollama function calling support)
        if tools:
            payload["tools"] = tools
            logger.info(f"Enabling function calling with {len(tools)} tools")

        # Add any additional kwargs
        payload.update(kwargs)

        if stream:
            async with self.client.stream("POST", f"{self.effective_url}/api/chat", json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        import json

                        try:
                            chunk = json.loads(line)
                            message = chunk.get("message", {})

                            # Check for tool calls in the response
                            if "tool_calls" in message:
                                # Return the entire message with tool_calls as JSON
                                yield json.dumps({"tool_calls": message["tool_calls"]})
                            elif "content" in message:
                                yield message["content"]
                        except json.JSONDecodeError:
                            continue
        else:
            response = await self.client.post(f"{self.effective_url}/api/chat", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {})

            # Check for tool calls
            if "tool_calls" in message:
                import json

                yield json.dumps({"tool_calls": message["tool_calls"]})
            elif "content" in message:
                yield message["content"]


class LMStudioProvider(BaseLLMProvider):
    """LM Studio provider (local OpenAI-compatible API)."""

    DEFAULT_URL = "http://localhost:1234/v1"

    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(provider_type=ProviderType.LM_STUDIO, base_url=self.DEFAULT_URL, model="local-model")
        super().__init__(config)

    async def is_available(self) -> bool:
        """Check if LM Studio is running."""
        try:
            response = await self.client.get(f"{self.config.base_url}/models")
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"LM Studio not available: {e}")
            return False

    async def list_models(self) -> List[str]:
        """List LM Studio models."""
        try:
            response = await self.client.get(f"{self.config.base_url}/models")
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to list LM Studio models: {e}")
            return []

    async def chat_completion(self, messages: List[Message], stream: bool = True, temperature: float = 0.7, max_tokens: Optional[int] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> AsyncIterator[str]:
        """Generate LM Studio chat completion."""
        openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        payload = {"model": self.config.model, "messages": openai_messages, "temperature": temperature, "stream": stream}

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if stream:
            async with self.client.stream("POST", f"{self.config.base_url}/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            import json

                            chunk = json.loads(data_str)
                            if chunk["choices"][0]["delta"].get("content"):
                                yield chunk["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError:
                            continue
        else:
            response = await self.client.post(f"{self.config.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            yield data["choices"][0]["message"]["content"]


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider."""

    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(provider_type=ProviderType.OPENAI, api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1", model="gpt-4-turbo-preview")
        super().__init__(config)

    async def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if not self.config.api_key:
            return False
        try:
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            response = await self.client.get(f"{self.config.base_url}/models", headers=headers)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"OpenAI not available: {e}")
            return False

    async def list_models(self) -> List[str]:
        """List OpenAI models."""
        if not self.config.api_key:
            return []

        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        try:
            response = await self.client.get(f"{self.config.base_url}/models", headers=headers)
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {e}")
            return []

    async def chat_completion(self, messages: List[Message], stream: bool = True, temperature: float = 0.7, max_tokens: Optional[int] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> AsyncIterator[str]:
        """Generate OpenAI chat completion."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key not configured")

        openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}

        payload = {"model": self.config.model, "messages": openai_messages, "temperature": temperature, "stream": stream}

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if stream:
            async with self.client.stream("POST", f"{self.config.base_url}/chat/completions", json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            import json

                            chunk = json.loads(data_str)
                            if chunk["choices"][0]["delta"].get("content"):
                                yield chunk["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError:
                            continue
        else:
            response = await self.client.post(f"{self.config.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            yield data["choices"][0]["message"]["content"]


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(provider_type=ProviderType.ANTHROPIC, api_key=os.getenv("ANTHROPIC_API_KEY"), base_url="https://api.anthropic.com/v1", model="claude-3-5-sonnet-20241022")
        super().__init__(config)

    async def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        return bool(self.config.api_key)

    async def list_models(self) -> List[str]:
        """List Anthropic models (hardcoded as API doesn't provide list endpoint)."""
        return ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]

    async def chat_completion(self, messages: List[Message], stream: bool = True, temperature: float = 0.7, max_tokens: Optional[int] = 4096, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> AsyncIterator[str]:
        """Generate Anthropic chat completion."""
        if not self.config.api_key:
            raise ValueError("Anthropic API key not configured")

        # Anthropic requires system messages separate
        system_msg = None
        anthropic_messages = []
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        headers = {"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}

        payload = {"model": self.config.model, "messages": anthropic_messages, "max_tokens": max_tokens or 4096, "temperature": temperature, "stream": stream}

        if system_msg:
            payload["system"] = system_msg

        if stream:
            async with self.client.stream("POST", f"{self.config.base_url}/messages", json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            import json

                            chunk = json.loads(data_str)
                            if chunk.get("type") == "content_block_delta":
                                if "delta" in chunk and "text" in chunk["delta"]:
                                    yield chunk["delta"]["text"]
                        except json.JSONDecodeError:
                            continue
        else:
            response = await self.client.post(f"{self.config.base_url}/messages", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            for content in data.get("content", []):
                if content.get("type") == "text":
                    yield content["text"]


async def get_available_providers() -> Dict[ProviderType, BaseLLMProvider]:
    """
    Auto-detect and return all available LLM providers.
    Prioritizes Ollama as the default.
    """
    providers = {}

    # Try Ollama first (default)
    async with OllamaProvider() as ollama:
        if await ollama.is_available():
            providers[ProviderType.OLLAMA] = ollama
            logger.info("Ollama provider available")

    # Try LM Studio
    async with LMStudioProvider() as lm_studio:
        if await lm_studio.is_available():
            providers[ProviderType.LM_STUDIO] = lm_studio
            logger.info("LM Studio provider available")

    # Try OpenAI
    async with OpenAIProvider() as openai:
        if await openai.is_available():
            providers[ProviderType.OPENAI] = openai
            logger.info("OpenAI provider available")

    # Try Anthropic
    async with AnthropicProvider() as anthropic:
        if await anthropic.is_available():
            providers[ProviderType.ANTHROPIC] = anthropic
            logger.info("Anthropic provider available")

    return providers


def create_provider(provider_type: ProviderType, config: Optional[ProviderConfig] = None) -> BaseLLMProvider:
    """Factory function to create an LLM provider instance."""
    providers_map = {
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.LM_STUDIO: LMStudioProvider,
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
    }

    provider_class = providers_map.get(provider_type)
    if not provider_class:
        raise ValueError(f"Unknown provider type: {provider_type}")

    return provider_class(config)
