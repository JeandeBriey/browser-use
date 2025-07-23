import json
import aiohttp
import re
from dataclasses import dataclass, field
from typing import TypeVar, overload, Any
from pydantic import BaseModel
from .parser import try_parse_ollama_output

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.ollama.serializer import OllamaMessageSerializer
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatOllama(BaseChatModel):
    """
    Wrapper pour le modèle Ollama (ex: llama3:latest) via l'API HTTP locale.
    """
    model: str = "llama3:latest"
    temperature: float = 0.7
    base_url: str = "http://localhost:11434"
    default_query: dict[str, Any] = field(default_factory=dict)

    @property
    def provider(self) -> str:
        return 'ollama'

    @property
    def name(self) -> str:
        return str(self.model)

    def _get_usage(self, response: dict) -> ChatInvokeUsage | None:
        # Ollama ne fournit pas de comptage de tokens par défaut
        return None

    @overload
    async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """
        Appelle le modèle Ollama avec les messages fournis.
        """
        prompt = OllamaMessageSerializer.serialize_messages(messages)
        # Fusionne les options par défaut et celles du prompt
        options = {"temperature": self.temperature}
        if self.default_query:
            options.update(self.default_query)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": options,
            "stream": False
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as resp:
                    if resp.status != 200:
                        raise ModelProviderError(f"Ollama API error: {resp.status}", status_code=resp.status, model=self.model)
                    data = await resp.json()
                    text = data.get("response", "")
                    # Si output_format est fourni, on tente de parser le JSON
                    if output_format is not None:
                        try:
                            parsed = try_parse_ollama_output(text, output_format)
                            return ChatInvokeCompletion(
                                completion=parsed,
                                usage=self._get_usage(data),
                            )
                        except Exception as e:
                            raise ModelProviderError(f"Failed to parse Ollama output as JSON: {e}\nRéponse brute:\n{text}", model=self.model)
                    return ChatInvokeCompletion(
                        completion=text,
                        usage=self._get_usage(data),
                    )
        except Exception as e:
            raise ModelProviderError(f"Ollama API call failed: {e}", model=self.model)
