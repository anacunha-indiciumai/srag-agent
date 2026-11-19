"""Builds the chat model from configuration so the agent never depends on a specific provider.

Settings come from environment variables:
    SRAG_LLM_PROVIDER   "anthropic" (default), "ollama" or "gateway"
    SRAG_LLM_MODEL      model name; required for "gateway" (the service name catalog.schema.service)
    SRAG_OLLAMA_URL     Ollama server URL (default http://localhost:11434)
    ANTHROPIC_API_KEY   read by the Anthropic client; never stored or logged here
    DATABRICKS_HOST     workspace URL for "gateway"
    DATABRICKS_TOKEN    Databricks credential for "gateway"; never stored or logged here
"""
import os
import warnings
from dataclasses import dataclass
from urllib.parse import urlparse

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

PROVIDERS = ("anthropic", "ollama", "gateway")
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "ollama": "llama3.1"}  # "gateway" has no default
DEFAULT_OLLAMA_URL = "http://localhost:11434"
GATEWAY_PATH = "/ai-gateway/mlflow/v1"  # OpenAI-compatible Unity Gateway API
DATABRICKS_DOMAINS = (".cloud.databricks.com", ".gcp.databricks.com", ".azuredatabricks.net")  # AWS, GCP, Azure
MAX_OUTPUT_TOKENS = 2048


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    ollama_url: str = DEFAULT_OLLAMA_URL
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMSettings":
        provider = (os.environ.get("SRAG_LLM_PROVIDER") or "anthropic").lower()
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider '{provider}'. Use one of: {', '.join(PROVIDERS)}")
        model = os.environ.get("SRAG_LLM_MODEL") or DEFAULT_MODELS.get(provider)
        if not model:
            raise ValueError(f"SRAG_LLM_MODEL is required for provider '{provider}'")
        return cls(provider=provider, model=model, ollama_url=os.environ.get("SRAG_OLLAMA_URL") or DEFAULT_OLLAMA_URL)


def build_chat_model(settings: LLMSettings | None = None) -> BaseChatModel:
    settings = settings or LLMSettings.from_env()
    # Recent Claude models (Haiku 5.5, Sonnet 5.5) reject a non-default `temperature` with a 400,
    # so the Claude providers leave it unset. Only the local model keeps temperature 0.
    if settings.provider == "anthropic":
        _require_env("ANTHROPIC_API_KEY", hint="or use SRAG_LLM_PROVIDER=ollama")
        return ChatAnthropic(model=settings.model, temperature=None, max_tokens=MAX_OUTPUT_TOKENS)
    if settings.provider == "ollama":
        return ChatOllama(model=settings.model, base_url=settings.ollama_url, temperature=settings.temperature)
    if settings.provider == "gateway":
        host = _require_env("DATABRICKS_HOST")
        token = _require_env("DATABRICKS_TOKEN")
        if not host.startswith("https://"):
            raise ValueError("DATABRICKS_HOST must start with https:// (the token is sent to this host)")
        hostname = urlparse(host).hostname or ""
        if not hostname.endswith(DATABRICKS_DOMAINS):  # a warning, not an error: custom or private-link domains exist
            warnings.warn(f"DATABRICKS_HOST ({hostname}) is not a known Databricks domain; the token is sent to it.")
        return ChatOpenAI(model=settings.model, base_url=host.rstrip("/") + GATEWAY_PATH, api_key=token, temperature=None)
    raise ValueError(f"Unknown LLM provider '{settings.provider}'")


def _require_env(name: str, hint: str = "") -> str:
    value = os.environ.get(name)
    if not value:
        suffix = f" ({hint})" if hint else ""
        raise RuntimeError(f"{name} is not set. Export it in your shell{suffix}.")
    return value
