import warnings

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from srag.llm.factory import LLMSettings, build_chat_model


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("SRAG_LLM_PROVIDER", "SRAG_LLM_MODEL", "SRAG_OLLAMA_URL", "ANTHROPIC_API_KEY",
                 "DATABRICKS_HOST", "DATABRICKS_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_settings_default_to_anthropic_with_its_default_model():
    settings = LLMSettings.from_env()
    assert (settings.provider, settings.model) == ("anthropic", "claude-sonnet-5-5")


def test_settings_read_provider_and_model_from_the_environment(monkeypatch):
    monkeypatch.setenv("SRAG_LLM_PROVIDER", "OLLAMA")
    monkeypatch.setenv("SRAG_LLM_MODEL", "qwen2.5:7b")
    settings = LLMSettings.from_env()
    assert (settings.provider, settings.model) == ("ollama", "qwen2.5:7b")


def test_blank_variables_from_a_sourced_env_file_fall_back_to_the_defaults(monkeypatch):
    monkeypatch.setenv("SRAG_LLM_PROVIDER", "")
    monkeypatch.setenv("SRAG_LLM_MODEL", "")
    monkeypatch.setenv("SRAG_OLLAMA_URL", "")
    settings = LLMSettings.from_env()
    assert (settings.provider, settings.model) == ("anthropic", "claude-sonnet-5-5")
    assert settings.ollama_url == "http://localhost:11434"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("SRAG_LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMSettings.from_env()


def test_anthropic_without_api_key_fails_with_a_clear_message():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_chat_model(LLMSettings("anthropic", "claude-sonnet-5-5"))


def test_anthropic_builds_a_claude_chat_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert isinstance(build_chat_model(LLMSettings("anthropic", "claude-sonnet-5-5")), ChatAnthropic)


def test_gateway_requires_the_service_name_because_it_has_no_default_model(monkeypatch):
    monkeypatch.setenv("SRAG_LLM_PROVIDER", "gateway")
    with pytest.raises(ValueError, match="SRAG_LLM_MODEL is required"):
        LLMSettings.from_env()


@pytest.mark.parametrize("missing", ["DATABRICKS_HOST", "DATABRICKS_TOKEN"])
def test_gateway_without_workspace_or_token_fails_with_a_clear_message(monkeypatch, missing):
    present = {"DATABRICKS_HOST", "DATABRICKS_TOKEN"} - {missing}
    for name in present:
        monkeypatch.setenv(name, "https://example.cloud.databricks.com" if name.endswith("HOST") else "token-not-real")
    with pytest.raises(RuntimeError, match=missing):
        build_chat_model(LLMSettings("gateway", "catalog.schema.service"))


def test_gateway_points_the_openai_compatible_client_at_the_workspace_gateway_path(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com/")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token-not-real")
    model = build_chat_model(LLMSettings("gateway", "catalog.schema.service"))
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == "https://example.cloud.databricks.com/ai-gateway/mlflow/v1"
    assert model.model_name == "catalog.schema.service"


def test_gateway_refuses_a_host_that_is_not_https_because_the_token_would_travel_in_clear(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "http://example.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token-not-real")
    with pytest.raises(ValueError, match="https://"):
        build_chat_model(LLMSettings("gateway", "catalog.schema.service"))


def test_gateway_warns_when_the_token_would_go_to_a_host_that_is_not_a_databricks_domain(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://exemplo.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token-not-real")
    with pytest.warns(UserWarning, match="exemplo.com"):
        build_chat_model(LLMSettings("gateway", "catalog.schema.service"))


@pytest.mark.parametrize("host", [
    "https://dbc-1234.cloud.databricks.com", "https://adb-1.2.azuredatabricks.net", "https://x.gcp.databricks.com/",
])
def test_gateway_stays_quiet_for_the_known_databricks_domains(monkeypatch, host):
    monkeypatch.setenv("DATABRICKS_HOST", host)
    monkeypatch.setenv("DATABRICKS_TOKEN", "token-not-real")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_chat_model(LLMSettings("gateway", "catalog.schema.service"))


def test_gateway_does_not_send_temperature_because_recent_claude_models_reject_it(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token-not-real")
    model = build_chat_model(LLMSettings("gateway", "catalog.schema.service"))
    payload = model._get_request_payload([("user", "oi")])
    assert "temperature" not in payload


def test_anthropic_does_not_send_temperature_because_recent_claude_models_reject_it(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    model = build_chat_model(LLMSettings("anthropic", "claude-sonnet-5-5"))
    assert "temperature" not in model._get_request_payload([("user", "oi")])


def test_ollama_keeps_temperature_zero_for_repeatable_local_runs():
    assert build_chat_model(LLMSettings("ollama", "llama3.1")).temperature == 0.0


def test_ollama_builds_a_local_chat_model_without_any_key():
    model = build_chat_model(LLMSettings("ollama", "llama3.1"))
    assert isinstance(model, ChatOllama)
    assert model.base_url == "http://localhost:11434"
