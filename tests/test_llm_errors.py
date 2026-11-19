import httpx
import pytest

from srag.llm.errors import is_transient


def with_status(code: int) -> Exception:
    return type("StatusError", (Exception,), {"status_code": code})()


class APITimeoutError(Exception):
    """Same name as the one the Anthropic/OpenAI SDKs raise."""


@pytest.mark.parametrize("error", [
    with_status(429), with_status(500), with_status(503), with_status(408),
    httpx.ConnectError("x"), httpx.ReadTimeout("x"), TimeoutError(), ConnectionError(), APITimeoutError(),
])
def test_temporary_failures_are_recognized(error):
    assert is_transient(error)


@pytest.mark.parametrize("error", [
    with_status(400), with_status(401), with_status(403), with_status(404), ValueError("x"), KeyError("x"),
])
def test_configuration_and_programming_errors_are_not_temporary(error):
    assert not is_transient(error)
