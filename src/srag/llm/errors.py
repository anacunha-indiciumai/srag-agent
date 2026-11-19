"""Tells temporary LLM failures apart from configuration ones.

Temporary (rate limit, server error, timeout, no connection): the report can still be produced with
the code-generated explanation. Anything else (bad token, wrong model name, rejected parameter) must
stop the run, otherwise a broken setup would look like a working one that always falls back.
"""
import httpx

_TRANSIENT_STATUS = {408, 429}
# The Anthropic and OpenAI SDKs raise their own classes for these; matching by name avoids importing both.
_TRANSIENT_CLASS_NAMES = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}


def is_transient(error: BaseException) -> bool:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS or status >= 500
    if isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError)):
        return True
    return any(cls.__name__ in _TRANSIENT_CLASS_NAMES for cls in type(error).__mro__)
