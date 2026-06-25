from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from agents import RunContextWrapper, function_tool
from agents.tracing import SpanError, get_current_span
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

_POPULATIONS: dict[str, int] = {
    "paris": 2_100_000,
    "tokyo": 14_000_000,
    "new york": 8_300_000,
    "new york city": 8_300_000,
    "nyc": 8_300_000,
}

_TIMEZONES: dict[str, str] = {
    "paris": "Europe/Paris",
    "tokyo": "Asia/Tokyo",
    "new york": "America/New_York",
    "new york city": "America/New_York",
    "nyc": "America/New_York",
}


def _normalize_city(city: str) -> str:
    return city.strip().lower()


@function_tool
def get_city_population(city: str) -> str:
    """Return the population of a supported city."""
    key = _normalize_city(city)
    population = _POPULATIONS.get(key)
    if population is None:
        return f"Unknown city: {city}. Supported cities: Paris, Tokyo, New York."
    return f"The population of {city.title()} is approximately {population:,}."


@function_tool
def get_local_time(city: str) -> str:
    """Return the current local time for a supported city."""
    key = _normalize_city(city)
    tz_name = _TIMEZONES.get(key)
    if tz_name is None:
        return f"Unknown city: {city}. Supported cities: Paris, Tokyo, New York."
    now = datetime.now(ZoneInfo(tz_name))
    return f"The local time in {city.title()} is {now.isoformat(timespec='seconds')}."


def _record_tool_error(ctx: RunContextWrapper[Any], error: Exception) -> str:
    """Surface a tool failure in the LangSmith trace, then return a message.

    Two things are needed for the failure to show up as an errored run:

    1. ``record_exception`` on the live OpenTelemetry span. LangSmith derives a
       run's error/status from the OTEL ``exception`` event (``exception.message``
       / ``exception.stacktrace``), *not* from the span's status code. Without
       this event the tool would look like a normal, successful response.
    2. ``set_error`` on the Agents SDK span, which the GenAI instrumentor maps to
       an OTEL ERROR status (belt-and-suspenders alongside the event).

    Returning a string (rather than re-raising) lets the agent recover and still
    answer the user.
    """
    otel_span = otel_trace.get_current_span()
    otel_span.record_exception(error)
    otel_span.set_status(Status(StatusCode.ERROR, str(error)))

    sdk_span = get_current_span()
    if sdk_span is not None:
        sdk_span.set_error(
            SpanError(
                message="Error running tool",
                data={"tool_name": "get_weather", "error": str(error)},
            )
        )
    return f"The weather service is currently unavailable: {error}"


@function_tool(failure_error_function=_record_tool_error)
def get_weather(city: str) -> str:
    """Return the current weather for a city (simulated downstream service)."""
    raise RuntimeError("Weather service unavailable (HTTP 503)")
