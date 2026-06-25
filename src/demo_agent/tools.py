from datetime import datetime
from zoneinfo import ZoneInfo

from agents import function_tool

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


@function_tool
def get_weather(city: str) -> str:
    """Return the current weather for a city (simulated downstream service)."""
    raise RuntimeError("Weather service unavailable (HTTP 503)")
