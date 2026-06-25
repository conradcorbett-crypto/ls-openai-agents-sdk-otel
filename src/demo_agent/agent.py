from agents import Agent

from demo_agent.tools import get_city_population, get_local_time, get_weather

agent = Agent(
    name="CityHelper",
    instructions=(
        "You help users learn about cities. "
        "Use get_city_population for population questions, "
        "get_local_time for the current local time, and "
        "get_weather for weather questions. "
        "If a city is unknown, say so clearly. "
        "If a tool reports that a service is unavailable, tell the user plainly."
    ),
    tools=[get_city_population, get_local_time, get_weather],
)
