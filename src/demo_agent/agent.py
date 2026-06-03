from agents import Agent

from demo_agent.tools import get_city_population, get_local_time

agent = Agent(
    name="CityHelper",
    instructions=(
        "You help users learn about cities. "
        "Use get_city_population for population questions and "
        "get_local_time for the current local time. "
        "If a city is unknown, say so clearly."
    ),
    tools=[get_city_population, get_local_time],
)
