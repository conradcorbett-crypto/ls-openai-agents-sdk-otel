from agents import Runner

from demo_agent.agent import agent


def main() -> None:
    prompt = "What's the population of Tokyo, and what time is it there now?"
    result = Runner.run_sync(agent, prompt)
    print(result.final_output)


if __name__ == "__main__":
    main()
