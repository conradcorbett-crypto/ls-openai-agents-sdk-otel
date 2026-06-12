import os
from pathlib import Path

from agents import Runner
from dotenv import load_dotenv

from demo_agent.agent import agent
from demo_agent.otel import setup_tracing, shutdown_tracing

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to .env in the project root "
            f"({_PROJECT_ROOT / '.env'}) or export it in your shell."
        )
    provider, instrumentor = setup_tracing()
    try:
        prompt = "What's the population of Tokyo, and what time is it there now?"
        result = Runner.run_sync(agent, prompt)
        print(result.final_output)
    finally:
        shutdown_tracing(provider, instrumentor)


if __name__ == "__main__":
    main()
