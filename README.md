# OpenAI Agents SDK demo

A minimal demo of the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/): one agent with two function tools.

## Setup

Requires **Python 3.10+**. If your default `python3` is older, use [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Or with plain pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy `.env.example` to `.env` and set your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

Or export it directly:

```bash
export OPENAI_API_KEY=sk-...
```

## Run

```bash
python -m demo_agent
```

The demo prompt asks for Tokyo's population and local time, which should invoke both tools (`get_city_population` and `get_local_time`).

## Project layout

- `src/demo_agent/tools.py` — `@function_tool` definitions
- `src/demo_agent/agent.py` — `Agent` configuration
- `src/demo_agent/__main__.py` — entrypoint using `Runner.run_sync`
