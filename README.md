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

Copy `.env.example` to `.env` and set your API keys:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-... and LANGSMITH_API_KEY=lsv2_...
```

Or export them directly:

```bash
export OPENAI_API_KEY=sk-...
export LANGSMITH_API_KEY=lsv2_...
```

## Run

```bash
python -m demo_agent
```

The demo prompt asks for Tokyo's population and local time, which should invoke both tools (`get_city_population` and `get_local_time`).

## Tracing

Traces go to [LangSmith](https://smith.langchain.com/) over **OpenTelemetry** (no LangSmith SDK). The setup is env vars plus a short call to `setup_tracing()` in [`src/demo_agent/otel.py`](src/demo_agent/otel.py):

```python
from demo_agent.otel import setup_tracing, shutdown_tracing

provider, instrumentor = setup_tracing()
try:
    result = Runner.run_sync(agent, prompt)
finally:
    shutdown_tracing(provider, instrumentor)
```

LangSmith’s [OpenAI Agents SDK guide](https://docs.langchain.com/langsmith/trace-with-openai-agents-sdk) documents the `OpenAIAgentsTracingProcessor` integration (which requires the LangSmith SDK). This demo instead stays on the pure [OpenTelemetry path](https://docs.langchain.com/langsmith/trace-with-opentelemetry): standard OTLP exporter, no LangSmith SDK.

Under the hood, the [OpenInference instrumentor for OpenAI Agents](https://github.com/Arize-ai/openinference/tree/main/python/instrumentation/openinference-instrumentation-openai-agents) registers a trace processor on the Agents SDK that emits OpenTelemetry spans using the [OpenInference semantic conventions](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md). LangSmith maps those attributes (`openinference.span.kind`, `input.value`, `output.value`, `llm.model_name`, `tool.name`, …) to native run types, so agent/turn spans show up as `chain`, model calls as `llm`, and tools as `tool`, each with proper inputs/outputs.

> The barebones `opentelemetry-instrumentation-openai-agents-v2` (GenAI semconv) instrumentor was tried first, but LangSmith does not map its `gen_ai.operation.name` values for `invoke_agent`/workflow spans — every non-leaf span landed as `llm` with `unknown` names and no I/O. OpenInference resolves this.

You must flush the Agents SDK trace queue (`flush_traces()`) before flushing the OTLP exporter — the trace processor’s own `force_flush()` is a no-op.

A small custom `SpanProcessor` (`_CleanLLMIOProcessor`, registered ahead of the exporter) rewrites the LLM spans’ I/O before export. OpenInference stores `input.value` as a stringified request list and `output.value` as the entire OpenAI Responses object (billing, timestamps, store flags, …), which LangSmith renders verbatim. The processor replaces them with a compact `{"messages": [...]}` shape so model calls show clean chat messages and tool calls.

Set in `.env`:

- `LANGSMITH_API_KEY` — required
- `LANGSMITH_PROJECT` — optional, defaults to `demo-agent`

After running, open that project in LangSmith to see the agent, LLM, and tool spans.

For self-hosted LangSmith, set `OTEL_EXPORTER_OTLP_ENDPOINT` to `<your-host>/api/v1/otel` (do not include `/v1/traces`; the exporter adds that suffix).

## Project layout

- `src/demo_agent/tools.py` — `@function_tool` definitions
- `src/demo_agent/agent.py` — `Agent` configuration
- `src/demo_agent/otel.py` — OTEL export to LangSmith (OpenInference instrumentor + OTLP exporter)
- `src/demo_agent/__main__.py` — entrypoint using `Runner.run_sync`
