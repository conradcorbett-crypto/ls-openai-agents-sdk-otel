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

The demo prompt asks for Tokyo's population, local time, and weather, which invokes all three tools (`get_city_population`, `get_local_time`, and `get_weather`). `get_weather` deliberately fails to demonstrate error tracing (see [Tool errors](#tool-errors)).

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

LangSmith's [OpenAI Agents SDK guide](https://docs.langchain.com/langsmith/trace-with-openai-agents-sdk) documents the `OpenAIAgentsTracingProcessor` integration (which requires the LangSmith SDK). This demo instead stays on the pure [OpenTelemetry path](https://docs.langchain.com/langsmith/trace-with-opentelemetry): standard OTLP exporter, no LangSmith SDK.

Under the hood, the [official OpenTelemetry GenAI instrumentor for OpenAI Agents](https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2) registers a trace processor on the Agents SDK that emits spans using the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/). LangSmith maps those attributes (`gen_ai.operation.name`, `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.tool.name`, …) to native run types.

> A parallel branch (`cursor/langsmith-otel-tracing`) uses the [OpenInference instrumentor](https://github.com/Arize-ai/openinference/tree/main/python/instrumentation/openinference-instrumentation-openai-agents) instead. This branch uses the official GenAI semconv instrumentor. LangSmith's published OTEL mapping table only lists `gen_ai.operation.name` → run type for `chat`/`completion`/`embedding`, not `invoke_agent` → `chain`, so a small custom `SpanProcessor` (`_LangSmithGenAIMappingProcessor`, registered ahead of the exporter) sets `langsmith.span.kind` and `langsmith.trace.name` for agent, tool, and workflow spans.

You must flush the Agents SDK trace queue (`flush_traces()`) before flushing the OTLP exporter — the trace processor's own `force_flush()` is a no-op.

Set in `.env`:

- `LANGSMITH_API_KEY` — required
- `LANGSMITH_PROJECT` — optional, defaults to `demo-agent`
- `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` — optional, defaults to `span_and_event` (captures prompts/responses on spans and as OTEL events)

After running, open that project in LangSmith to see the agent, LLM, and tool spans.

## Tool errors

By default, when a `@function_tool` raises, the Agents SDK's built-in `failure_error_function` swallows the exception and returns an error *string* to the model — so the tool span looks like a normal, successful response in LangSmith.

Crucially, LangSmith derives a run's error/status from the OpenTelemetry [`exception` event](https://docs.langchain.com/langsmith/trace-with-opentelemetry#supported-opentelemetry-attribute-and-event-mapping) (`exception.message` / `exception.stacktrace`) — **not** from the span's status code. The GenAI instrumentor sets the span status to `ERROR` but never records an `exception` event, so a failing tool still shows up as green.

To fix this, `get_weather` in [`src/demo_agent/tools.py`](src/demo_agent/tools.py) uses a custom `failure_error_function` that records the exception on the live OTEL span (and marks the SDK span errored) before returning a graceful message:

```python
from agents.tracing import SpanError, get_current_span
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

def _record_tool_error(ctx, error):
    otel_span = otel_trace.get_current_span()
    otel_span.record_exception(error)            # LangSmith reads this event
    otel_span.set_status(Status(StatusCode.ERROR, str(error)))
    sdk_span = get_current_span()
    if sdk_span is not None:
        sdk_span.set_error(SpanError(message="Error running tool", data={...}))
    return f"The weather service is currently unavailable: {error}"

@function_tool(failure_error_function=_record_tool_error)
def get_weather(city: str) -> str:
    raise RuntimeError("Weather service unavailable (HTTP 503)")
```

The `get_weather` span now shows up as **errored** in LangSmith (with the exception message/stacktrace) while the population/time spans stay green. Because the handler returns a string instead of re-raising, the agent still produces a final answer.

For self-hosted LangSmith, set `OTEL_EXPORTER_OTLP_ENDPOINT` to `<your-host>/api/v1/otel` (do not include `/v1/traces`; the exporter adds that suffix).

## Project layout

- `src/demo_agent/tools.py` — `@function_tool` definitions (including `get_weather`, which records errors on its trace span)
- `src/demo_agent/agent.py` — `Agent` configuration
- `src/demo_agent/otel.py` — OTEL export to LangSmith (GenAI instrumentor + OTLP exporter)
- `src/demo_agent/__main__.py` — entrypoint using `Runner.run_sync`
