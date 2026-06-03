import json
import os

from agents.tracing import flush_traces
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_DEFAULT_OTLP_ENDPOINT = "https://api.smith.langchain.com/otel"

# OpenInference semantic-convention attribute keys.
_SPAN_KIND = "openinference.span.kind"
_LLM_KIND = "LLM"
_INPUT_VALUE = "input.value"
_OUTPUT_VALUE = "output.value"


def _clean_messages(items: list) -> list[dict]:
    """Reduce OpenAI Responses-API items to readable chat messages."""
    messages: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in (None, "message"):
            content = item.get("content")
            if isinstance(content, list):
                text = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
            else:
                text = content
            messages.append(
                {"role": item.get("role", "assistant"), "content": text}
            )
        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "name": item.get("name"),
                            "arguments": item.get("arguments"),
                            "call_id": item.get("call_id"),
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "call_id": item.get("call_id"),
                    "content": item.get("output"),
                }
            )
    return messages


class _CleanLLMIOProcessor(SpanProcessor):
    """Rewrite OpenInference's raw LLM I/O into readable messages before export.

    OpenInference stores ``input.value`` as a stringified request list and
    ``output.value`` as the full OpenAI Responses object (billing, timestamps,
    store flags, …). LangSmith renders those verbatim, so LLM spans look like an
    unreadable blob. This processor (registered ahead of the exporter, mirroring
    LangSmith's documented ``span._attributes`` editing pattern) replaces them
    with a compact ``{"messages": [...]}`` shape that LangSmith renders cleanly.
    """

    def on_start(
        self, span: Span, parent_context: Context | None = None
    ) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        attributes = span._attributes
        if not attributes or attributes.get(_SPAN_KIND) != _LLM_KIND:
            return

        raw_input = attributes.get(_INPUT_VALUE)
        if isinstance(raw_input, str):
            try:
                parsed = json.loads(raw_input)
                items = parsed if isinstance(parsed, list) else [parsed]
                attributes[_INPUT_VALUE] = json.dumps(
                    {"messages": _clean_messages(items)}
                )
            except (ValueError, TypeError):
                pass

        raw_output = attributes.get(_OUTPUT_VALUE)
        if isinstance(raw_output, str):
            try:
                response = json.loads(raw_output)
                output_items = response.get("output", []) or []
                attributes[_OUTPUT_VALUE] = json.dumps(
                    {"messages": _clean_messages(output_items)}
                )
            except (ValueError, TypeError, AttributeError):
                pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _otlp_trace_endpoint() -> str:
    """Resolve the OTLP/HTTP traces URL LangSmith expects.

    The HTTP exporter appends ``/v1/traces`` when the endpoint is read from
    ``OTEL_EXPORTER_OTLP_ENDPOINT``, but not when passed to the constructor.
    """
    endpoint = os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", _DEFAULT_OTLP_ENDPOINT
    ).rstrip("/")
    if endpoint.endswith("/v1/traces"):
        return endpoint
    return f"{endpoint}/v1/traces"


def setup_tracing() -> tuple[TracerProvider, OpenAIAgentsInstrumentor]:
    """Send OpenAI Agents SDK traces to LangSmith over OTLP."""
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        raise SystemExit("LANGSMITH_API_KEY is not set.")

    project = os.environ.get("LANGSMITH_PROJECT", "demo-agent")
    endpoint = _otlp_trace_endpoint()
    headers = {
        "x-api-key": api_key,
        "Langsmith-Project": project,
    }

    # Keep env vars in sync for any library that reads them directly.
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", _DEFAULT_OTLP_ENDPOINT)
    os.environ.setdefault(
        "OTEL_EXPORTER_OTLP_HEADERS",
        f"x-api-key={api_key},Langsmith-Project={project}",
    )

    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=10)
    provider = TracerProvider()
    # Order matters: clean up LLM spans before the exporter serializes them.
    provider.add_span_processor(_CleanLLMIOProcessor())
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    return provider, instrumentor


def shutdown_tracing(
    provider: TracerProvider, instrumentor: OpenAIAgentsInstrumentor
) -> None:
    """Flush Agents SDK spans, export OTEL spans, then tear down instrumentation."""
    flush_traces()
    provider.force_flush()
    instrumentor.uninstrument()
    provider.shutdown()
