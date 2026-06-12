import os

from agents.tracing import flush_traces
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_DEFAULT_OTLP_ENDPOINT = "https://api.smith.langchain.com/otel"

# GenAI semantic-convention attribute keys.
_GEN_AI_OPERATION = "gen_ai.operation.name"
_GEN_AI_AGENT_NAME = "gen_ai.agent.name"
_GEN_AI_TOOL_NAME = "gen_ai.tool.name"

# LangSmith OTEL attribute keys.
_LANGSMITH_SPAN_KIND = "langsmith.span.kind"
_LANGSMITH_TRACE_NAME = "langsmith.trace.name"

_OPERATION_TO_KIND = {
    "invoke_agent": "chain",
    "handoff": "chain",
    "guardrail": "chain",
    "create_agent": "chain",
    "chat": "llm",
    "text_completion": "llm",
    "execute_tool": "tool",
}


class _LangSmithGenAIMappingProcessor(SpanProcessor):
    """Map GenAI operation names to LangSmith run types before export.

    LangSmith maps ``gen_ai.operation.name`` for ``chat``/``completion`` to
    ``llm`` and ``gen_ai.tool.name`` to ``tool``, but agent/workflow spans
    (``invoke_agent``, ``handoff``, ``guardrail``) may not become ``chain``
    without an explicit ``langsmith.span.kind``. This processor (registered
    ahead of the exporter, mirroring LangSmith's documented ``span._attributes``
    editing pattern) adds those hints and sets run names from agent/tool attrs.
    """

    def on_start(
        self, span: Span, parent_context: Context | None = None
    ) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        attributes = span._attributes
        if not attributes:
            return

        operation = attributes.get(_GEN_AI_OPERATION)
        if isinstance(operation, str):
            kind = _OPERATION_TO_KIND.get(operation)
            if kind is not None:
                attributes[_LANGSMITH_SPAN_KIND] = kind

        if _LANGSMITH_TRACE_NAME not in attributes:
            tool_name = attributes.get(_GEN_AI_TOOL_NAME)
            agent_name = attributes.get(_GEN_AI_AGENT_NAME)
            if operation == "execute_tool" and isinstance(tool_name, str) and tool_name:
                attributes[_LANGSMITH_TRACE_NAME] = tool_name
            elif isinstance(agent_name, str) and agent_name:
                attributes[_LANGSMITH_TRACE_NAME] = agent_name
            elif isinstance(tool_name, str) and tool_name:
                attributes[_LANGSMITH_TRACE_NAME] = tool_name

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
    """Send OpenAI Agents SDK traces to LangSmith over OTLP (GenAI semconv)."""
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
    os.environ.setdefault(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "span_and_event"
    )

    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=10)
    provider = TracerProvider()
    # Order matters: map GenAI spans before the exporter serializes them.
    provider.add_span_processor(_LangSmithGenAIMappingProcessor())
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument(
        tracer_provider=provider,
        capture_message_content="span_and_event",
    )
    return provider, instrumentor


def shutdown_tracing(
    provider: TracerProvider, instrumentor: OpenAIAgentsInstrumentor
) -> None:
    """Flush Agents SDK spans, export OTEL spans, then tear down instrumentation."""
    flush_traces()
    provider.force_flush()
    instrumentor.uninstrument()
    provider.shutdown()
