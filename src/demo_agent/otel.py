import os

from agents.tracing import flush_traces
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.sdk.trace import (
    Event,
    ReadableSpan,
    Span,
    SpanProcessor,
    TracerProvider,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import StatusCode

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
    """Adapt GenAI semconv spans to LangSmith before export.

    Two adaptations happen here, ahead of the exporter, mirroring LangSmith's
    documented ``span._attributes`` editing pattern:

    1. Run-type mapping. LangSmith maps ``gen_ai.operation.name`` for
       ``chat``/``completion`` to ``llm`` and ``gen_ai.tool.name`` to ``tool``,
       but agent/workflow spans (``invoke_agent``, ``handoff``, ``guardrail``)
       may not become ``chain`` without an explicit ``langsmith.span.kind``.
       We add those hints and set run names from agent/tool attrs.

    2. Error capture. LangSmith derives a run's error/status from the OTEL
       ``exception`` event (``exception.message`` / ``exception.stacktrace``),
       *not* from the span's status code. The GenAI instrumentor sets an ERROR
       status (from the Agents SDK span's error, e.g. a tool that raised) but
       never records the event, so a failed tool call would otherwise look like
       a normal response. We synthesize the missing ``exception`` event for any
       span whose status is ERROR.
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

        self._ensure_exception_event(span)

    @staticmethod
    def _ensure_exception_event(span: ReadableSpan) -> None:
        """Add the ``exception`` event LangSmith needs for ERROR spans.

        The exporter reads ``span.events``; we append directly (as with
        ``_attributes``) so the synthesized event is serialized for export.
        """
        status = span.status
        if status is None or status.status_code is not StatusCode.ERROR:
            return
        if any(event.name == "exception" for event in span.events):
            return

        attributes = span._attributes or {}
        exception_type = attributes.get("error.type") or "ToolError"
        message = status.description or "Tool execution failed"
        span._events.append(
            Event(
                name="exception",
                attributes={
                    "exception.type": str(exception_type),
                    "exception.message": message,
                    "exception.escaped": "False",
                },
            )
        )

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
