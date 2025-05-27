import requests
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (OTLPSpanExporter)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (BatchSpanProcessor, ConsoleSpanExporter)
from opentelemetry.trace import SpanKind, set_tracer_provider
from opentelemetry.trace.propagation.tracecontext import (TraceContextTextMapPropagator)

trace_provider = TracerProvider()
set_tracer_provider(trace_provider)

trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

tracer = trace_provider.get_tracer("dummy-client")

vllm_url = "http://9.30.109.130:8000/v1/completions"
with tracer.start_as_current_span("client-span", kind=SpanKind.CLIENT) as span:
  prompt = "San Francisco is a"
  span.set_attribute("prompt", prompt)
  headers = {}
  TraceContextTextMapPropagator().inject(headers)
  payload = {
    "model": "ibm-granite/granite-3.0-2b-instruct",
    "prompt": prompt,
    "max_tokens": 10,
    "n": 1,
    "best_of": 1,
    "use_beam_search": "false",
    "temperature": 0.0,
  }
  response = requests.post(vllm_url, headers=headers, json=payload)
  print(response)