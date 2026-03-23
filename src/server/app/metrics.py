from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

rag_requests_total = Counter("rag_requests_total", "Total number of RAG requests")
rag_fallback_total = Counter("rag_fallback_total", "Total number of RAG fallback responses")
rag_errors_total = Counter("rag_errors_total", "Total number of RAG errors", ["stage"])
rag_retrieval_seconds = Histogram(
    "rag_retrieval_seconds",
    "Retrieval latency in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
rag_generation_seconds = Histogram(
    "rag_generation_seconds",
    "Generation latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
)
rag_total_seconds = Histogram(
    "rag_total_seconds",
    "End-to-end RAG latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30),
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
