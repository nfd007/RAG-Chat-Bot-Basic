# ADR-007: Use ChromaDB as the vector store for the knowledge assistant

- Status: Accepted
- Date: 2026-03-14
- Deciders: Arif Chowdhury (Tech Lead), Nusrat Jahan (Platform), Imran Kabir (Security)

## Context

The internal knowledge assistant needs a vector store for document chunks. Phase 1 is
expected to hold about 120,000 chunks. The retrieval latency target is p95 under 200 ms,
measured end to end from query embedding to returned results. Security requires that no
document content leaves our own infrastructure, which rules out hosted vector services.

## Options considered

1. PostgreSQL with pgvector
2. Qdrant (self-hosted, separate service)
3. ChromaDB (embedded, persistent mode)

## Benchmark

We indexed 100,000 chunks embedded with nomic-embed-text (768 dimensions) and ran 1,000
queries on the standard 8 vCPU / 32 GB build server.

| Option   | p95 query latency | Extra services to operate |
|----------|-------------------|---------------------------|
| pgvector | 71 ms             | 0 (existing Postgres)     |
| Qdrant   | 29 ms             | 1                         |
| ChromaDB | 38 ms             | 0                         |

## Decision

We will use ChromaDB in embedded persistent mode for Phase 1. It meets the latency target
with no new service to operate. Qdrant was faster but the team did not want to run an
extra stateful service for a Phase 1 workload.

We will revisit this decision when the collection exceeds 1 million vectors or when we
need multiple application servers writing to the same index.

## Consequences

- The index is a single directory on one node. It is backed up nightly at 01:30 by
  snapshotting the chroma_db directory; snapshots are kept for 14 days.
- Changing the embedding model changes the vector dimensions, so it requires a full
  re-index of every document.
- Distance metric is cosine, set when the collection is created.
