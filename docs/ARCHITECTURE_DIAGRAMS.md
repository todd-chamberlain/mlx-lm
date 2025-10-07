# Architecture Diagrams

Visual representations of the MLX Distributed Inference System architecture.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                          CLIENT APPLICATIONS                            │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │   Web App    │  │  Mobile App  │  │  CLI Tool    │                │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                │
│         │                  │                  │                         │
│         └──────────────────┴──────────────────┘                         │
│                            │                                            │
│                            │ HTTP/REST (OpenAI Compatible)              │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    FastAPI Server (Port 8000)                    │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │
│  │  │     Auth     │  │   Streaming  │  │ Rate Limiting│          │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │  │
│  └─────────────────────────────┬───────────────────────────────────┘  │
│                                │ gRPC                                  │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │              Coordinator Service (Port 50051)                    │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │
│  │  │Worker Manager│  │Request Router│  │Cache Coord   │          │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │
│  │  │Batch Scheduler│ │Health Monitor│  │Load Balancer │          │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │  │
│  └─────────┬──────────────┬──────────────┬────────────────────────┘  │
│            │ gRPC         │ gRPC         │ gRPC                       │
│            ▼              ▼              ▼                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  Worker 1   │  │  Worker 2   │  │  Worker N   │                  │
│  │             │  │             │  │             │                  │
│  │ Layers 0-N  │  │ Layers N-M  │  │ Layers M-L  │                  │
│  │             │  │             │  │             │                  │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │                  │
│  │ │MLX Engine│ │  │ │MLX Engine│ │  │ │MLX Engine│ │                  │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │                  │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │                  │
│  │ │Metal GPU│ │  │ │Metal GPU│ │  │ │Metal GPU│ │                  │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
│    Mac Studio 1     Mac Studio 2     Mac Studio N                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Request Flow

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ 1. POST /v1/chat/completions
     ▼
┌─────────────────┐
│   API Server    │
│  (Port 8000)    │
└────┬────────────┘
     │ 2. Authenticate & Validate
     │ 3. Forward to Coordinator (gRPC)
     ▼
┌─────────────────┐
│  Coordinator    │
│  (Port 50051)   │
└────┬────────────┘
     │ 4. Select Workers
     │ 5. Route Request
     ▼
┌─────────────────┐
│   Worker 1      │  6. Process Layers 0-N
│  (Layers 0-N)   │  7. Generate Activations
└────┬────────────┘
     │ 8. Transfer Activations + Cache
     ▼
┌─────────────────┐
│   Worker 2      │  9. Process Layers N-M
│  (Layers N-M)   │  10. Generate Activations
└────┬────────────┘
     │ 11. Transfer Activations + Cache
     ▼
┌─────────────────┐
│   Worker N      │  12. Process Layers M-L
│  (Layers M-L)   │  13. Generate Final Output
└────┬────────────┘
     │ 14. Return Tokens
     ▼
┌─────────────────┐
│  Coordinator    │  15. Aggregate Results
└────┬────────────┘
     │ 16. Return Response
     ▼
┌─────────────────┐
│   API Server    │  17. Format Response
└────┬────────────┘
     │ 18. Stream Tokens (SSE)
     ▼
┌──────────┐
│  Client  │  19. Receive Tokens
└──────────┘
```

