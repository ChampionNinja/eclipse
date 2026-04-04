# STEP 1: System Architecture

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                          │
│  Mic Capture ──▶ WebSocket Stream ──▶ Audio Player (TTS)  │
└─────────────────────┬─────────────────────────────────────┘
                      │ audio bytes / text
                      ▼
┌───────────────────────────────────────────────────────────┐
│                  GATEWAY (FastAPI)                         │
│  STT (Whisper tiny) ──▶ Rules Layer ──▶ Router (Intent)   │
└─────────────────────────────┬─────────────────────────────┘
            ┌─────────────────┼──────────────────┐
            ▼                 ▼                  ▼
     BARCODE FLOW      NON-BARCODE FLOW    CASUAL FLOW
     (OFF API)         (Food Resolver)     (Templates)
            │                 │
            └────────┬────────┘
                     ▼
          UNIFIED PRODUCT SCHEMA
                     │
                     ▼
         PRODUCT ANALYSIS AGENT (SLM)
         Qwen2.5-0.5B/1.5B (4-bit)
                     │
                     ▼
          RESPONSE FORMATTER + TTS
```

## Layer Definitions

| Layer | Technology | Latency Budget | Purpose |
|-------|-----------|---------------|---------|
| STT | Whisper tiny/base (ONNX) | ~200ms | Speech → text |
| Rules Layer | Regex + heuristics | ~5ms | Fast-path classification |
| Router | Keyword scorer / zero-shot | ~50ms | Intent when rules fail |
| Data Layer | OFF API + local JSON cache | ~200ms | Product data retrieval |
| Analysis Agent | Qwen2.5-0.5B fine-tuned 4-bit | ~300ms | Nutritional verdict |
| Response Gen | Template + formatting | ~50ms | Human-friendly text |
| TTS | edge-tts / Piper | ~200ms | Text → speech |
| **Total** | | **~1000ms** | Under 1.5s target |

## Data Flow

```
Audio → STT(text) → Rules(fast) → Router(intent)
  → [barcode]: OFF API ──────────────┐
  → [food_query]: Food Resolver ─────┤
  → [casual]: template → TTS         │
                                      ▼
                            Unified Schema → SLM → Verdict → TTS → Audio
```
