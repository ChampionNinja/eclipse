<p align="center">
  <img src="frontend/logo.png" alt="Bite.ai Logo" width="120">
</p>

<h1 align="center">Bite.ai</h1>
<p align="center">
  <strong>AI-Powered Voice Food Assistant</strong><br>
  <em>Ask about any food. Scan any barcode. Get instant nutritional verdicts.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Ollama-Local_SLM-black?logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/Oumi-Powered-D88A73" alt="Powered by Oumi">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
</p>

---

## 📸 Screenshots

<!-- Add your screenshots here -->

| Home Screen | Voice Call Mode | Barcode Scanner |
|:-----------:|:---------------:|:---------------:|
| [Home](/home.png) | [Voice](/voice_call.png) | [Barcode](/barcode_scanner.png) |

| Analysis Result | Profile Setup | Mobile View |
|:--------------:|:-------------:|:-----------:|
| [Result](screenshots/analysis_result.png) | [Profile](/profile.png) | [Mobile](screenshots/mobile.png) |

---

## 🎬 Demo Video

<!-- Add your demo video here -->

[[Watch the Demo](https://www.youtube.com/playlist?list=PL6tiFuZ8NQQuWYiFOFMxbcHK5nShAFAkJ)]

> *Click the thumbnail above to watch the full demo, or see the video file at [`demo/bite_ai_demo.mp4`](demo/bite_ai_demo.mp4)*

---

## ✨ Features

- **🎙️ Voice Call Mode** — Tap the orb or mic to start a continuous voice conversation. Bite.ai listens, processes, speaks back, and auto-listens again — like a real phone call.
- **📷 Barcode Scanner** — Scan any product barcode using your phone camera (supports multiple cameras) or type it manually. Fetches real nutrition data from Open Food Facts.
- **🧠 Hybrid Analysis Engine** — Rule-based analyzer delivers instant, accurate verdicts. Ollama-hosted SLM provides natural language enhancement as a fallback.
- **👤 Personalized Profiles** — Set your allergies (peanut, lactose, gluten, fish, soy), health conditions (diabetes, hypertension, cholesterol), and dietary preferences for tailored advice.
- **🔮 Interactive 3D Orb** — A living, breathing Three.js orb that changes color and animation based on state: idle → listening → processing → speaking → verdict (eat/avoid/sometimes).
- **📱 Mobile-First Design** — Fully responsive, works on phone browsers over local WiFi. Camera, voice, and touch — all native.
- **🗣️ Browser TTS/STT** — No external APIs needed. Uses the Web Speech API for speech recognition and synthesis, with automatic voice selection for natural output.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                  CLIENT (Browser)                 │
│  Voice (STT) ──▶ Text Input ──▶ TTS (Speak)      │
│  Barcode Camera ──▶ Manual Entry                  │
│  3D Orb (Three.js) ──▶ State Visualization        │
└───────────────────────┬──────────────────────────┘
                        │ HTTP POST /query
                        ▼
┌──────────────────────────────────────────────────┐
│               GATEWAY (FastAPI)                   │
│  Intent Classifier ──▶ Rules Layer + Router       │
└──────┬──────────┬──────────┬─────────────────────┘
       │          │          │
       ▼          ▼          ▼
  BARCODE     FOOD QUERY   CASUAL
  (OFF API)   (INDB 1014   (Templates)
               foods)
       │          │
       └────┬─────┘
            ▼
   AGENT 1: Rule-Based Analyzer
   (Instant nutritional verdicts)
            │
            ▼
   AGENT 2: Response Generator
   (Templates primary + Ollama SLM fallback)
            │
            ▼
       Browser TTS
```

### Two-Agent Pipeline

| Agent | Role | Technology | Latency |
|-------|------|-----------|---------|
| **Agent 1** — Analysis | Nutritional verdict (eat/avoid/sometimes) | Rule-based engine + Ollama SLM  | ~5ms (rules) / ~1.2s (SLM) |
| **Agent 2** — Response | Natural language generation | Templates (primary) + Ollama SLM  | ~1ms (template) / ~1.3s (SLM) |

---

## 📁 Project Structure

```
bite.ai/
├── app/                          # FastAPI backend
│   ├── main.py                   # App entry point, routes, lifespan
│   ├── models/                   # Pydantic schemas
│   └── services/
│       ├── intent.py             # Intent classifier (Rules + Router)
│       ├── resolver.py           # Food resolver (INDB Excel → fuzzy match)
│       ├── analyzer.py           # Agent 1: Hybrid rule-based + SLM analyzer
│       ├── barcode.py            # Open Food Facts API client
│       ├── response.py           # Template-based response formatter
│       └── response_agent.py     # Agent 2: Template + Ollama SLM responses
│
├── frontend/                     # Browser-based UI
│   ├── index.html                # Single-page app
│   ├── styles.css                # Design system (warm aesthetic)
│   ├── app.js                    # Voice agent, API calls, UI logic
│   ├── orb.js                    # Three.js 3D orb with state animations
│   └── logo.png                  # App logo / favicon
│
├── data/                         # Datasets
│   ├── INDB_data.xlsx            # Indian Nutrient Database (1014 foods)
│   └── training_data.jsonl       # Fine-tuning dataset
│
├── docs/                         # Architecture & design docs
│   ├── 01_architecture.md
│   ├── 02_intent_classification.md
│   ├── 03_barcode_flow.md
│   ├── 04_nonbarcode_flow.md
│   ├── 05_unified_schema_and_agent.md
│   ├── 06_dataset_design.md
│   ├── 07_finetuning_and_inference.md
│   ├── 08_conversation_and_optimization.md
│   └── 09_backend_and_failure_handling.md
│
├── notebooks/                    # Training notebooks
│   └── colab_train.py            # Google Colab fine-tuning script
│
├── scripts/                      # Utility scripts
│   ├── generate_dataset.py       # Dataset augmentation
│   └── generate_dataset_fin.py   # Final dataset generator
│
├── configs/                      # Configuration files
├── requirements.txt              # Python dependencies
└── README.md                     # ← You are here
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11** — [Download](https://www.python.org/downloads/release/python-3110/)
- **Ollama** — [Download](https://ollama.com/download) (for SLM inference)
- **Chrome/Edge** — Required for voice features (Web Speech API)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/bite-ai.git
cd bite-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Ollama

```bash
# Install and start Ollama
ollama serve

# Pull the model (in a new terminal)
ollama pull qwen:1.8b
```

### 3. Run the Server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open in Browser

```
http://localhost:8000/app/
```

### 5. Access from Phone (Same WiFi)

```bash
# Find your PC's IP
ipconfig    # Windows — look for IPv4 under Wi-Fi adapter

# On your phone browser:
http://YOUR_PC_IP:8000/app/
```

---

## 🎙️ How to Use

### Voice Mode
1. Tap the **Voice** button or click the **orb** to start a voice call
2. Ask naturally: *"Is idli healthy?"*, *"Can I eat paneer?"*, *"What about dosa?"*
3. Bite.ai speaks the answer and auto-listens for your next question
4. Tap again to end the call

### Text Mode
Type any food query in the text box:
- `is biryani healthy`
- `can diabetics eat rice`
- `tell me about samosa`
- `gulab jamun` (just the food name works too)

### Barcode Scanner
1. Tap **Barcode** → use camera or type the number
2. Supports multiple cameras (switch via dropdown)
3. Works with EAN-13, EAN-8, UPC-A, UPC-E, Code-128, Code-39

### Profile Setup
1. Tap the **Profile** icon (top right)
2. Set your name, allergies, health conditions, and diet type
3. All verdicts are personalized to your profile

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | FastAPI + Uvicorn | API server, async request handling |
| **Analysis** | Rule-based engine | Instant nutritional verdicts (95% accuracy) |
| **SLM** | Ollama (qwen:1.8b) | Natural language fallback / enhancement |
| **Food Data** | INDB Excel (1014 foods) | Indian nutritional database with fuzzy matching |
| **Barcode** | Open Food Facts API | Global product database (2M+ products) |
| **Frontend** | Vanilla HTML/CSS/JS | No framework dependencies |
| **3D Orb** | Three.js | Interactive state visualization |
| **Voice** | Web Speech API | Browser-native STT + TTS |
| **Fine-tuning** | Oumi + QLoRA | Model training on Google Colab |

---

## 📊 Supported Food Queries

| Query Type | Examples | Coverage |
|-----------|---------|----------|
| Indian foods | idli, dosa, biryani, paneer, samosa, gulab jamun | 1014 foods from INDB |
| Barcode products | KitKat, Coca-Cola, Maggi, any packaged product | 2M+ via Open Food Facts |
| Natural language | "is X healthy?", "can I eat X?", "X for diabetics?" | Flexible pattern matching |
| Greetings | "hi", "hello", "thanks", "bye" | Template responses |

---

## 🧪 Fine-Tuning (Optional)

The system works out-of-the-box with the rule-based engine. For custom SLM training:

1. **Generate dataset**: `python scripts/generate_dataset_fin.py`
2. **Train on Colab**: Upload `notebooks/colab_train.py` to Google Colab
3. **Base model**: Qwen2.5-0.5B-Instruct with QLoRA (4-bit)
4. **Framework**: [Oumi](https://github.com/oumi-ai/oumi) for training orchestration
5. **Export**: GGUF quantization for Ollama deployment

See [`docs/07_finetuning_and_inference.md`](docs/07_finetuning_and_inference.md) for the full training guide.

---

## 📚 Documentation

| Document | Description |
|----------|------------|
| [01 — Architecture](docs/01_architecture.md) | System architecture, layer definitions, data flow |
| [02 — Intent Classification](docs/02_intent_classification.md) | Rules layer, keyword router, latency strategy |
| [03 — Barcode Flow](docs/03_barcode_flow.md) | OFF API client, cache, follow-up queries |
| [04 — Non-Barcode Flow](docs/04_nonbarcode_flow.md) | Food resolver, fuzzy matching, INDB lookup |
| [05 — Unified Schema & Agent](docs/05_unified_schema_and_agent.md) | Pydantic models, SLM prompt design, output parsing |
| [06 — Dataset Design](docs/06_dataset_design.md) | Training data structure, augmentation strategies |
| [07 — Fine-Tuning & Inference](docs/07_finetuning_and_inference.md) | QLoRA config, training script, inference pipeline |
| [08 — Conversation & Optimization](docs/08_conversation_and_optimization.md) | Session management, context handling |
| [09 — Backend & Error Handling](docs/09_backend_and_failure_handling.md) | Failure modes, fallbacks, graceful degradation |

---

## 🔧 Configuration

### Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `SLM_MODEL` | `qwen:1.8b` | Ollama model for response generation |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server bind address |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Bite.ai</strong> — Powered by <strong>Oumi</strong><br>
  <em>Built with ❤️ for healthier food choices</em>
</p>
