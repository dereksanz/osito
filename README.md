# Osito - Local Spanish Learning Voice Assistant

A friendly Spanish-speaking teddy bear voice assistant for children aged 3-6. Runs entirely locally on macOS or Raspberry Pi.

## Quick Start

```bash
./setup.sh
./run.sh
```

## Components

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Speech-to-Text | Whisper small | ~244MB | Transcribes child's speech (English/Spanish) |
| Language Model | Qwen2.5-3B via Ollama | ~2GB | Generates Spanish responses |
| Text-to-Speech | Piper TTS (sharvard) | ~63MB | Speaks responses in Spanish |

## Hardware Requirements

| Device | RAM | LLM Model | Notes |
|--------|-----|-----------|-------|
| MacBook (testing) | 8GB | qwen2.5:3b | Limited multitasking |
| **Raspberry Pi 5** | **16GB** | **qwen2.5:7b/14b** | **Recommended** |

## Usage

```
Osito: Hola! Soy Osito, tu amigo! Como te llamas?

Presiona Enter para hablar: [Enter]

Escuchando... (6 segundos)

Tu: "My name is Emma"
Osito: Hola Emma! Te gustan los colores?
  [Latency: STT 1.2s | LLM 2.3s | TTS 0.8s | Total 4.3s]
```

- Press **Enter** to record (6 seconds)
- Speak in English or Spanish
- Osito responds in simple Spanish
- Type `salir` to exit

## Configuration

Edit `osito.py`:

```python
WHISPER_MODEL = "small"       # tiny, base, small, medium, large
OLLAMA_MODEL = "qwen2.5:3b"   # qwen2.5:3b, qwen2.5:7b, qwen2.5:14b
RECORD_SECONDS = 6            # Recording duration
```

### Qwen2.5 Model Options

| Model | Size | RAM Needed | Use Case |
|-------|------|------------|----------|
| qwen2.5:1.5b | ~1GB | ~4GB | Very limited RAM |
| **qwen2.5:3b** | **~2GB** | **~5GB** | **8GB laptop (default)** |
| qwen2.5:7b | ~4.7GB | ~8GB | Pi 5 16GB |
| qwen2.5:14b | ~9GB | ~12GB | Pi 5 16GB (best) |

## Raspberry Pi 5 (16GB) Setup

```bash
# Edit osito.py
OLLAMA_MODEL = "qwen2.5:14b"

# Pull the model
ollama pull qwen2.5:14b
```

## Troubleshooting

### "No se puede conectar a Ollama"
```bash
ollama serve
```

### "Modelo no encontrado"
```bash
ollama pull qwen2.5:3b
```

### Language detection issues
- Speak clearly, closer to microphone
- Reduce background noise
- Increase `RECORD_SECONDS`

## Project Structure

```
osito/
├── osito.py           # Main application
├── pyproject.toml     # Dependencies
├── download_models.py # Piper voice downloader
├── setup.sh           # One-command setup
├── run.sh             # Run script
└── models/            # Downloaded TTS models
```

## License

MIT License
