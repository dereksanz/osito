#!/usr/bin/env python3
"""
Osito - Local Spanish Learning Voice Assistant

A friendly Spanish-speaking teddy bear voice assistant for children aged 3-5.
Runs entirely locally using:
- Whisper (Speech-to-Text)
- gpt-oss-20b via Ollama (Language Model)
- Piper TTS (Text-to-Speech)
"""

import os
import sys
import re
import time
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pyaudio
import whisper
import ollama

# =============================================================================
# Configuration
# =============================================================================

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
RECORD_SECONDS = 6
AUDIO_FORMAT = pyaudio.paInt16

WHISPER_MODEL = "small"

OLLAMA_MODEL = "qwen2.5:3b"
LLM_MAX_TOKENS = 40
LLM_TEMPERATURE = 0.7

PIPER_PATH = os.environ.get("PIPER_PATH", "piper")
PIPER_VOICE = os.environ.get(
    "PIPER_VOICE",
    str(Path.home() / ".local" / "share" / "piper" / "es_ES-sharvard-medium.onnx")
)

# Conversation history settings
MAX_HISTORY_TURNS = 4

SYSTEM_PROMPT = """You are Osito, a friendly teddy bear teaching Spanish to 4-year-old children.

STRICT RULES:
- Respond ONLY in simple Spanish (never English)
- Maximum 8 words per response
- End with ONE simple question
- Use vocabulary a 4-year-old understands
- NEVER use emojis
- Remember the child's name and use it

GOOD EXAMPLES:
- "Hola Ana! Te gustan los perros?"
- "Azul! Muy lindo! Te gusta rojo?"
- "Uno, dos, tres! Puedes contar?"
- "Tengo hambre tambien! Te gusta pizza?"

TOPICS: colors, animals, food, numbers 1-5, family.
FORBIDDEN: emojis, long sentences, difficult words, English in responses."""


# =============================================================================
# Model Loading
# =============================================================================

def load_whisper_model():
    """Load Whisper speech-to-text model."""
    print("Cargando modelo Whisper...")
    model = whisper.load_model(WHISPER_MODEL)
    print(f"  Whisper ({WHISPER_MODEL}) cargado.")
    return model


def check_ollama():
    """Verify Ollama is running and model is available."""
    print("Verificando Ollama...")

    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
        if OLLAMA_MODEL not in model_names and f"{OLLAMA_MODEL}:latest" not in model_names:
            print(f"\n  ERROR: Modelo {OLLAMA_MODEL} no encontrado")
            print("  Ejecuta: ollama pull gpt-oss:20b")
            sys.exit(1)
        print(f"  Ollama listo con {OLLAMA_MODEL}")
    except Exception as e:
        print(f"\n  ERROR: No se puede conectar a Ollama: {e}")
        print("  Asegurate de que Ollama este corriendo: ollama serve")
        sys.exit(1)


def check_piper_installation():
    """Verify Piper TTS is available."""
    print("Verificando Piper TTS...")

    try:
        result = subprocess.run(
            [PIPER_PATH, "--help"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"\n  ERROR: Piper no encontrado en: {PIPER_PATH}")
        sys.exit(1)

    if not Path(PIPER_VOICE).exists():
        print(f"\n  ERROR: Voz no encontrada en: {PIPER_VOICE}")
        sys.exit(1)

    print(f"  Piper TTS listo con voz: {Path(PIPER_VOICE).name}")


def load_models():
    """Load all required models."""
    print("\n" + "=" * 50)
    print("Cargando modelos...")
    print("=" * 50 + "\n")

    whisper_model = load_whisper_model()
    check_ollama()
    check_piper_installation()

    print("\n" + "=" * 50)
    print("Modelos cargados.")
    print("=" * 50 + "\n")

    return whisper_model


# =============================================================================
# Audio Recording
# =============================================================================

def record_audio():
    """Record audio from the built-in microphone."""
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print(f"Escuchando... ({RECORD_SECONDS} segundos)")

    frames = []
    num_chunks = int(SAMPLE_RATE / CHUNK_SIZE * RECORD_SECONDS)

    for _ in range(num_chunks):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio_float = audio_data.astype(np.float32) / 32768.0

    return audio_float


# =============================================================================
# Speech-to-Text
# =============================================================================

def transcribe(whisper_model, audio_data):
    """Transcribe audio - only accepts English or Spanish."""
    try:
        # First, detect the language
        audio_padded = whisper.pad_or_trim(audio_data)
        mel = whisper.log_mel_spectrogram(audio_padded).to(whisper_model.device)
        _, probs = whisper_model.detect_language(mel)

        # Get top language
        detected_lang = max(probs, key=probs.get)
        confidence = probs[detected_lang]

        # Only accept English or Spanish
        if detected_lang not in ("en", "es"):
            print(f"  [Idioma detectado: {detected_lang} ({confidence:.0%})]")
            return None, "unsupported_language"

        # Transcribe with the detected language
        result = whisper_model.transcribe(
            audio_data,
            language=detected_lang,
            task="transcribe",
            fp16=False,
            initial_prompt="Hola, como estas?" if detected_lang == "es" else "Hello, how are you?",
        )

        text = result["text"].strip()

        # Filter hallucinations
        hallucinations = [
            "Gracias por ver", "Suscribete", "subtitulos",
            "MBC", "Thank you for watching", "Subscribe",
        ]
        for h in hallucinations:
            if h.lower() in text.lower() and len(text) < 40:
                return None, "no_speech"

        if not text or len(text) < 2:
            return None, "no_speech"

        return text, detected_lang

    except (RuntimeError, ValueError, KeyError) as e:
        print(f"  Error en transcripcion: {e}")
        return None, "error"


# =============================================================================
# LLM Response Generation
# =============================================================================

def strip_emojis(text):
    """Remove emojis from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended
        "\U00002600-\U000026FF"  # misc symbols
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text).strip()


def generate_response(user_text: str, conversation_history: list):
    """Generate Osito's response using Ollama with conversation history."""
    try:
        # Build messages with system prompt + history + current message
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_text})

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "num_predict": LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE,
            },
        )

        response_text = response["message"]["content"].strip()

        # Clean up
        for prefix in ["Osito:", "osito:", "**"]:
            if response_text.startswith(prefix):
                response_text = response_text[len(prefix):].strip()
        response_text = response_text.replace("**", "").replace("*", "")

        # Remove emojis
        response_text = strip_emojis(response_text)

        # Clean up extra spaces
        response_text = re.sub(r'\s+', ' ', response_text).strip()

        if not response_text:
            response_text = "Hola! Como estas?"

        return response_text

    except Exception as e:
        print(f"  Error generando respuesta: {e}")
        return "Hola! Que quieres decirme?"


# =============================================================================
# Text-to-Speech
# =============================================================================

def speak(text):
    """Convert text to speech using Piper TTS."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        process = subprocess.run(
            [PIPER_PATH, "--model", PIPER_VOICE, "--output_file", tmp_path],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
            check=False,
        )

        if process.returncode != 0:
            raise RuntimeError(f"Piper error: {process.stderr.decode()}")

        subprocess.run(["afplay", tmp_path], check=True, timeout=30)
        os.unlink(tmp_path)

    except subprocess.TimeoutExpired:
        print(f"  [Audio]: {text}")
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        print(f"  (Error TTS: {e})")
        print(f"  [Audio]: {text}")


# =============================================================================
# Main Loop
# =============================================================================

def print_header():
    """Print the welcome header."""
    print("\n" + "=" * 50)
    print("      OSITO - Tu Amigo en Espanol")
    print("=" * 50)
    print("\n  Habla en espanol o ingles.")
    print("  Osito siempre responde en espanol.\n")
    print("  [Enter] = Hablar")
    print("  [salir] = Terminar\n")
    print("=" * 50 + "\n")


def main_loop(whisper_model):
    """Main conversation loop."""
    print_header()

    # Conversation history (stores user/assistant message pairs)
    conversation_history = []

    # Initial greeting
    greeting = "Hola! Soy Osito, tu amigo! Como te llamas?"
    print(f"Osito: {greeting}\n")
    speak(greeting)

    while True:
        user_input = input("Presiona Enter para hablar: ").strip().lower()

        if user_input == "salir":
            bye = "Adios amigo! Hasta pronto!"
            print(f"\nOsito: {bye}")
            speak(bye)
            break

        print()
        audio_data = record_audio()
        print()

        # Track latency
        turn_start = time.time()

        print("Procesando...")
        stt_start = time.time()
        text, status = transcribe(whisper_model, audio_data)
        stt_time = time.time() - stt_start

        if status == "unsupported_language":
            msg = "Hmm, solo entiendo espanol e ingles. Try again!"
            print(f"Osito: {msg}\n")
            speak(msg)
            continue

        if text is None:
            msg = "No te escuche bien. Can you say that again?"
            print(f"Osito: {msg}\n")
            speak(msg)
            continue

        print(f"Tu: \"{text}\"")

        llm_start = time.time()
        response = generate_response(text, conversation_history)
        llm_time = time.time() - llm_start

        print(f"Osito: {response}")

        tts_start = time.time()
        speak(response)
        tts_time = time.time() - tts_start

        total_time = time.time() - turn_start
        print(f"  [Latency: STT {stt_time:.1f}s | LLM {llm_time:.1f}s | TTS {tts_time:.1f}s | Total {total_time:.1f}s]\n")

        # Add this exchange to history
        conversation_history.append({"role": "user", "content": text})
        conversation_history.append({"role": "assistant", "content": response})

        # Keep only the last N turns (each turn = 2 messages)
        max_messages = MAX_HISTORY_TURNS * 2
        if len(conversation_history) > max_messages:
            conversation_history = conversation_history[-max_messages:]


def main():
    """Main entry point."""
    try:
        whisper_model = load_models()
        main_loop(whisper_model)

    except KeyboardInterrupt:
        print("\n\nAdios!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
