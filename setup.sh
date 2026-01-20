#!/bin/bash
# =============================================================================
# Osito - One-Command Setup Script
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}"
echo "=============================================="
echo "       OSITO - Setup Script"
echo "=============================================="
echo -e "${NC}"

# -----------------------------------------------------------------------------
# macOS check
# -----------------------------------------------------------------------------
if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}Error: This script is designed for macOS${NC}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Homebrew
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/6] Checking Homebrew...${NC}"
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    [[ -f /opt/homebrew/bin/brew ]] && eval "$(/opt/homebrew/bin/brew shellenv)"
else
    echo -e "${GREEN}  Homebrew ready${NC}"
fi

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/6] Installing system dependencies...${NC}"
brew list portaudio &>/dev/null || brew install portaudio
brew list ffmpeg &>/dev/null || brew install ffmpeg
echo -e "${GREEN}  System deps ready${NC}"

# -----------------------------------------------------------------------------
# Ollama
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/6] Installing Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo "  Installing Ollama..."
    brew install ollama
else
    echo -e "${GREEN}  Ollama already installed${NC}"
fi

# Start Ollama service if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "  Starting Ollama service..."
    ollama serve &>/dev/null &
    sleep 3
fi

# Pull Qwen model
echo "  Pulling qwen2.5:3b model (~2GB)..."
ollama pull qwen2.5:3b

echo -e "${GREEN}  Ollama ready with qwen2.5:3b${NC}"

# -----------------------------------------------------------------------------
# uv
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/6] Installing uv...${NC}"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
else
    echo -e "${GREEN}  uv ready${NC}"
fi
export PATH="$HOME/.local/bin:$PATH"

# -----------------------------------------------------------------------------
# Python packages
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[5/6] Installing Python packages...${NC}"

cd "$SCRIPT_DIR"

export CFLAGS="-I$(brew --prefix portaudio)/include"
export LDFLAGS="-L$(brew --prefix portaudio)/lib"

uv sync --python 3.11

echo -e "${GREEN}  Python packages ready${NC}"

# -----------------------------------------------------------------------------
# Piper voice
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[6/6] Downloading Piper voice...${NC}"

mkdir -p "$SCRIPT_DIR/models"
uv run python download_models.py --piper-only

# -----------------------------------------------------------------------------
# Create run script
# -----------------------------------------------------------------------------
cat > "$SCRIPT_DIR/run.sh" << 'RUNEOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PIPER_VOICE="$SCRIPT_DIR/models/es_ES-davefx-medium.onnx"

# Ensure Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 2
fi

cd "$SCRIPT_DIR"
uv run python osito.py
RUNEOF

chmod +x "$SCRIPT_DIR/run.sh"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}=============================================="
echo "       SETUP COMPLETE!"
echo "==============================================${NC}"
echo ""
echo "Run Osito:"
echo -e "  ${BLUE}./run.sh${NC}"
echo ""
