#!/bin/bash

set -e  # Exit immediately on error

echo "======================================"
echo " Bhashini Models Auto Installation"
echo "======================================"

# --------------------------------------------------
# Resolve script directory (important)
# --------------------------------------------------
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $BASE_DIR

VENV_DIR="$BASE_DIR/venv"
PYTHON_VERSION=3.10

# --------------------------------------------------
# 1. Check Python
# --------------------------------------------------
echo "[1/5] Checking Python ${PYTHON_VERSION}..."

if ! command -v python${PYTHON_VERSION} &> /dev/null
then
    echo "❌ Python ${PYTHON_VERSION} not found. Install it first."
    exit 1
fi

# --------------------------------------------------
# 2. Create Virtual Environment
# --------------------------------------------------
echo "[2/5] Creating virtual environment..."

python${PYTHON_VERSION} -m venv $VENV_DIR

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# --------------------------------------------------
# 3. Install Flite
# --------------------------------------------------
echo "[3/5] Installing Flite..."

cd tts

if [ ! -d "flite" ]; then
    echo "Cloning Flite repository..."
    git clone http://github.com/festvox/flite
else
    echo "Flite repo already exists. Skipping clone."
fi

cd flite

echo "Configuring Flite..."
./configure

echo "Building Flite..."
make

echo "Downloading default voices..."
make get_voices

echo "Downloading Indic voices..."
./bin/get_voices indic_voices

echo "Flite setup completed successfully."

cd $BASE_DIR

# --------------------------------------------------
# 4. Validation
# --------------------------------------------------
echo "[4/5] Validating installation..."

echo "Flite version:"
$BASE_DIR/tts/flite/bin/flite --version || \
echo "Flite binary not found"

# --------------------------------------------------
# 5. Completion
# --------------------------------------------------
echo "======================================"
echo " Installation Completed Successfully "
echo "======================================"

echo ""
echo "To activate environment later run:"
echo "source venv/bin/activate"
