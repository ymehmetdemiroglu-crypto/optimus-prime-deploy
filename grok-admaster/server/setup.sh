#!/bin/bash
# Grok-AdMaster Setup Script (Linux/Mac/Git Bash)
# This script automates the first-time setup process

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Grok-AdMaster Setup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Python is installed
echo -e "${YELLOW}[1/7]${NC} Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed!${NC}"
    echo ""
    echo "Please install Python 3.11 or newer:"
    echo "  - Linux: sudo apt install python3 python3-pip python3-venv"
    echo "  - Mac: brew install python@3.11"
    echo "  - Windows: Download from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"

# Check Python version (must be 3.8+)
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}ERROR: Python 3.8 or newer is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

# Check if .env file exists
echo ""
echo -e "${YELLOW}[2/7]${NC} Checking environment configuration..."
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠${NC}  .env file not found, copying from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}⚠${NC}  Please edit .env with your actual credentials before continuing!"
    echo ""
    echo "Required variables:"
    echo "  - SECRET_KEY (generate with: openssl rand -base64 32)"
    echo "  - POSTGRES_PASSWORD"
    echo "  - DATABASE_URL"
    echo ""
    read -p "Press Enter after updating .env or Ctrl+C to exit..."
fi

# Validate critical env vars
if ! grep -q "SECRET_KEY=" .env || grep -q "SECRET_KEY=your-secret-key-here" .env; then
    echo -e "${RED}ERROR: SECRET_KEY not set in .env file!${NC}"
    echo "Generate one with: openssl rand -base64 32"
    exit 1
fi

if ! grep -q "POSTGRES_PASSWORD=" .env; then
    echo -e "${RED}ERROR: POSTGRES_PASSWORD not set in .env file!${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Environment configuration valid"

# Create virtual environment
echo ""
echo -e "${YELLOW}[3/7]${NC} Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠${NC}  Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}[4/7]${NC} Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Upgrade pip
echo ""
echo -e "${YELLOW}[5/7]${NC} Upgrading pip..."
pip install --quiet --upgrade pip setuptools wheel
echo -e "${GREEN}✓${NC} pip upgraded"

# Install dependencies
echo ""
echo -e "${YELLOW}[6/7]${NC} Installing dependencies..."
echo "This may take a few minutes (torch, numpy, etc. are large)..."
pip install --quiet -r requirements.txt
echo -e "${GREEN}✓${NC} All dependencies installed"

# Run database migrations
echo ""
echo -e "${YELLOW}[7/7]${NC} Running database migrations..."
if python manage_db.py migrate; then
    echo -e "${GREEN}✓${NC} Database migrations completed"
else
    echo -e "${RED}ERROR: Migration failed!${NC}"
    echo ""
    echo "Possible issues:"
    echo "  1. Database URL is incorrect in .env"
    echo "  2. Database server is not running"
    echo "  3. Network connectivity issue"
    echo ""
    echo "Check your DATABASE_URL in .env and try again"
    exit 1
fi

# Success!
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete! 🎉${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To start the application:"
echo ""
echo -e "  ${BLUE}# Activate virtual environment (if not already active)${NC}"
echo "  source venv/bin/activate"
echo ""
echo -e "  ${BLUE}# Start the development server${NC}"
echo "  uvicorn app.main:app --reload"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API docs (Swagger): http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Note: Keep this terminal open with venv activated!${NC}"
