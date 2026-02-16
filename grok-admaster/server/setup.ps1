# Grok-AdMaster Setup Script (Windows PowerShell)
# This script automates the first-time setup process

# Enable strict mode
$ErrorActionPreference = "Stop"

# Colors
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }
function Write-Info { Write-Host $args -ForegroundColor Cyan }

Write-Host "========================================" -ForegroundColor Blue
Write-Host "  Grok-AdMaster Setup Script" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

# Check if running with execution policy that allows scripts
$executionPolicy = Get-ExecutionPolicy
if ($executionPolicy -eq "Restricted") {
    Write-Error "ERROR: PowerShell execution policy is Restricted!"
    Write-Host ""
    Write-Host "Run this command as Administrator to fix:"
    Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    Write-Host ""
    Write-Host "Or run setup.bat instead"
    exit 1
}

# Check if py is installed
Write-Warning "[1/7] Checking py installation..."
try {
    $pythonVersion = & py --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw }
    Write-Success "[OK] $pythonVersion found"
} catch {
    Write-Error "ERROR: py is not installed or not in PATH!"
    Write-Host ""
    Write-Host "Please install py 3.11 or newer from:"
    Write-Host "  https://www.python.org/downloads/"
    Write-Host ""
    Write-Host "Or use Windows Store: winget install Python.Python.3.11"
    Write-Host ""
    Write-Host "Make sure to check 'Add py to PATH' during installation!"
    exit 1
}
Write-Host ""

# Check py version
$versionMatch = $pythonVersion -match "py (\d+)\.(\d+)"
if ($versionMatch) {
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
        Write-Error "ERROR: py 3.8 or newer required (found $pythonVersion)"
        exit 1
    }
}

# Check if .env file exists
Write-Warning "[2/7] Checking environment configuration..."
if (-not (Test-Path .env)) {
    Write-Warning "[!] .env file not found, copying from .env.example..."
    Copy-Item .env.example .env
    Write-Host ""
    Write-Warning "WARNING: Please edit .env with your actual credentials!"
    Write-Host ""
    Write-Host "Required variables:"
    Write-Host "  - SECRET_KEY (generate with: openssl rand -base64 32)"
    Write-Host "  - POSTGRES_PASSWORD"
    Write-Host "  - DATABASE_URL"
    Write-Host ""
    Read-Host "Press Enter after updating .env (or Ctrl+C to exit)"
}

# Validate critical env vars
$envContent = Get-Content .env -Raw
if ($envContent -notmatch "SECRET_KEY=.+" -or $envContent -match "SECRET_KEY=your-secret-key-here") {
    Write-Error "ERROR: SECRET_KEY not properly set in .env file!"
    Write-Host "Generate one with: openssl rand -base64 32"
    exit 1
}

if ($envContent -notmatch "POSTGRES_PASSWORD=.+") {
    Write-Error "ERROR: POSTGRES_PASSWORD not set in .env file!"
    exit 1
}

Write-Success "[OK] Environment configuration valid"
Write-Host ""

# Create virtual environment
Write-Warning "[3/7] Creating virtual environment..."
if (Test-Path venv) {
    Write-Warning "[!] Virtual environment already exists, skipping..."
} else {
    & py -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "ERROR: Failed to create virtual environment!"
        exit 1
    }
    Write-Success "[OK] Virtual environment created"
}
Write-Host ""

# Activate virtual environment
Write-Warning "[4/7] Activating virtual environment..."
& .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: Failed to activate virtual environment!"
    Write-Host ""
    Write-Host "If you get execution policy errors, run as Administrator:"
    Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    exit 1
}
Write-Success "[OK] Virtual environment activated"
Write-Host ""

# Upgrade pip
Write-Warning "[5/7] Upgrading pip..."
try {
    & py -m pip install --quiet --upgrade pip setuptools wheel
    Write-Success "[OK] pip upgraded"
} catch {
    Write-Warning "WARNING: pip upgrade had issues, continuing anyway..."
}
Write-Host ""

# Install dependencies
Write-Warning "[6/7] Installing dependencies..."
Write-Info "This may take several minutes (torch, numpy, etc. are large)..."
& py -m pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: Failed to install dependencies!"
    Write-Host ""
    Write-Host "Try running manually:"
    Write-Host "  pip install -r requirements.txt"
    exit 1
}
Write-Success "[OK] All dependencies installed"
Write-Host ""

# Run database migrations
Write-Warning "[7/7] Running database migrations..."
& py manage_db.py migrate
if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: Migration failed!"
    Write-Host ""
    Write-Host "Possible issues:"
    Write-Host "  1. Database URL is incorrect in .env"
    Write-Host "  2. Database server is not running"
    Write-Host "  3. Network connectivity issue"
    Write-Host ""
    Write-Host "Check your DATABASE_URL in .env and try again"
    exit 1
}
Write-Success "[OK] Database migrations completed"
Write-Host ""

# Success!
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete! 🎉" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application:"
Write-Host ""
Write-Info "  # Activate virtual environment (if not already active)"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Info "  # Start the development server"
Write-Host "  uvicorn app.main:app --reload"
Write-Host ""
Write-Host "API will be available at: http://localhost:8000"
Write-Host "API docs (Swagger): http://localhost:8000/docs"
Write-Host ""
Write-Warning "Note: Keep this terminal open with venv activated!"
Write-Host ""
Read-Host "Press Enter to exit"
