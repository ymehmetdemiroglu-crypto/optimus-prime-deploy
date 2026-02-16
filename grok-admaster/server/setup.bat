@echo off
REM Grok-AdMaster Setup Script (Windows CMD)
REM This script automates the first-time setup process

setlocal enabledelayedexpansion

echo ========================================
echo   Grok-AdMaster Setup Script
echo ========================================
echo.

REM Check if Python is installed
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.11 or newer from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation!
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found
echo.

REM Check if .env file exists
echo [2/7] Checking environment configuration...
if not exist .env (
    echo [!] .env file not found, copying from .env.example...
    copy .env.example .env >nul
    echo.
    echo WARNING: Please edit .env with your actual credentials!
    echo.
    echo Required variables:
    echo   - SECRET_KEY (generate with: openssl rand -base64 32)
    echo   - POSTGRES_PASSWORD
    echo   - DATABASE_URL
    echo.
    pause
)

REM Validate critical env vars
findstr /C:"SECRET_KEY=" .env >nul
if errorlevel 1 (
    echo ERROR: SECRET_KEY not set in .env file!
    pause
    exit /b 1
)

findstr /C:"POSTGRES_PASSWORD=" .env >nul
if errorlevel 1 (
    echo ERROR: POSTGRES_PASSWORD not set in .env file!
    pause
    exit /b 1
)

echo [OK] Environment configuration valid
echo.

REM Create virtual environment
echo [3/7] Creating virtual environment...
if exist venv (
    echo [!] Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

REM Activate virtual environment
echo [4/7] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo [5/7] Upgrading pip...
python -m pip install --quiet --upgrade pip setuptools wheel
if errorlevel 1 (
    echo WARNING: pip upgrade had issues, continuing anyway...
)
echo [OK] pip upgraded
echo.

REM Install dependencies
echo [6/7] Installing dependencies...
echo This may take several minutes (torch, numpy, etc. are large)...
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    echo.
    echo Try running manually:
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)
echo [OK] All dependencies installed
echo.

REM Run database migrations
echo [7/7] Running database migrations...
python manage_db.py migrate
if errorlevel 1 (
    echo ERROR: Migration failed!
    echo.
    echo Possible issues:
    echo   1. Database URL is incorrect in .env
    echo   2. Database server is not running
    echo   3. Network connectivity issue
    echo.
    echo Check your DATABASE_URL in .env and try again
    pause
    exit /b 1
)
echo [OK] Database migrations completed
echo.

REM Success!
echo ========================================
echo   Setup Complete! 🎉
echo ========================================
echo.
echo To start the application:
echo.
echo   1. Activate virtual environment (if not already active):
echo      venv\Scripts\activate.bat
echo.
echo   2. Start the development server:
echo      run_server.bat
echo.
echo API will be available at: http://localhost:8000
echo API docs (Swagger): http://localhost:8000/docs
echo.
echo Note: Keep this terminal open with venv activated!
echo.
pause
