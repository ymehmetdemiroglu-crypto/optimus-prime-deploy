# Grok-AdMaster Setup Guide

Quick and easy setup for the Grok-AdMaster API server.

---

## 🚀 Quick Start (Automated)

Choose the script for your platform and run it:

### **Windows (PowerShell)** - RECOMMENDED
```powershell
# Run from the server directory
.\setup.ps1
```

### **Windows (Command Prompt)**
```cmd
# Run from the server directory
setup.bat
```

### **Linux / Mac / Git Bash**
```bash
# Run from the server directory
./setup.sh
```

The script will:
1. ✅ Check Python installation
2. ✅ Validate environment configuration
3. ✅ Create virtual environment
4. ✅ Install all dependencies
5. ✅ Run database migrations
6. ✅ Verify setup

---

## 📋 Prerequisites

### Required

- **Python 3.8+** (3.11 recommended)
  - Windows: https://www.python.org/downloads/ or `winget install Python.Python.3.11`
  - Mac: `brew install python@3.11`
  - Linux: `sudo apt install python3 python3-pip python3-venv`

- **PostgreSQL Database** (or Supabase)
  - The app connects to the database specified in `DATABASE_URL`
  - You can use Supabase (as configured in `.env`)

### Optional

- **OpenSSL** (for generating SECRET_KEY)
  - Included with Git Bash on Windows
  - Pre-installed on Linux/Mac

---

## 🔧 Manual Setup

If you prefer manual control or the automated script fails:

### 1. Install Python

Make sure Python 3.8+ is installed:
```bash
python --version
# Should show: Python 3.11.x or newer
```

### 2. Navigate to Server Directory

```bash
cd grok-admaster/server
```

### 3. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate it
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (CMD):
venv\Scripts\activate.bat

# Linux/Mac/Git Bash:
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- FastAPI, Uvicorn (web framework)
- SQLAlchemy, Alembic, asyncpg (database)
- Cryptography (encryption)
- PyTorch, NumPy, scikit-learn (ML)
- All other dependencies

### 5. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit with your values
# Required:
#   - SECRET_KEY (generate with: openssl rand -base64 32)
#   - POSTGRES_PASSWORD
#   - DATABASE_URL
```

### 6. Run Database Migrations

```bash
python manage_db.py migrate
```

### 7. Start the Server

```bash
uvicorn app.main:app --reload
```

Server will start at: http://localhost:8000

---

## 🔍 Troubleshooting

### Python Not Found

**Error:** `python: command not found`

**Fix:**
```bash
# Try python3 instead
python3 --version

# Or install Python
# Windows: winget install Python.Python.3.11
# Mac: brew install python@3.11
# Linux: sudo apt install python3
```

### Execution Policy Error (PowerShell)

**Error:** `running scripts is disabled on this system`

**Fix:**
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or use `setup.bat` instead.

### Module Not Found Errors

**Error:** `ModuleNotFoundError: No module named 'fastapi'`

**Fix:**
```bash
# Make sure venv is activated (you should see (venv) in prompt)
# If not:
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\Activate.ps1  # Windows PowerShell

# Reinstall dependencies
pip install -r requirements.txt
```

### SECRET_KEY Error

**Error:** `SECRET_KEY is required. Set it via environment variable.`

**Fix:**
```bash
# Generate a secure key
openssl rand -base64 32

# Or use Python
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to .env file:
SECRET_KEY=<generated-key-here>
```

### Database Connection Error

**Error:** `Database not initialized` or connection timeouts

**Fix:**
```bash
# Check DATABASE_URL in .env
cat .env | grep DATABASE_URL

# Test connection
python -c "from app.core.config import settings; print(settings.ASYNC_DATABASE_URL)"

# Verify Supabase is accessible
# Make sure your IP is allowed in Supabase settings
```

### Migration Failed

**Error:** `ERROR: Migration failed!`

**Check:**
1. Database URL is correct
2. Database server is running
3. Network connectivity
4. Database user has CREATE TABLE permissions

**Debug:**
```bash
# Check current migration status
python manage_db.py current

# Try manual migration
alembic upgrade head
```

### Port Already in Use

**Error:** `Address already in use` when starting uvicorn

**Fix:**
```bash
# Use different port
uvicorn app.main:app --reload --port 8001

# Or kill process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <process_id> /F

# Linux/Mac:
lsof -ti:8000 | xargs kill -9
```

---

## 🎯 Verify Setup

After setup completes, verify everything works:

### 1. Check API is Running

```bash
curl http://localhost:8000/api/v1/dashboard/summary
```

Should return JSON with dashboard data.

### 2. Check API Documentation

Open in browser: http://localhost:8000/docs

You should see Swagger UI with all API endpoints.

### 3. Check Database Connection

```bash
python -c "from app.core.database import engine; import asyncio; asyncio.run(engine.connect())"
```

Should complete without errors.

### 4. Check Migrations

```bash
python manage_db.py current
```

Should show: `20260215_0001 (head)`

---

## 📂 Directory Structure After Setup

```
grok-admaster/server/
├── venv/                    # Virtual environment (created by setup)
├── alembic/                 # Database migrations
│   └── versions/
│       └── 20260215_0001_initial_schema.py
├── app/                     # Application code
│   ├── api/                 # API endpoints
│   ├── core/                # Core functionality
│   │   ├── config.py        # Configuration
│   │   ├── database.py      # Database connection
│   │   ├── encryption.py    # Credential encryption
│   │   └── logging_config.py # Logging setup
│   ├── models/              # Data models
│   └── services/            # Business logic
├── .env                     # Environment variables (YOU EDIT THIS)
├── .env.example             # Example configuration
├── requirements.txt         # Python dependencies
├── manage_db.py             # Database management CLI
├── alembic.ini              # Alembic configuration
├── setup.sh                 # Setup script (Linux/Mac)
├── setup.bat                # Setup script (Windows CMD)
├── setup.ps1                # Setup script (Windows PowerShell)
└── SETUP_GUIDE.md           # This file
```

---

## 🔒 Security Checklist

Before deploying to production:

- [ ] Changed `SECRET_KEY` to unique value (min 32 chars)
- [ ] Changed `POSTGRES_PASSWORD` from default
- [ ] `DATABASE_URL` points to production database
- [ ] `.env` file is in `.gitignore` (never commit!)
- [ ] `ENV=production` in `.env`
- [ ] CORS origins limited to your domains
- [ ] API authentication enabled (if using)
- [ ] SSL/HTTPS configured

---

## 🚀 Running in Production

### Using systemd (Linux)

Create `/etc/systemd/system/grok-admaster.service`:

```ini
[Unit]
Description=Grok-AdMaster API
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/grok-admaster/server
Environment="PATH=/var/www/grok-admaster/server/venv/bin"
ExecStart=/var/www/grok-admaster/server/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable grok-admaster
sudo systemctl start grok-admaster
```

### Using Docker

See `../Dockerfile` for containerized deployment.

---

## 🆘 Getting Help

1. **Check this guide** - Common issues covered above
2. **Review logs** - Check terminal output for errors
3. **Check documentation**:
   - [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Database issues
   - [SECURITY_FIXES_SUMMARY.md](../SECURITY_FIXES_SUMMARY.md) - Security setup
   - [PHASE2_SUMMARY.md](../PHASE2_SUMMARY.md) - Migration framework
4. **Create an issue** - If problem persists

---

## 🎉 Next Steps

After successful setup:

1. **Explore API documentation**: http://localhost:8000/docs
2. **Review configuration**: Check `.env` file
3. **Run tests**: `pytest` (if tests available)
4. **Connect frontend**: Configure API URL in frontend
5. **Deploy**: Follow production deployment guide

---

**Setup Script Version:** 1.0
**Last Updated:** 2026-02-15
