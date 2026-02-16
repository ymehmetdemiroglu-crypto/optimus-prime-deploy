# Setup Scripts Created

**Date:** 2026-02-15
**Status:** ✅ READY TO USE

---

## 📦 What Was Created

I've created **automated setup scripts** for all platforms to streamline the installation process:

### 1. **setup.sh** (Linux/Mac/Git Bash)
- Bash script with color-coded output
- Full error checking and validation
- Environment variable validation
- Virtual environment management
- Automatic dependency installation
- Database migration execution

### 2. **setup.bat** (Windows Command Prompt)
- Batch script for Windows CMD
- Same functionality as bash version
- Windows-specific error handling
- Pause prompts for user review

### 3. **setup.ps1** (Windows PowerShell)
- PowerShell script with enhanced features
- Color-coded output (Success/Warning/Error)
- Execution policy detection
- Same core functionality
- Better Windows integration

### 4. **SETUP_GUIDE.md**
- Comprehensive setup documentation
- Platform-specific instructions
- Troubleshooting section
- Manual setup steps
- Production deployment guidance

---

## 🚀 How to Use

### **Windows Users (Recommended: PowerShell)**

```powershell
# Open PowerShell in the server directory
cd "C:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server"

# Run setup
.\setup.ps1
```

**If you get execution policy errors:**
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try again
.\setup.ps1
```

**Alternative (Command Prompt):**
```cmd
cd "C:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server"
setup.bat
```

### **Linux/Mac/Git Bash Users**

```bash
cd grok-admaster/server

# Make executable (if needed)
chmod +x setup.sh

# Run setup
./setup.sh
```

---

## ✨ What Each Script Does

### **Step 1: Check Python** ✅
- Verifies Python 3.8+ is installed
- Checks if Python is in PATH
- Validates version compatibility

### **Step 2: Validate Environment** ✅
- Checks if `.env` exists
- Copies from `.env.example` if missing
- Validates `SECRET_KEY` is set
- Validates `POSTGRES_PASSWORD` is set
- Prompts user if configuration needed

### **Step 3: Create Virtual Environment** ✅
- Creates `venv` directory
- Skips if already exists
- Handles errors gracefully

### **Step 4: Activate Virtual Environment** ✅
- Activates venv for current session
- Platform-specific activation
- Verifies activation succeeded

### **Step 5: Upgrade pip** ✅
- Updates pip to latest version
- Installs setuptools and wheel
- Ensures smooth dependency installation

### **Step 6: Install Dependencies** ✅
- Installs all packages from `requirements.txt`
- Includes:
  - FastAPI, Uvicorn
  - SQLAlchemy, Alembic, asyncpg
  - Cryptography
  - PyTorch, NumPy, scikit-learn
  - All ML and API dependencies
- Shows progress and any errors

### **Step 7: Run Migrations** ✅
- Executes `python manage_db.py migrate`
- Creates database schema
- Validates database connection
- Shows clear error messages if fails

---

## 🎯 Expected Output

### **Successful Setup:**

```
========================================
  Grok-AdMaster Setup Script
========================================

[1/7] Checking Python installation...
[OK] Python 3.11.5 found

[2/7] Checking environment configuration...
[OK] Environment configuration valid

[3/7] Creating virtual environment...
[OK] Virtual environment created

[4/7] Activating virtual environment...
[OK] Virtual environment activated

[5/7] Upgrading pip...
[OK] pip upgraded

[6/7] Installing dependencies...
This may take several minutes...
[OK] All dependencies installed

[7/7] Running database migrations...
[OK] Database migrations completed

========================================
  Setup Complete! 🎉
========================================

To start the application:

  # Activate virtual environment
  source venv/bin/activate

  # Start development server
  uvicorn app.main:app --reload

API will be available at: http://localhost:8000
API docs (Swagger): http://localhost:8000/docs
```

---

## 🐛 Troubleshooting

### **Python Not Found**

**Windows:**
```powershell
# Install Python from Windows Store
winget install Python.Python.3.11

# Or download from python.org
# Make sure "Add to PATH" is checked!
```

**Linux:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**Mac:**
```bash
brew install python@3.11
```

### **Execution Policy Error (PowerShell)**

```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then run setup again
.\setup.ps1
```

**Alternative:** Use `setup.bat` instead

### **Missing .env File**

Script will copy `.env.example` to `.env` automatically.

You **MUST** edit `.env` and set:
```bash
SECRET_KEY=<your-32-char-secret>  # Generate with: openssl rand -base64 32
POSTGRES_PASSWORD=<your-db-password>
DATABASE_URL=<your-database-url>
```

### **Database Connection Fails**

Check:
1. ✅ `DATABASE_URL` is correct in `.env`
2. ✅ Database server is running (Supabase or local PostgreSQL)
3. ✅ Network allows connection
4. ✅ Credentials are correct

Test manually:
```bash
python -c "from app.core.config import settings; print(settings.ASYNC_DATABASE_URL)"
```

### **Migration Fails**

If migration fails:

```bash
# Check what went wrong
python manage_db.py current

# See migration logs
python manage_db.py history

# Reset (DEV ONLY - destroys data!)
python manage_db.py reset
```

---

## 📁 Files Created

```
grok-admaster/server/
├── setup.sh              # Bash setup script (Linux/Mac/Git Bash)
├── setup.bat             # Batch setup script (Windows CMD)
├── setup.ps1             # PowerShell setup script (Windows)
└── SETUP_GUIDE.md        # Comprehensive setup documentation
```

Plus this summary at:
```
grok-admaster/
└── SETUP_SCRIPTS_SUMMARY.md  # This file
```

---

## 🔄 After Setup

Once setup completes successfully:

1. **Virtual environment will be activated** (you'll see `(venv)` in prompt)
2. **Database schema is created** via migrations
3. **All dependencies are installed**

### **Start the Server:**

```bash
# Make sure venv is active (you should see (venv) in prompt)
uvicorn app.main:app --reload
```

### **Access API:**

- **API:** http://localhost:8000
- **Swagger Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### **Test Database:**

```bash
python manage_db.py current
# Should show: 20260215_0001 (head)
```

---

## 🚀 Production Deployment

For production deployment:

1. **Update `.env`:**
   ```bash
   ENV=production
   SECRET_KEY=<strong-unique-key>
   DATABASE_URL=<production-database>
   CORS_ORIGINS=https://yourdomain.com
   ```

2. **Run setup script** (same as development)

3. **Use production server:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

   Or with gunicorn:
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

4. **Set up reverse proxy** (Nginx/Caddy)

5. **Enable HTTPS** (Let's Encrypt)

See [SETUP_GUIDE.md](grok-admaster/server/SETUP_GUIDE.md) for detailed production instructions.

---

## ✅ Verification Checklist

After running setup, verify:

- [x] Python 3.8+ installed
- [x] Virtual environment created (`venv/` directory exists)
- [x] `.env` file configured with secrets
- [x] All dependencies installed (check `pip list`)
- [x] Database migrations run (`python manage_db.py current` shows revision)
- [x] Server starts without errors (`uvicorn app.main:app --reload`)
- [x] API accessible at http://localhost:8000
- [x] Swagger docs work at http://localhost:8000/docs

---

## 🎓 What You Learned

The setup scripts demonstrate:

1. **Environment validation** - Check prerequisites before proceeding
2. **Error handling** - Graceful failures with helpful messages
3. **Cross-platform support** - Same workflow on all OSs
4. **Automation** - One command to full working environment
5. **Safety** - Validation prevents common mistakes

---

## 📚 Additional Resources

- **[SETUP_GUIDE.md](grok-admaster/server/SETUP_GUIDE.md)** - Detailed setup instructions
- **[MIGRATION_GUIDE.md](grok-admaster/server/MIGRATION_GUIDE.md)** - Database migration reference
- **[SECURITY_FIXES_SUMMARY.md](grok-admaster/SECURITY_FIXES_SUMMARY.md)** - Security improvements
- **[PHASE2_SUMMARY.md](grok-admaster/PHASE2_SUMMARY.md)** - Migration framework details

---

**Ready to run? Choose your platform's script and execute!**

```bash
# Linux/Mac/Git Bash
./setup.sh

# Windows PowerShell
.\setup.ps1

# Windows CMD
setup.bat
```

**Setup Scripts Status: READY ✅**
