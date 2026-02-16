Write-Host "Starting Grok-AdMaster Server with proper exclusions..."
& ".\venv\Scripts\Activate.ps1"
uvicorn app.main:app --reload --reload-exclude "venv" --reload-exclude "__pycache__"
