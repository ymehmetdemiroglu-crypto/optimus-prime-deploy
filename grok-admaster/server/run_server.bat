@echo off
echo Starting Grok-AdMaster Server with proper exclusions...
call venv\Scripts\activate
uvicorn app.main:app --reload --reload-exclude "venv" --reload-exclude "__pycache__"
