@echo off
REM Backend Server Launcher Script (Windows)

echo ==========================================
echo HE Team LLM Assistant - Backend
echo ==========================================

REM Check if .env exists
if not exist ".env" (
    echo Error: .env file not found
    echo Please copy .env.example to .env and configure it
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run server
echo Starting backend server...
python server.py

pause
