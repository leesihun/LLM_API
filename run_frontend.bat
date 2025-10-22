@echo off
REM Frontend Server Launcher for Windows
REM Double-click this file to start the frontend server

echo Starting Frontend Server...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

REM Run the frontend server
python run_frontend.py

pause
