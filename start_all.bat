@echo off
REM Start Both Backend and Frontend (Windows)

echo ==========================================
echo HE Team LLM Assistant - Full Stack
echo ==========================================

REM Check if .env exists
if not exist ".env" (
    echo Error: .env file not found
    echo Please copy .env.example to .env and configure it
    pause
    exit /b 1
)

REM Start backend in new window
echo Starting backend server...
start "Backend Server" cmd /k run_backend.bat

REM Wait for backend to start
timeout /t 5 /nobreak >nul

REM Start frontend in new window
echo Starting frontend server...
start "Frontend Server" cmd /k run_frontend.bat

echo.
echo ==========================================
echo Both servers are starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo ==========================================
pause
