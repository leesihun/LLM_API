@echo off
REM Start both the main server and tools server

echo Starting LLM API Servers...
echo.

REM Start tools server in a new window
start "Tools API (Port 1006)" cmd /k python tools_server.py

REM Wait a moment for tools server to start
timeout /t 2 /nobreak >nul

REM Start main server in a new window
start "Main API (Port 1007)" cmd /k python server.py

echo.
echo Both servers started!
echo   - Tools API: http://localhost:1006
echo   - Main API:  http://localhost:1007
echo.
echo Press any key to close this window...
pause >nul
