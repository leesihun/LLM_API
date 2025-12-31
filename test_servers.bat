@echo off
REM Quick test to verify servers start correctly with workers

echo ====================================================================
echo Testing Server Startup with Workers
echo ====================================================================
echo.

echo Starting Tools Server (should show 4 workers)...
echo.
start cmd /k "python tools_server.py"

timeout /t 3 /nobreak > nul

echo Starting Main Server (should show 4 workers)...
echo.
start cmd /k "python server.py"

echo.
echo ====================================================================
echo Both servers should now be starting in separate windows
echo Watch for "Started worker process" messages
echo ====================================================================
echo.
echo Press any key to continue...
pause > nul
