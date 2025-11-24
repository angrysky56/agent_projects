@echo off
echo Starting COMPASS Unified Cognitive System...

echo Starting backend server on port 8000...
start "COMPASS Backend" cmd /k "python -m backend.api_server"

echo Starting Web UI...
start "COMPASS Frontend" cmd /k "cd web-ui && npm run dev"

echo COMPASS is running!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo Close the opened windows to stop the services.
pause
