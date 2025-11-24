@echo off
setlocal enabledelayedexpansion

echo 🧠 COMPASS Web UI Setup
echo =======================
echo.

:: Check prerequisites
echo Checking prerequisites...

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.10 or higher.
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo ✓ %%i found

:: Check uv
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  uv not found. Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
)
echo ✓ uv found

:: Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found. Please install Node.js 18 or higher.
    exit /b 1
)
for /f "tokens=*" %%i in ('node --version') do echo ✓ Node.js %%i found

:: Check npm
call npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm not found. Please install npm.
    exit /b 1
)
for /f "tokens=*" %%i in ('call npm --version') do echo ✓ npm %%i found

echo.
echo Setting up backend...
cd backend

:: Create virtual environment
echo Creating Python virtual environment...
call uv venv

:: Activate and install dependencies
echo Installing Python dependencies...
call .venv\Scripts\activate
call uv pip install -r requirements.txt

echo ✓ Backend setup complete
echo.

:: Setup frontend
echo Setting up frontend...
cd ..\web-ui

echo Installing Node.js dependencies...
call npm install

echo ✓ Frontend setup complete
echo.

echo ✅ Setup complete!
echo.
echo To start the application, run:
echo   start.bat
echo.
echo For more information, see web-ui\README.md

pause
