#!/bin/bash
# Setup script for COMPASS Web UI

set -e

echo "🧠 COMPASS Web UI Setup"
echo "======================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10 or higher."
    exit 1
fi
echo "✓ Python found: $(python3 --version)"

# Check uv
if ! command -v uv &> /dev/null; then
    echo "⚠️  uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo "✓ uv found"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18 or higher."
    exit 1
fi
echo "✓ Node.js found: $(node --version)"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Please install npm."
    exit 1
fi
echo "✓ npm found: $(npm --version)"

echo ""
echo "Setting up backend..."
cd backend

# Create virtual environment
echo "Creating Python virtual environment..."
uv venv

# Activate and install dependencies
echo "Installing Python dependencies..."
source .venv/bin/activate
uv pip install -r requirements.txt

echo "✓ Backend setup complete"
echo ""

# Setup frontend
echo "Setting up frontend..."
cd ../web-ui

echo "Installing Node.js dependencies..."
npm install

echo "✓ Frontend setup complete"
echo ""

echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo ""
echo "Terminal 1 (Backend):"
echo "  cd backend"
echo "  source .venv/bin/activate"
echo "  python api_server.py"
echo ""
echo "Terminal 2 (Frontend):"
echo "  cd web-ui"
echo "  npm run dev"
echo ""
echo "Then open http://localhost:3000 in your browser"
echo ""
echo "For more information, see web-ui/README.md"
