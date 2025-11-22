#!/bin/bash
# Test backend startup

cd "$(dirname "$0")/backend"

# Activate virtual environment
source .venv/bin/activate

# Test imports
python -c "
import sys
try:
    from fastapi import FastAPI
    from llm_providers import OllamaProvider
    from mcp_client import get_mcp_client
    from auto_config import auto_configure
    from compass_api import get_compass_api
    print('✅ All backend modules imported successfully')
    sys.exit(0)
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "Backend is ready! Start with:"
    echo "  cd backend"
    echo "  source .venv/bin/activate"
    echo "  python api_server.py"
else
    echo ""
    echo "Backend has import errors. Check dependencies."
    exit 1
fi
