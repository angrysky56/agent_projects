# COMPASS Web UI - Quick Start Guide

## Prerequisites

- **Python 3.10+** with `uv` installed
- **Node.js 18+** and `npm`
- **Ollama** (optional, for local LLM) - [Install here](https://ollama.com)

## Installation

### 1. Backend Setup

```bash
cd backend

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd web-ui

# Install dependencies
npm install
```

## Configuration

### Environment Variables (Optional)

Create `.env` files for API keys:

**Backend** (backend/.env):
```env
# Ollama Cloud (optional - auto-detects local first)
OLLAMA_API_KEY=your_ollama_cloud_key

# OpenAI (optional)
OPENAI_API_KEY=your_openai_key

# Anthropic (optional)
ANTHROPIC_API_KEY=your_anthropic_key
```

**Frontend** (web-ui/.env):
```env
# API base URL (default: http://localhost:8000)
VITE_API_URL=http://localhost:8000
```

## Running the Application

### Start Backend Server

```bash
cd backend
source .venv/bin/activate
python api_server.py
```

Backend will start on `http://localhost:8000`

### Start Frontend Dev Server

```bash
cd web-ui
npm run dev
```

Frontend will start on `http://localhost:3000`

## First Use

1. Open `http://localhost:3000` in your browser
2. The UI will auto-detect available LLM providers:
   - **Ollama** (local if installed, or cloud with API key)
   - **LM Studio** (if running)
   - **OpenAI** (with API key)
   - **Anthropic** (with API key)
3. Select a provider from the right panel
4. Start chatting! COMPASS auto-configuration is enabled by default

## Features

### 🤖 Multi-Provider LLM Support
- **Ollama** (default): Auto-detects local instance or falls back to cloud
- **LM Studio**: Local OpenAI-compatible server
- **OpenAI**: GPT models via API
- **Anthropic**: Claude models via API

### 🧠 COMPASS Cognitive Framework
- **Auto-Configuration**: AI analyzes task complexity and selects optimal parameters
- **6 Integrated Frameworks**: SLAP, SHAPE, SMART, oMCD, Self-Discover, Intelligence
- **Adaptive Activation**: Light/Medium/Complex/Critical task modes

### 🔧 MCP Integration
- **Desktop-Commander**: Auto-connects on startup (file system, terminal, processes)
- **Tool Discovery**: Browse and execute MCP tools
- **Extensible**: Add custom MCP servers

### 💬 Advanced Chat Interface
- **Streaming Responses**: Real-time message streaming
- **Reasoning Traces**: View COMPASS processing updates
- **Markdown Support**: Rich text formatting
- **Code Highlighting**: Syntax highlighting for code blocks

## Troubleshooting

### Backend won't start
- Check Python version: `python --version` (need 3.10+)
- Ensure all dependencies installed: `uv pip install -r requirements.txt`
- Check if port 8000 is available

### Frontend won't start
- Check Node version: `node --version` (need 18+)
- Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Check if port 3000 is available

### Ollama not detected
- Verify Ollama is running: `ollama list`
- Check Ollama is on default port: `http://localhost:11434`
- For cloud: Set `OLLAMA_API_KEY` environment variable

### MCP desktop-commander fails
- Ensure Node.js is installed (required for npx)
- Check network access for package downloads
- Desktop-commander requires system permissions for file/process access

## Development

### Backend Development
```bash
# Run with hot reload
cd backend
uvicorn api_server:app --reload --port 8000
```

### Frontend Development
```bash
# TypeScript type checking
npm run type-check

# Linting
npm run lint

# Build for production
npm run build
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Web UI (React + TypeScript)      │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │   Chat   │  │ Provider │  │ COMPASS│ │
│  │Interface │  │ Selector │  │Controls│ │
│  └──────────┘  └──────────┘  └────────┘ │
└────────────────┬────────────────────────┘
                 │ WebSocket / REST API
┌────────────────▼────────────────────────┐
│      FastAPI Backend (Python)            │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │   LLM    │  │   MCP    │  │COMPASS │ │
│  │Providers │  │  Client  │  │  API   │ │
│  └──────────┘  └──────────┘  └────────┘ │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌────────┐  ┌────────┐  ┌────────────┐
│ Ollama │  │  MCP   │  │  COMPASS   │
│ (Local/│  │Servers │  │ Framework  │
│ Cloud) │  │(desktop│  │(6 Systems) │
│        │  │-cmd...)│  │            │
└────────┘  └────────┘  └────────────┘
```

## Next Steps

- Explore different LLM providers
- Experiment with COMPASS auto-configuration
- Browse MCP tools available from desktop-commander
- Try complex reasoning tasks to see full COMPASS activation
- Customize COMPASS parameters (toggle auto-config off for manual control)

## Support

For issues, check:
- Backend logs in terminal running `api_server.py`
- Frontend console in browser DevTools (F12)
- Network tab for API request/response debugging

Enjoy exploring the COMPASS cognitive AI system! 🧠✨
