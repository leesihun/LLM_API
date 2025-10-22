# LLM API - Frontend and Backend

This directory contains the frontend static files and backend API server for the HE Team LLM Assistant.

## Quick Start

### Running the Frontend

The frontend is a static HTML/CSS/JavaScript application that requires a simple HTTP server.

#### Windows
Double-click `run_frontend.bat` or run:
```cmd
python run_frontend.py
```

#### Linux/Mac
```bash
chmod +x run_frontend.sh
./run_frontend.sh
```

#### Custom Port
```cmd
set FRONTEND_PORT=3001
python run_frontend.py
```

Or on Linux/Mac:
```bash
FRONTEND_PORT=3001 python3 run_frontend.py
```

#### Prevent Auto-Open Browser
```cmd
python run_frontend.py --no-browser
```

The frontend will be available at:
- **Login Page**: http://localhost:3000/login.html
- **Main Chat**: http://localhost:3000/index.html
- **Legacy Chat**: http://localhost:3000/index_legacy.html

### Running the Backend

The backend is a FastAPI server with optimized configuration for LLM integration.

#### Quick Start

1. **Generate Configuration** (First time setup):
```bash
python create_env.py
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run Backend**:
```bash
# Cross-platform Python script (recommended)
python run_backend.py

# Windows
run_backend.bat

# Linux/Mac
./run_backend.sh

# Or manually
python -m backend.api.app
```

The backend will be available at:
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Base URL**: http://localhost:8000

## Configuration

### Backend Configuration

The backend uses environment variables for configuration. Use the provided script to generate a `.env` file:

```bash
# Generate .env file with optimized defaults
python create_env.py

# Edit the generated .env file with your specific settings
nano .env  # or your preferred editor
```

#### Key Configuration Options

**Server Settings:**
- `SERVER_HOST` - Host binding (use `localhost` for production, `0.0.0.0` for development)
- `SERVER_PORT` - API server port (default: 8000)
- `SECRET_KEY` - JWT secret key (auto-generated, change for production)

**Ollama Integration:**
- `OLLAMA_HOST` - Ollama service URL (default: http://localhost:11434)
- `OLLAMA_MODEL` - AI model selection (default: llama2:7b)
- `OLLAMA_TIMEOUT` - Request timeout in milliseconds
- `OLLAMA_TEMPERATURE` - Response creativity (0.1-1.0)

**API Keys:**
- `TAVILY_API_KEY` - Web search API key (get from https://tavily.com/)

**Storage Paths:**
- `VECTOR_DB_PATH` - Vector database storage
- `CONVERSATIONS_PATH` - Chat history storage
- `UPLOADS_PATH` - File uploads directory

#### Production Setup

For production deployment, ensure:

1. **Generate secure SECRET_KEY**:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Set appropriate host**:
   ```bash
   SERVER_HOST=localhost  # Not 0.0.0.0 in production
   ```

3. **Configure logging**:
   ```bash
   LOG_LEVEL=WARNING  # Reduce log verbosity
   ```

4. **Set API keys** and validate all paths are writable

### Frontend Configuration

Edit [frontend/static/config.js](frontend/static/config.js) to change the backend API URL:

```javascript
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000'  // Change this to your backend URL
};
```

### Environment Variables

**Frontend:**
- `FRONTEND_PORT` - Port for frontend server (default: 3000)
- `FRONTEND_HOST` - Host for frontend server (default: localhost)

**Backend:** See `.env.example` for complete list of backend environment variables.

## Project Structure

```
LLM_API/
├── create_env.py              # Environment configuration generator
├── .env.example               # Configuration template
├── backend/
│   ├── api/
│   │   ├── app.py             # FastAPI main application
│   │   └── routes.py           # API endpoints
│   ├── config/
│   │   ├── settings.py        # Configuration management (optimized)
│   │   └── users.json         # User database
│   ├── core/
│   │   └── agent_graph.py     # AI agent orchestration
│   ├── models/
│   │   └── schemas.py         # Data models and schemas
│   ├── storage/
│   │   └── conversation_store.py  # Conversation persistence
│   ├── tasks/
│   │   ├── agentic_task.py    # Agent task management
│   │   └── chat_task.py       # Chat processing tasks
│   ├── tools/
│   │   ├── rag_retriever.py   # Document retrieval system
│   │   └── web_search.py      # Web search integration
│   └── utils/
│       └── auth.py            # Authentication utilities
├── frontend/
│   └── static/
│       ├── index.html         # Main chat interface
│       ├── login.html         # Login page
│       ├── index_legacy.html  # Legacy chat interface
│       └── config.js          # Frontend configuration
├── requirements.txt           # Python dependencies
├── run_backend.py             # Python backend launcher (cross-platform)
├── run_backend.bat            # Windows backend launcher
├── run_backend.sh             # Linux/Mac backend launcher
├── run_frontend.py            # Python frontend server
├── run_frontend.bat           # Windows frontend launcher
├── run_frontend.sh            # Linux/Mac frontend launcher
└── README.md                  # This file
```

## Default Credentials

- **Guest**: `guest` / `guest_test1`
- **Admin**: `admin` / `administrator`

## Development

### Testing Frontend Changes

1. Edit HTML/CSS/JavaScript files in [frontend/static/](frontend/static/)
2. Refresh browser (no build step needed)
3. Check browser DevTools console for errors

### No Build Process Required

The frontend uses pure HTML/CSS/JavaScript with no dependencies. Just edit and refresh!

## Troubleshooting

### Port Already in Use

```cmd
# Use a different port
set FRONTEND_PORT=3001
python run_frontend.py
```

### Backend Connection Issues

1. Ensure backend is running on port 8000
2. Check [frontend/static/config.js](frontend/static/config.js) has correct API_BASE_URL
3. Check browser console for CORS errors

### Browser Shows Old Code

- Hard refresh: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
- Clear browser cache

## Version History

### Version 1.1.2 (October 22, 2025) - Python Backend Launcher
- **Cross-platform Backend Script**
  - Added `run_backend.py` for cross-platform backend launching
  - Automated virtual environment setup and dependency installation
  - Handles environment file validation with helpful error messages
  - Works consistently across Windows, Linux, and macOS
  - Includes Python version checking (requires 3.8+)
  - Provides clear status messages and error reporting

### Version 1.1.1 (October 22, 2025) - API Examples Simplification
- **API Examples Notebook**
  - Simplified `API_examples.ipynb` to minimal working example
  - Reduced to essential functionality: login and basic chat
  - Removed advanced examples (web search, RAG, JSON processing) for clarity
  - Added commented example for continuing conversations
  - Streamlined code cells from 11 to 3 for easier understanding

### Version 1.1.0 (October 22, 2025) - Configuration & Security Update
- **Enhanced Configuration Management**
  - Created `create_env.py` script for automated .env file generation
  - Optimized `backend/config/settings.py` with security and performance improvements
  - Added comprehensive default values for all configuration parameters
  - Implemented secure secret key generation
- **Security Improvements**
  - Replaced hardcoded API keys with environment variable placeholders
  - Added secure JWT configuration with HS256 algorithm
  - Improved server host binding recommendations (localhost for production)
  - Enhanced logging configuration for production environments
- **Performance Optimizations**
  - Optimized Ollama settings for better response times
  - Configured appropriate embedding model (all-MiniLM-L6-v2)
  - Set reasonable timeout and context window defaults
  - Organized storage paths for better data management
- **Documentation Updates**
  - Updated README with comprehensive backend setup instructions
  - Added production deployment checklist
  - Documented all configuration parameters with explanations

### Version 1.0.0 (October 22, 2025)
- **Initial Release**
- Linked project to GitHub repository: https://github.com/leesihun/LLM_API
- Set up git version control with initial commit
- Configured .gitignore to exclude Python cache files, virtual environments, node_modules, and environment files
- Project structure includes:
  - FastAPI backend with agent graph, chat tasks, and web search functionality
  - Static HTML/CSS/JavaScript frontend with login and chat interfaces
  - Web search TypeScript module for DuckDuckGo and Google integration
  - Helper scripts for running frontend and backend on Windows/Linux/Mac

## License

*(Add your license here)*
