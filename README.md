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

*(Backend implementation coming soon)*

The backend should run on `http://localhost:8000` by default (as configured in [frontend/static/config.js](frontend/static/config.js)).

## Configuration

### Frontend Configuration

Edit [frontend/static/config.js](frontend/static/config.js) to change the backend API URL:

```javascript
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000'  // Change this to your backend URL
};
```

### Environment Variables

- `FRONTEND_PORT` - Port for frontend server (default: 3000)
- `FRONTEND_HOST` - Host for frontend server (default: localhost)

## Project Structure

```
LLM_API/
├── frontend/
│   └── static/
│       ├── index.html         # Main chat interface
│       ├── login.html         # Login page
│       ├── index_legacy.html  # Legacy chat interface
│       └── config.js          # Frontend configuration
├── run_frontend.py            # Python script to run frontend
├── run_frontend.bat           # Windows batch script
├── run_frontend.sh            # Linux/Mac shell script
├── CLAUDE.md                  # Project instructions for Claude Code
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

## License

*(Add your license here)*
