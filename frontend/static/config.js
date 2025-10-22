/**
 * Frontend Configuration
 * Configure the backend API URL here
 */

const CONFIG = {
    // Backend API URL - change this if your backend is running on a different host/port
    API_BASE_URL: 'http://localhost:8000',

    // Alternative configurations for different environments
    // Development: 'http://localhost:8000'
    // Production: 'http://your-server-ip:8000'
    // Or use window.location to auto-detect: `${window.location.protocol}//${window.location.hostname}:8000`
};

// Auto-detect if running on same host but different port
// Uncomment this if you want automatic detection
// CONFIG.API_BASE_URL = `${window.location.protocol}//${window.location.hostname}:8000`;
