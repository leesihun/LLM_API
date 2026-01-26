"""
OpenCode Server Manager
Manages the lifecycle of opencode persistent server
"""
import subprocess
import time
import threading
from typing import Optional

import config


class OpenCodeServerManager:
    """
    Singleton manager for opencode server lifecycle

    Responsibilities:
    - Start server on initialization
    - Auto-restart once on failure
    - Provide server URL for executors
    """

    _instance: Optional["OpenCodeServerManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "OpenCodeServerManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._process: Optional[subprocess.Popen] = None
        self._restart_attempted = False
        self._server_url = f"http://{config.OPENCODE_SERVER_HOST}:{config.OPENCODE_SERVER_PORT}"
        self._initialized = True

    @property
    def server_url(self) -> str:
        """Get server URL for --attach flag"""
        return self._server_url

    @property
    def is_running(self) -> bool:
        """Check if server process is running"""
        if self._process is None:
            return False
        return self._process.poll() is None

    def start(self) -> None:
        """
        Start opencode server

        Raises:
            RuntimeError: If server fails to start
        """
        if self.is_running:
            print(f"[OPENCODE SERVER] Already running on {self._server_url}")
            return

        print(f"[OPENCODE SERVER] Starting on port {config.OPENCODE_SERVER_PORT}...")

        # On Windows, use .cmd extension for npm global binaries
        import sys
        opencode_cmd = config.OPENCODE_PATH
        if sys.platform == "win32" and not opencode_cmd.endswith(".cmd"):
            opencode_cmd = f"{opencode_cmd}.cmd"

        cmd = [
            opencode_cmd,
            "serve",
            "--port", str(config.OPENCODE_SERVER_PORT),
            "--hostname", config.OPENCODE_SERVER_HOST,
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to be ready (max 30 seconds)
        if not self._wait_for_server(timeout=30):
            self._process.terminate()
            self._process = None
            raise RuntimeError(
                f"OpenCode server failed to start on {self._server_url}. "
                f"Check if opencode is installed: npm install -g opencode-ai@latest"
            )

        print(f"[OPENCODE SERVER] Running on {self._server_url}")

    def _wait_for_server(self, timeout: int) -> bool:
        """Wait for server to be ready"""
        import urllib.request
        import urllib.error

        start = time.time()
        while time.time() - start < timeout:
            try:
                # Try to connect to server
                req = urllib.request.Request(f"{self._server_url}/health")
                with urllib.request.urlopen(req, timeout=2):
                    return True
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                time.sleep(0.5)
        return False

    def stop(self) -> None:
        """Stop opencode server"""
        if self._process is not None:
            print("[OPENCODE SERVER] Stopping...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            print("[OPENCODE SERVER] Stopped")

    def ensure_running(self) -> None:
        """
        Ensure server is running, restart once if needed

        Raises:
            RuntimeError: If server is down and restart fails
        """
        if self.is_running:
            return

        if self._restart_attempted:
            raise RuntimeError(
                "OpenCode server is down and restart already attempted. "
                "Manual intervention required."
            )

        print("[OPENCODE SERVER] Server down, attempting restart...")
        self._restart_attempted = True
        self.start()
        self._restart_attempted = False  # Reset on successful restart


# Global instance
_server_manager: Optional[OpenCodeServerManager] = None


def get_server_manager() -> OpenCodeServerManager:
    """Get the global server manager instance"""
    global _server_manager
    if _server_manager is None:
        _server_manager = OpenCodeServerManager()
    return _server_manager


def start_opencode_server() -> None:
    """Start the opencode server (call on tools_server startup)"""
    from tools.python_coder.opencode_config import ensure_opencode_config

    # Generate config first
    ensure_opencode_config()

    # Start server
    manager = get_server_manager()
    manager.start()


def stop_opencode_server() -> None:
    """Stop the opencode server (call on tools_server shutdown)"""
    global _server_manager
    if _server_manager is not None:
        _server_manager.stop()
        _server_manager = None
