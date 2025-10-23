"""
SQL Database Query Tool
Safely execute read-only SQL queries
"""

import logging
import sqlite3
import re
from typing import List, Dict, Any, Optional
from pathlib import Path


logger = logging.getLogger(__name__)


class SQLQueryTool:
    """
    SQL database query executor

    Security Features:
    - READ-ONLY queries only
    - No DDL operations (CREATE, DROP, ALTER, etc.)
    - No DML operations (INSERT, UPDATE, DELETE)
    - Query timeout
    - Result row limits
    """

    # Forbidden SQL keywords (case-insensitive)
    FORBIDDEN_KEYWORDS = {
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'REPLACE', 'EXEC', 'EXECUTE', 'PRAGMA'
    }

    def __init__(self, db_path: str = "./data/app.db", timeout: int = 5, max_rows: int = 100):
        self.db_path = Path(db_path)
        self.timeout = timeout
        self.max_rows = max_rows

        # Create database if it doesn't exist
        self._initialize_db()

    def _initialize_db(self):
        """Initialize database with example tables"""
        if not self.db_path.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create sample database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create sample tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS products (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        price REAL NOT NULL,
                        category TEXT,
                        stock INTEGER DEFAULT 0
                    )
                """)

                # Insert sample data
                cursor.execute("""
                    INSERT OR IGNORE INTO users (id, name, email) VALUES
                    (1, 'Alice Smith', 'alice@example.com'),
                    (2, 'Bob Johnson', 'bob@example.com'),
                    (3, 'Carol Williams', 'carol@example.com')
                """)

                cursor.execute("""
                    INSERT OR IGNORE INTO products (id, name, price, category, stock) VALUES
                    (1, 'Laptop', 999.99, 'Electronics', 15),
                    (2, 'Mouse', 29.99, 'Electronics', 50),
                    (3, 'Desk Chair', 199.99, 'Furniture', 8),
                    (4, 'Coffee Mug', 12.99, 'Kitchen', 100)
                """)

                conn.commit()
                logger.info(f"[SQL Tool] Initialized database at {self.db_path}")

    def _validate_query(self, query: str) -> bool:
        """
        Validate SQL query for security

        Raises:
            ValueError: If query contains forbidden operations
        """
        query_upper = query.upper()

        # Check for forbidden keywords
        for keyword in self.FORBIDDEN_KEYWORDS:
            if keyword in query_upper:
                raise ValueError(f"Forbidden SQL operation: {keyword}")

        # Must be a SELECT query
        if not query_upper.strip().startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed")

        return True

    async def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a read-only SQL query

        Args:
            query: SQL SELECT query

        Returns:
            Dictionary with:
                - success: bool
                - columns: List[str]
                - rows: List[Dict]
                - row_count: int
                - error: Optional[str]
        """
        logger.info(f"[SQL Tool] Executing query: {query[:100]}...")

        try:
            # Validate query
            self._validate_query(query)

            # Execute query
            with sqlite3.connect(self.db_path, timeout=self.timeout) as conn:
                conn.row_factory = sqlite3.Row  # Enable column names
                cursor = conn.cursor()

                cursor.execute(query)

                # Fetch results (with row limit)
                rows = cursor.fetchmany(self.max_rows)

                # Get column names
                columns = [description[0] for description in cursor.description] if cursor.description else []

                # Convert to list of dicts
                results = [dict(row) for row in rows]

                row_count = len(results)

                # Check if there are more rows
                has_more = len(cursor.fetchone() or []) > 0

                logger.info(f"[SQL Tool] Query successful, {row_count} rows returned")

                return {
                    "success": True,
                    "columns": columns,
                    "rows": results,
                    "row_count": row_count,
                    "has_more": has_more,
                    "error": None
                }

        except ValueError as e:
            # Validation error
            logger.error(f"[SQL Tool] Validation error: {e}")
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "has_more": False,
                "error": str(e)
            }

        except Exception as e:
            # Execution error
            logger.error(f"[SQL Tool] Execution error: {e}")
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "has_more": False,
                "error": f"SQL error: {str(e)}"
            }

    def format_results(self, result: Dict[str, Any]) -> str:
        """
        Format query results for display
        """
        if not result["success"]:
            return f"❌ Error: {result['error']}"

        if result["row_count"] == 0:
            return "No results found"

        # Build table
        output = []
        output.append(f"✅ Query successful - {result['row_count']} row(s) returned")

        if result["has_more"]:
            output.append(f"⚠️ Results truncated to {self.max_rows} rows")

        output.append("")

        # Column headers
        headers = " | ".join(result["columns"])
        output.append(headers)
        output.append("-" * len(headers))

        # Rows
        for row in result["rows"]:
            row_values = " | ".join(str(row.get(col, "NULL")) for col in result["columns"])
            output.append(row_values)

        return "\n".join(output)

    async def get_schema(self) -> str:
        """
        Get database schema (table and column information)
        """
        logger.info("[SQL Tool] Retrieving database schema")

        try:
            with sqlite3.connect(self.db_path, timeout=self.timeout) as conn:
                cursor = conn.cursor()

                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                tables = cursor.fetchall()

                schema_info = ["📊 Database Schema:\n"]

                for (table_name,) in tables:
                    # Get columns for each table
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()

                    schema_info.append(f"\nTable: {table_name}")
                    schema_info.append("Columns:")

                    for col in columns:
                        col_name = col[1]
                        col_type = col[2]
                        is_pk = " (PRIMARY KEY)" if col[5] else ""
                        schema_info.append(f"  - {col_name}: {col_type}{is_pk}")

                return "\n".join(schema_info)

        except Exception as e:
            logger.error(f"[SQL Tool] Error getting schema: {e}")
            return f"Error retrieving schema: {str(e)}"


# Global instance
sql_query_tool = SQLQueryTool(db_path="./data/app.db", timeout=5, max_rows=100)
