"""
Hibiscus Leaf Disease Detection — Database Module
==================================================
Manages MySQL connection pooling, schema initialization,
and parameterized CRUD operations for scan_history table.
Designed with graceful degradation: if MySQL is offline,
calls fail gracefully with diagnostic errors without crashing.
"""

import os
import logging
from typing import Tuple, List, Dict, Any, Optional
from dotenv import load_dotenv
import pymysql
import pymysql.cursors

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logger = logging.getLogger(__name__)

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "hibiscus_ai")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
CONNECT_TIMEOUT = 1  # Fast failover if DB is offline


def get_db_connection(use_database: bool = True) -> pymysql.Connection:
    """
    Creates and returns a connection to MySQL server.
    If use_database is False, connects without specifying DB_NAME (for setup/init).
    """
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME if use_database else None,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=CONNECT_TIMEOUT,
        autocommit=True
    )


def check_db_status() -> Tuple[bool, str]:
    """
    Verifies whether MySQL server is accessible and database exists.
    Returns (is_connected, message).
    """
    try:
        conn = get_db_connection(use_database=True)
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 AS alive")
        conn.close()
        return True, "MySQL database connected."
    except pymysql.err.OperationalError as e:
        code, msg = e.args[0], e.args[1] if len(e.args) > 1 else str(e)
        if code == 1049:  # Unknown database
            return False, f"MySQL server is running, but database '{DB_NAME}' does not exist yet."
        return False, f"MySQL server is unavailable on {DB_HOST}:{DB_PORT} ({msg})."
    except Exception as e:
        return False, f"Database connection error: {str(e)}"


def initialize_schema() -> Tuple[bool, str]:
    """
    Ensures database and scan_history table exist.
    """
    try:
        # Step 1: Connect to server and create database if needed
        conn = get_db_connection(use_database=False)
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` DEFAULT CHARACTER SET utf8mb4;")
        conn.close()

        # Step 2: Connect to database and create table
        conn = get_db_connection(use_database=True)
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    predicted_class VARCHAR(50) NOT NULL,
                    raw_score DECIMAL(6, 4) NOT NULL,
                    threshold DECIMAL(6, 4) NOT NULL,
                    threshold_status VARCHAR(50) NOT NULL,
                    scan_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_timestamp (scan_timestamp DESC),
                    INDEX idx_class (predicted_class)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)
        conn.close()
        return True, "Database schema initialized successfully."
    except Exception as e:
        logger.error(f"Schema initialization failed: {e}")
        return False, str(e)


def save_scan(
    filename: str,
    predicted_class: str,
    raw_score: float,
    threshold: float,
    threshold_status: str
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Inserts a leaf prediction record into scan_history using parameterized SQL.
    Returns (success, scan_id, error_message).
    """
    query = """
        INSERT INTO scan_history (filename, predicted_class, raw_score, threshold, threshold_status)
        VALUES (%s, %s, %s, %s, %s)
    """
    try:
        conn = get_db_connection(use_database=True)
        with conn.cursor() as cursor:
            cursor.execute(
                query,
                (filename, predicted_class, raw_score, threshold, threshold_status)
            )
            scan_id = cursor.lastrowid
        conn.close()
        return True, scan_id, None
    except Exception as e:
        logger.warning(f"Failed to persist scan record to database: {e}")
        return False, None, str(e)


def get_scan_history(limit: int = 20) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """
    Retrieves recent leaf scans from scan_history, newest first.
    Returns (success, list_of_records, error_message).
    """
    # Guard against invalid limit values
    safe_limit = max(1, min(int(limit), 100))
    query = """
        SELECT id, filename, predicted_class, raw_score, threshold, threshold_status, scan_timestamp
        FROM scan_history
        ORDER BY scan_timestamp DESC
        LIMIT %s
    """
    try:
        conn = get_db_connection(use_database=True)
        with conn.cursor() as cursor:
            cursor.execute(query, (safe_limit,))
            records = cursor.fetchall()
        conn.close()

        # Format records for JSON serialization
        formatted = []
        for r in records:
            formatted.append({
                "id": r["id"],
                "filename": r["filename"],
                "predicted_class": r["predicted_class"],
                "raw_score": float(r["raw_score"]),
                "threshold": float(r["threshold"]),
                "threshold_status": r["threshold_status"],
                "scan_timestamp": r["scan_timestamp"].strftime("%Y-%m-%d %H:%M:%S") if r.get("scan_timestamp") else None
            })
        return True, formatted, None
    except Exception as e:
        logger.warning(f"Failed to retrieve scan history from database: {e}")
        return False, [], str(e)
