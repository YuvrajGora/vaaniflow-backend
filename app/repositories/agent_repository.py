import os
import sqlite3
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional, Generator, List
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class DatabasePersistenceError(Exception):
    """Raised when persisting an agent to SQLite fails."""
    pass


class AgentRepository:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = get_settings().database_path
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        business_type TEXT NOT NULL,
                        voice_id TEXT NOT NULL,
                        configuration_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.commit()
        except Exception as e:
            logger.error("Failed to initialize SQLite database schema: %s", type(e).__name__)
            raise DatabasePersistenceError("Failed to initialize database.") from e

    def save_agent(
        self,
        agent_id: str,
        name: str,
        business_type: str,
        voice_id: str,
        configuration_json: str,
        created_at: str,
    ) -> int:
        """Persists agent record to SQLite using parameterized SQL queries."""
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO agents (
                        agent_id, name, business_type, voice_id, configuration_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (agent_id, name, business_type, voice_id, configuration_json, created_at),
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error("Failed to save agent record %s: %s", agent_id, type(e).__name__)
            raise DatabasePersistenceError(f"Failed to persist agent record to database: {type(e).__name__}") from e

    def list_agents(self) -> List[Dict[str, Any]]:
        """Retrieves all registered agent records ordered by id descending."""
        try:
            with self._connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, agent_id, name, business_type, voice_id, created_at
                    FROM agents
                    ORDER BY id DESC;
                    """
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error("Failed to list agent records: %s", type(e).__name__)
            raise DatabasePersistenceError("Database query failed.") from e

    def get_agent_by_id(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an agent record by ElevenLabs agent_id."""
        try:
            with self._connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, agent_id, name, business_type, voice_id, configuration_json, created_at
                    FROM agents
                    WHERE agent_id = ?;
                    """,
                    (agent_id,),
                )
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error("Failed to query agent record: %s", type(e).__name__)
            raise DatabasePersistenceError("Database query failed.") from e


def get_agent_repository() -> AgentRepository:
    return AgentRepository()
