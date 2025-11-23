"""
Conversation Manager - Persistence and State Management

Manages conversation history using SQLite database and JSON file backup.
Provides CRUD operations for conversations and messages.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid

from .core.utils import COMPASSLogger


class ConversationManager:
    """Manages conversation persistence and retrieval."""

    def __init__(self, data_dir: str = "./chat_history", logger: Optional[COMPASSLogger] = None):
        """
        Initialize conversation manager.

        Args:
            data_dir: Directory for storing conversations
            logger: Optional logger instance
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "conversations.db"
        self.json_dir = self.data_dir / "json"
        self.json_dir.mkdir(exist_ok=True)

        self.logger = logger or COMPASSLogger("ConversationManager")

        self._init_database()
        self.logger.info(f"Conversation manager initialized (DB: {self.db_path})")

    def _init_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                metadata TEXT
            )
        """)

        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                thinking TEXT,
                tool_calls TEXT,
                attachments TEXT,
                reasoning TEXT,
                timestamp INTEGER NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, timestamp)")

        conn.commit()
        conn.close()

    def create_conversation(self, title: str = "New Conversation", metadata: Optional[Dict] = None) -> str:
        """
        Create a new conversation.

        Args:
            title: Conversation title
            metadata: Optional metadata dictionary

        Returns:
            Conversation ID
        """
        conv_id = str(uuid.uuid4())
        now = int(datetime.now().timestamp() * 1000)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("INSERT INTO conversations (id, title, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?)", (conv_id, title, now, now, json.dumps(metadata or {})))

        conn.commit()
        conn.close()

        self.logger.info(f"Created conversation: {conv_id} - {title}")
        return conv_id

    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> str:
        """
        Add a message to a conversation.

        Args:
            conversation_id: Target conversation ID
            message: Message dictionary

        Returns:
            Message ID
        """
        msg_id = str(uuid.uuid4())
        now = int(datetime.now().timestamp() * 1000)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Insert message
        cursor.execute(
            """INSERT INTO messages
            (id, conversation_id, role, content, thinking, tool_calls, attachments, reasoning, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg_id, conversation_id, message.get("role", "user"), message.get("content", ""), json.dumps(message.get("thinking", [])), json.dumps(message.get("tool_calls", [])), json.dumps(message.get("attachments", [])), json.dumps(message.get("reasoning")), message.get("timestamp", now)),
        )

        # Update conversation timestamp
        cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))

        conn.commit()
        conn.close()

        # Save JSON backup
        self._save_json_backup(conversation_id)

        return msg_id

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a full conversation with all messages.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation dictionary or None if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get conversation metadata
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        conv_row = cursor.fetchone()

        if not conv_row:
            conn.close()
            return None

        # Get all messages
        cursor.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC", (conversation_id,))
        msg_rows = cursor.fetchall()

        conn.close()

        # Build conversation object
        conversation = {"id": conv_row["id"], "title": conv_row["title"], "created_at": conv_row["created_at"], "updated_at": conv_row["updated_at"], "metadata": json.loads(conv_row["metadata"]) if conv_row["metadata"] else {}, "messages": []}

        for msg_row in msg_rows:
            message = {"id": msg_row["id"], "role": msg_row["role"], "content": msg_row["content"], "timestamp": msg_row["timestamp"]}

            # Add optional fields if present
            if msg_row["thinking"]:
                message["thinking"] = json.loads(msg_row["thinking"])
            if msg_row["tool_calls"]:
                message["tool_calls"] = json.loads(msg_row["tool_calls"])
            if msg_row["attachments"]:
                message["attachments"] = json.loads(msg_row["attachments"])
            if msg_row["reasoning"]:
                message["reasoning"] = json.loads(msg_row["reasoning"])

            conversation["messages"].append(message)

        return conversation

    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List conversations (most recent first).

        Args:
            limit: Maximum number of conversations to return
            offset: Offset for pagination

        Returns:
            List of conversation summaries
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?""",
            (limit, offset),
        )

        rows = cursor.fetchall()
        conn.close()

        return [{"id": row["id"], "title": row["title"], "created_at": row["created_at"], "updated_at": row["updated_at"]} for row in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.

        Args:
            conversation_id: Conversation ID to delete

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        # Delete JSON backup
        json_file = self.json_dir / f"{conversation_id}.json"
        if json_file.exists():
            json_file.unlink()

        if deleted:
            self.logger.info(f"Deleted conversation: {conversation_id}")

        return deleted

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title, int(datetime.now().timestamp() * 1000), conversation_id))

        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if updated:
            self._save_json_backup(conversation_id)

        return updated

    def _save_json_backup(self, conversation_id: str):
        """Save conversation as JSON file for backup."""
        conversation = self.get_conversation(conversation_id)
        if conversation:
            json_file = self.json_dir / f"{conversation_id}.json"
            with open(json_file, "w") as f:
                json.dump(conversation, f, indent=2)
