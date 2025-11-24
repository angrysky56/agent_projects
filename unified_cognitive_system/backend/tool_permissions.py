"""
Tool Permission Management System

Manages user permissions for MCP tool execution with risk-based classification.
Implements Allow Once, Always Allow, and Deny authorization patterns.
"""

from typing import Dict, Literal, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Type definitions
PermissionLevel = Literal["SAFE", "MODERATE", "DANGEROUS"]
PermissionAction = Literal["ALLOW_ALWAYS", "DENY_ALWAYS", "ASK"]


@dataclass
class ToolPermission:
    """Represents a stored permission decision for a tool."""

    tool_name: str
    server_name: str
    action: PermissionAction
    risk_level: PermissionLevel
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ToolPermission":
        return cls(**data)


def classify_tool_risk(tool_name: str, tool_description: str = "") -> PermissionLevel:
    """
    Classify a tool's risk level based on its name and description.

    Args:
        tool_name: Name of the tool
        tool_description: Optional description

    Returns:
        Risk level: SAFE, MODERATE, or DANGEROUS
    """
    tool_lower = tool_name.lower()
    desc_lower = tool_description.lower()
    combined = f"{tool_lower} {desc_lower}"

    # Dangerous patterns (irreversible, high-impact)
    dangerous_patterns = ["delete", "remove", "rm", "kill", "execute", "exec", "run", "drop", "truncate", "destroy", "format", "wipe", "clear", "sudo", "admin", "root"]

    if any(pattern in combined for pattern in dangerous_patterns):
        return "DANGEROUS"

    # Moderate patterns (state changes, reversible)
    moderate_patterns = ["write", "create", "modify", "update", "move", "rename", "copy", "set", "change", "edit", "send", "post", "put"]

    if any(pattern in combined for pattern in moderate_patterns):
        return "MODERATE"

    # Default to safe (read-only, no side effects)
    return "SAFE"


class ToolPermissionManager:
    """Manages tool permissions with persistent storage."""

    def __init__(self, config_path: str = ".tool_permissions.json"):
        self.config_path = Path(config_path)
        self.permissions: Dict[str, ToolPermission] = {}
        self.load()

    def _get_key(self, tool_name: str, server_name: str) -> str:
        """Generate unique key for tool-server combination."""
        return f"{server_name}:{tool_name}"

    def check_permission(self, tool_name: str, server_name: str) -> Optional[PermissionAction]:
        """
        Check if a tool has a stored permission.

        Args:
            tool_name: Name of the tool
            server_name: Name of the MCP server

        Returns:
            Permission action if stored, None if should ask
        """
        key = self._get_key(tool_name, server_name)
        perm = self.permissions.get(key)
        return perm.action if perm else None

    def set_permission(self, tool_name: str, server_name: str, action: PermissionAction, risk_level: PermissionLevel) -> None:
        """
        Store a permission decision.

        Args:
            tool_name: Name of the tool
            server_name: Name of the MCP server
            action: Permission action to store
            risk_level: Risk level of the tool
        """
        key = self._get_key(tool_name, server_name)

        self.permissions[key] = ToolPermission(tool_name=tool_name, server_name=server_name, action=action, risk_level=risk_level, timestamp=datetime.now().isoformat())

        self.save()
        logger.info(f"Permission saved: {tool_name} ({server_name}) -> {action}")

    def remove_permission(self, tool_name: str, server_name: str) -> bool:
        """
        Remove a stored permission.

        Returns:
            True if permission was removed, False if not found
        """
        key = self._get_key(tool_name, server_name)
        if key in self.permissions:
            del self.permissions[key]
            self.save()
            logger.info(f"Permission removed: {tool_name} ({server_name})")
            return True
        return False

    def list_permissions(self) -> List[ToolPermission]:
        """Get all stored permissions."""
        return list(self.permissions.values())

    def clear_all(self) -> None:
        """Clear all stored permissions."""
        self.permissions.clear()
        self.save()
        logger.info("All permissions cleared")

    def load(self) -> None:
        """Load permissions from file."""
        if not self.config_path.exists():
            logger.info(f"No permissions file found at {self.config_path}, starting fresh")
            return

        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)
                self.permissions = {key: ToolPermission.from_dict(perm_data) for key, perm_data in data.items()}
            logger.info(f"Loaded {len(self.permissions)} permissions from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading permissions: {e}", exc_info=True)
            self.permissions = {}

    def save(self) -> None:
        """Save permissions to file."""
        try:
            data = {key: perm.to_dict() for key, perm in self.permissions.items()}

            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self.permissions)} permissions to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving permissions: {e}", exc_info=True)


# Global permission manager instance
_permission_manager: Optional[ToolPermissionManager] = None


def get_permission_manager() -> ToolPermissionManager:
    """Get or create the global permission manager instance."""
    global _permission_manager
    if _permission_manager is None:
        _permission_manager = ToolPermissionManager()
    return _permission_manager
