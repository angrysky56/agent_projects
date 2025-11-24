import { useState, useEffect } from 'react';

export type PermissionAction = 'ALLOW_ALWAYS' | 'DENY_ALWAYS' | 'ASK';
export type RiskLevel = 'SAFE' | 'MODERATE' | 'DANGEROUS';

export interface ToolPermission {
    tool_name: string;
    server_name: string;
    action: PermissionAction;
    risk_level: RiskLevel;
    timestamp: string;
}

/**
 * Hook for managing tool permissions
 */
export function useToolPermissions() {
    const [permissions, setPermissions] = useState<Map<string, ToolPermission>>(new Map());
    const [loading, setLoading] = useState(true);

    const getKey = (toolName: string, serverName: string) => `${serverName}:${toolName}`;

    // Load permissions on mount
    useEffect(() => {
        loadPermissions();
    }, []);

    const loadPermissions = async () => {
        setLoading(true);
        try {
            const response = await fetch('http://localhost:8000/api/tool-permissions');
            const data = await response.json();

            const permMap = new Map<string, ToolPermission>();
            data.permissions.forEach((perm: ToolPermission) => {
                const key = getKey(perm.tool_name, perm.server_name);
                permMap.set(key, perm);
            });

            setPermissions(permMap);
        } catch (error) {
            console.error('Failed to load permissions:', error);
        } finally {
            setLoading(false);
        }
    };

    const checkPermission = (toolName: string, serverName: string): ToolPermission | undefined => {
        const key = getKey(toolName, serverName);
        return permissions.get(key);
    };

    const savePermission = async (permission: Omit<ToolPermission, 'timestamp'>) => {
        try {
            const response = await fetch('http://localhost:8000/api/tool-permissions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(permission),
            });

            if (!response.ok) throw new Error('Failed to save permission');

            // Update local state
            const key = getKey(permission.tool_name, permission.server_name);
            setPermissions(prev => {
                const newMap = new Map(prev);
                newMap.set(key, { ...permission, timestamp: new Date().toISOString() });
                return newMap;
            });

            return true;
        } catch (error) {
            console.error('Error saving permission:', error);
            return false;
        }
    };

    const removePermission = async (toolName: string, serverName: string) => {
        try {
            const response = await fetch(
                `http://localhost:8000/api/tool-permissions/${serverName}/${toolName}`,
                { method: 'DELETE' }
            );

            if (!response.ok) throw new Error('Failed to remove permission');

            // Update local state
            const key = getKey(toolName, serverName);
            setPermissions(prev => {
                const newMap = new Map(prev);
                newMap.delete(key);
                return newMap;
            });

            return true;
        } catch (error) {
            console.error('Error removing permission:', error);
            return false;
        }
    };

    const clearAll = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/tool-permissions', {
                method: 'DELETE',
            });

            if (!response.ok) throw new Error('Failed to clear permissions');

            setPermissions(new Map());
            return true;
        } catch (error) {
            console.error('Error clearing permissions:', error);
            return false;
        }
    };

    return {
        permissions: Array.from(permissions.values()),
        loading,
        checkPermission,
        savePermission,
        removePermission,
        clearAll,
        reload: loadPermissions,
    };
}
