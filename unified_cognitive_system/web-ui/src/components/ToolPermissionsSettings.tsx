import React, { useEffect } from 'react';
import { useToolPermissions, type RiskLevel } from '../hooks/useToolPermissions';

const getRiskBadge = (risk: RiskLevel) => {
    switch (risk) {
        case 'SAFE':
            return <span className="text-green-400">🟢 SAFE</span>;
        case 'MODERATE':
            return <span className="text-yellow-400">🟡 MODERATE</span>;
        case 'DANGEROUS':
            return <span className="text-red-400">🔴 DANGEROUS</span>;
    }
};

const getActionBadge = (action: string) => {
    switch (action) {
        case 'ALLOW_ALWAYS':
            return <span className="text-green-400">ALLOW ALWAYS</span>;
        case 'DENY_ALWAYS':
            return <span className="text-red-400">DENY ALWAYS</span>;
        case 'ASK':
            return <span className="text-blue-400">ASK</span>;
        default:
            return <span className="text-slate-400">{action}</span>;
    }
};

export const ToolPermissionsSettings: React.FC = () => {
    const {
        permissions,
        loading,
        removePermission,
        clearAll,
        reload,
    } = useToolPermissions();

    useEffect(() => {
        reload();
    }, []);

    const handleRemove = async (toolName: string, serverName: string) => {
        if (confirm(`Remove permission for ${toolName}?`)) {
            await removePermission(toolName, serverName);
        }
    };

    const handleClearAll = async () => {
        if (confirm('Clear all tool permissions? This cannot be undone.')) {
            await clearAll();
        }
    };

    return (
        <div className="flex-1 p-8 overflow-y-auto custom-scrollbar">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="mb-6">
                    <h1 className="text-2xl font-bold text-slate-100 mb-2">
                        🔐 Tool Permissions
                    </h1>
                    <p className="text-slate-400 text-sm">
                        Manage authorization settings for MCP tool execution
                    </p>
                </div>

                {/* Actions */}
                <div className="flex gap-3 mb-6">
                    <button
                        onClick={reload}
                        className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg text-sm font-medium transition-colors"
                    >
                        🔄 Refresh
                    </button>
                    <button
                        onClick={handleClearAll}
                        disabled={permissions.length === 0}
                        className="px-4 py-2 bg-red-600 hover:bg-red-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium transition-colors"
                    >
                        🗑️ Clear All
                    </button>
                </div>

                {/* Permissions List */}
                {loading ? (
                    <div className="text-center py-12 text-slate-400">
                        Loading permissions...
                    </div>
                ) : permissions.length === 0 ? (
                    <div className="text-center py-12 bg-slate-800 rounded-lg border border-slate-700">
                        <div className="text-4xl mb-3">🔓</div>
                        <div className="text-slate-400">No saved permissions</div>
                        <div className="text-sm text-slate-500 mt-2">
                            Permissions will appear here after you authorize tools
                        </div>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {permissions.map((perm) => (
                            <div
                                key={`${perm.server_name}:${perm.tool_name}`}
                                className="bg-slate-800 border border-slate-700 rounded-lg p-4 hover:border-slate-600 transition-colors"
                            >
                                <div className="flex items-start justify-between gap-4">
                                    <div className="flex-1 space-y-2">
                                        {/* Tool Name */}
                                        <div>
                                            <span className="font-mono text-primary-300 font-medium">
                                                {perm.tool_name}
                                            </span>
                                            <span className="text-slate-500 text-sm ml-2">
                                                ({perm.server_name})
                                            </span>
                                        </div>

                                        {/* Badges */}
                                        <div className="flex items-center gap-4 text-sm">
                                            <div className="flex items-center gap-2">
                                                <span className="text-slate-500">Risk:</span>
                                                {getRiskBadge(perm.risk_level)}
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <span className="text-slate-500">Action:</span>
                                                {getActionBadge(perm.action)}
                                            </div>
                                        </div>

                                        {/* Timestamp */}
                                        <div className="text-xs text-slate-500">
                                            Saved: {new Date(perm.timestamp).toLocaleString()}
                                        </div>
                                    </div>

                                    {/* Remove Button */}
                                    <button
                                        onClick={() => handleRemove(perm.tool_name, perm.server_name)}
                                        className="px-3 py-1.5 text-sm bg-slate-700 hover:bg-red-600 text-slate-300 hover:text-white rounded transition-colors"
                                    >
                                        Remove
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Info Section */}
                <div className="mt-8 p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
                    <h3 className="font-semibold text-slate-200 mb-2 text-sm">
                        ℹ️ About Tool Permissions
                    </h3>
                    <div className="text-xs text-slate-400 space-y-1">
                        <p>• <strong>Allow Always:</strong> Tool will execute without asking</p>
                        <p>• <strong>Deny Always:</strong> Tool execution will be blocked</p>
                        <p>• <strong>Ask:</strong> You'll be prompted each time (default)</p>
                    </div>
                </div>
            </div>
        </div>
    );
};
