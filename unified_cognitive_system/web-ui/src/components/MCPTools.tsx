import React, { useEffect, useState } from 'react';
import { Server, Wrench, RefreshCw, AlertCircle } from 'lucide-react';
import { api } from '../services/api';
import type { MCPTool } from '../types';

export const MCPTools: React.FC = () => {
    const [tools, setTools] = useState<MCPTool[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [expandedTool, setExpandedTool] = useState<string | null>(null);

    const loadTools = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await api.getMCPTools();
            setTools(data.tools);
        } catch (err) {
            console.error('Failed to load tools:', err);
            setError('Failed to load MCP tools. Please ensure the backend is running.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadTools();
    }, []);

    return (
        <div className="flex-1 p-8 overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                    <Server className="w-8 h-8 text-primary-400" />
                    MCP Tools
                </h2>
                <button
                    onClick={loadTools}
                    className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 transition-colors"
                    title="Refresh Tools"
                >
                    <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
                </button>
            </div>

            {error && (
                <div className="glass-card p-4 border-red-500/30 bg-red-500/10 text-red-200 flex items-center gap-3 mb-6">
                    <AlertCircle className="w-5 h-5" />
                    {error}
                </div>
            )}

            {loading && !tools.length ? (
                <div className="text-center py-12 text-slate-400">
                    Loading available tools...
                </div>
            ) : tools.length === 0 ? (
                <div className="glass-card p-8 text-center">
                    <Wrench className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-slate-300 mb-2">No Tools Available</h3>
                    <p className="text-slate-400">
                        Connect an MCP server to see available tools here.
                    </p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {tools.map((tool) => (
                        <div
                            key={tool.name}
                            className={`glass-card p-5 hover:border-primary-500/30 transition-all duration-200 group cursor-pointer ${expandedTool === tool.name ? 'ring-1 ring-primary-500/50 bg-slate-800/80' : ''}`}
                            onClick={() => setExpandedTool(expandedTool === tool.name ? null : tool.name)}
                        >
                            <div className="flex items-start justify-between mb-3">
                                <div className="p-2 rounded-lg bg-primary-500/10 text-primary-400 group-hover:bg-primary-500/20 transition-colors">
                                    <Wrench className="w-5 h-5" />
                                </div>
                                <span className="text-xs font-mono text-slate-500 bg-slate-900/50 px-2 py-1 rounded">
                                    {tool.server_name || 'default'}
                                </span>
                            </div>
                            <h3 className="font-semibold text-slate-200 mb-2">{tool.name}</h3>
                            <p className={`text-sm text-slate-400 mb-4 ${expandedTool === tool.name ? '' : 'line-clamp-3'}`}>
                                {tool.description || 'No description provided.'}
                            </p>

                            <div className="border-t border-white/5 pt-3 mt-auto">
                                <h4 className="text-xs font-medium text-slate-500 mb-2">Arguments:</h4>
                                <div className="flex flex-wrap gap-1">
                                    {Object.keys(tool.inputSchema?.properties || {}).map((arg) => (
                                        <span key={arg} className="text-xs px-2 py-1 rounded bg-white/5 text-slate-400 font-mono">
                                            {arg}
                                        </span>
                                    ))}
                                    {Object.keys(tool.inputSchema?.properties || {}).length === 0 && (
                                        <span className="text-xs text-slate-600 italic">None</span>
                                    )}
                                </div>

                                {expandedTool === tool.name && tool.inputSchema && (
                                    <div className="mt-4 pt-3 border-t border-white/5 animate-fadeIn">
                                        <h4 className="text-xs font-medium text-slate-500 mb-2">Schema:</h4>
                                        <pre className="text-[10px] font-mono text-slate-400 bg-slate-950/50 p-2 rounded overflow-x-auto">
                                            {JSON.stringify(tool.inputSchema, null, 2)}
                                        </pre>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
