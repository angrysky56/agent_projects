// API service for backend communication

import type { ChatRequest, ProviderInfo, MCPTool } from '../types';

const API_BASE = '';  // Vite proxy handles this

export const api = {
    // Provider endpoints
    async getProviders(): Promise<Record<string, ProviderInfo>> {
        const res = await fetch(`${API_BASE}/api/providers`);
        return res.json();
    },

    // Chat endpoints
    async chatCompletion(request: ChatRequest) {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        });

        if (!res.ok) throw new Error('Chat request failed');

        if (request.stream) {
            return res.body;
        } else {
            return res.json();
        }
    },

    // MCP endpoints
    async getMCPServers(): Promise<{ connected: string[]; all_servers: string[] }> {
        const res = await fetch(`${API_BASE}/api/mcp/servers`);
        return res.json();
    },

    async getMCPTools(serverName?: string): Promise<{ tools: MCPTool[] }> {
        const url = serverName
            ? `${API_BASE}/api/mcp/tools?server_name=${serverName}`
            : `${API_BASE}/api/mcp/tools`;
        const res = await fetch(url);
        return res.json();
    },

    async callMCPTool(toolName: string, args: Record<string, any>, serverName?: string) {
        const res = await fetch(`${API_BASE}/api/mcp/call-tool`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tool_name: toolName,
                arguments: args,
                server_name: serverName,
            }),
        });
        return res.json();
    },

    // Conversation endpoints
    async getConversations() {
        const res = await fetch(`${API_BASE}/api/conversations`);
        return res.json();
    },

    async getConversation(conversationId: string) {
        const res = await fetch(`${API_BASE}/api/conversations/${conversationId}`);
        return res.json();
    },

    // COMPASS endpoints
    async getCOMPASSStatus() {
        const res = await fetch(`${API_BASE}/api/compass/status`);
        return res.json();
    },

    async getCOMPASSTrace() {
        const res = await fetch(`${API_BASE}/api/compass/trace`);
        return res.json();
    },
};
