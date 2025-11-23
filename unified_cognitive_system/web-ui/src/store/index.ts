// Zustand store for global state management

import { create } from 'zustand';
import type { Message, ProviderType, ProviderInfo, MCPTool, MCPServer } from '../types';

interface AppState {
    // Chat state
    messages: Message[];
    addMessage: (message: Message) => void;
    clearMessages: () => void;

    // Provider state
    currentProvider: ProviderType;
    currentModel: string | null;
    providers: Record<ProviderType, ProviderInfo>;
    setCurrentProvider: (provider: ProviderType) => void;
    setCurrentModel: (model: string) => void;
    setProviders: (providers: Record<ProviderType, ProviderInfo>) => void;

    // COMPASS state
    useCompass: boolean;
    autoConfig: boolean;
    compassTrace: string[];
    toggleCompass: () => void;
    toggleAutoConfig: () => void;
    addCompassStep: (step: string) => void;
    clearCompassTrace: () => void;

    // MCP state
    mcpServers: MCPServer[];
    mcpTools: MCPTool[];
    setMCPServers: (servers: MCPServer[]) => void;
    setMCPTools: (tools: MCPTool[]) => void;

    // Conversation state
    conversations: any[];
    currentConversationId: string | null;
    setConversations: (conversations: any[]) => void;
    setCurrentConversationId: (id: string | null) => void;
    loadConversation: (id: string, messages: Message[]) => void;

    // UI state
    sidebarOpen: boolean;
    currentView: 'chat' | 'mcp' | 'settings';
    setCurrentView: (view: 'chat' | 'mcp' | 'settings') => void;
    toggleSidebar: () => void;
}

export const useAppStore = create<AppState>((set) => ({
    // Chat state
    messages: [],
    addMessage: (message) => set((state) => ({
        messages: [...state.messages, { ...message, timestamp: Date.now() }]
    })),
    clearMessages: () => set({ messages: [] }),

    // Provider state
    currentProvider: 'ollama',
    currentModel: null,
    providers: {} as Record<ProviderType, ProviderInfo>,
    setCurrentProvider: (provider) => set({ currentProvider: provider }),
    setCurrentModel: (model) => set({ currentModel: model }),
    setProviders: (providers) => set({ providers }),

    // COMPASS state
    useCompass: true,
    autoConfig: true,
    compassTrace: [],
    toggleCompass: () => set((state) => ({ useCompass: !state.useCompass })),
    toggleAutoConfig: () => set((state) => ({ autoConfig: !state.autoConfig })),
    addCompassStep: (step) => set((state) => ({ compassTrace: [...state.compassTrace, step] })),
    clearCompassTrace: () => set({ compassTrace: [] }),

    // MCP state
    mcpServers: [],
    mcpTools: [],
    setMCPServers: (servers) => set({ mcpServers: servers }),
    setMCPTools: (tools) => set({ mcpTools: tools }),

    // Conversation state
    conversations: [],
    currentConversationId: null,
    setConversations: (conversations) => set({ conversations }),
    setCurrentConversationId: (id) => set({ currentConversationId: id }),
    loadConversation: (id, messages) => set({
        currentConversationId: id,
        messages: messages,
        compassTrace: [] // Clear trace when loading new conversation
    }),

    // UI state
    sidebarOpen: true,
    currentView: 'chat',
    setCurrentView: (view) => set({ currentView: view }),
    toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
}));
