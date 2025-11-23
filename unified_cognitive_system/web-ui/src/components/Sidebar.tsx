import React, { useEffect } from 'react';
import { Brain, MessageSquare, Settings, Wrench, Menu, X, Plus } from 'lucide-react';
import { useAppStore } from '../store';
import { api } from '../services/api';
import clsx from 'clsx';

export const Sidebar: React.FC = () => {
    const {
        sidebarOpen,
        toggleSidebar,
        useCompass,
        toggleCompass,
        currentView,
        setCurrentView,
        conversations,
        setConversations,
        currentConversationId,
        loadConversation,
        clearMessages,
        setCurrentConversationId
    } = useAppStore();

    useEffect(() => {
        loadConversations();
    }, []);

    const loadConversations = async () => {
        try {
            const data = await api.getConversations();
            setConversations(data);
        } catch (error) {
            console.error('Failed to load conversations:', error);
        }
    };

    const handleNewChat = () => {
        clearMessages();
        setCurrentConversationId(null);
        setCurrentView('chat');
        if (window.innerWidth < 1024) toggleSidebar();
    };

    const handleSelectConversation = async (id: string) => {
        try {
            const data = await api.getConversation(id);
            // Transform backend messages to frontend format if needed
            // Assuming backend returns { messages: [...] }
            loadConversation(id, data.messages);
            setCurrentView('chat');
            if (window.innerWidth < 1024) toggleSidebar();
        } catch (error) {
            console.error('Failed to load conversation:', error);
        }
    };

    return (
        <>
            {/* Mobile Toggle */}
            <button
                onClick={toggleSidebar}
                className="lg:hidden fixed top-4 left-4 z-50 glass-card p-2 rounded-lg"
            >
                {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>

            {/* Sidebar */}
            <div
                className={clsx(
                    'fixed lg:relative inset-y-0 left-0 z-40 w-64 glass-card border-r border-white/10',
                    'transform transition-transform duration-300 ease-in-out lg:translate-x-0',
                    sidebarOpen ? 'translate-x-0' : '-translate-x-full'
                )}
            >
                <div className="flex flex-col h-full p-4">
                    {/* Logo */}
                    <div className="flex items-center gap-3 mb-6">
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary-500 to-purple-600 flex items-center justify-center">
                            <Brain className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h1 className="font-bold text-lg gradient-text">COMPASS</h1>
                            <p className="text-xs text-slate-400">Cognitive AI System</p>
                        </div>
                    </div>

                    {/* New Chat Button */}
                    <button
                        onClick={handleNewChat}
                        className="w-full flex items-center justify-center gap-2 px-4 py-3 mb-6 rounded-lg
                                 bg-primary-600 hover:bg-primary-500 text-white font-medium transition-all duration-200 shadow-lg hover:shadow-primary-500/25"
                    >
                        <Plus className="w-5 h-5" />
                        <span>New Chat</span>
                    </button>

                    {/* Navigation */}
                    <nav className="space-y-1 mb-6">
                        <button
                            onClick={() => setCurrentView('chat')}
                            className={clsx(
                                'w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-200',
                                currentView === 'chat' && !currentConversationId
                                    ? 'bg-white/10 text-white'
                                    : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
                            )}
                        >
                            <MessageSquare className="w-4 h-4" />
                            <span className="text-sm font-medium">Current Chat</span>
                        </button>

                        <button
                            onClick={toggleCompass}
                            className={clsx(
                                'w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-200',
                                useCompass
                                    ? 'bg-primary-500/10 text-primary-300'
                                    : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
                            )}
                        >
                            <Brain className="w-4 h-4" />
                            <span className="text-sm font-medium">COMPASS</span>
                            <div className={clsx(
                                'ml-auto w-2 h-2 rounded-full',
                                useCompass ? 'bg-green-500 animate-pulse' : 'bg-slate-600'
                            )} />
                        </button>

                        <button
                            onClick={() => setCurrentView('mcp')}
                            className={clsx(
                                'w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-200',
                                currentView === 'mcp'
                                    ? 'bg-white/10 text-white'
                                    : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
                            )}
                        >
                            <Wrench className="w-4 h-4" />
                            <span className="text-sm font-medium">MCP Tools</span>
                        </button>

                        <button
                            onClick={() => setCurrentView('settings')}
                            className={clsx(
                                'w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-200',
                                currentView === 'settings'
                                    ? 'bg-white/10 text-white'
                                    : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
                            )}
                        >
                            <Settings className="w-4 h-4" />
                            <span className="text-sm font-medium">Settings</span>
                        </button>
                    </nav>

                    {/* Chat History */}
                    <div className="flex-1 overflow-hidden flex flex-col min-h-0">
                        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 px-2">
                            History
                        </h3>
                        <div className="flex-1 overflow-y-auto custom-scrollbar space-y-1 pr-2">
                            {conversations.map((conv) => (
                                <button
                                    key={conv.id}
                                    onClick={() => handleSelectConversation(conv.id)}
                                    className={clsx(
                                        'w-full text-left px-3 py-2 rounded-lg text-sm transition-colors truncate',
                                        currentConversationId === conv.id
                                            ? 'bg-white/10 text-white'
                                            : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
                                    )}
                                >
                                    {conv.title || 'New Conversation'}
                                </button>
                            ))}
                            {conversations.length === 0 && (
                                <div className="text-xs text-slate-600 px-3 italic">
                                    No history yet
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Footer */}
                    <div className="pt-4 border-t border-white/10 mt-auto">
                        <div className="text-xs text-slate-500 text-center">
                            COMPASS v0.1.0
                        </div>
                    </div>
                </div>
            </div>

            {/* Overlay for mobile */}
            {sidebarOpen && (
                <div
                    className="lg:hidden fixed inset-0 bg-black/50 z-30"
                    onClick={toggleSidebar}
                />
            )}
        </>
    );
};
