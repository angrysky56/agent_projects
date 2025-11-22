import React from 'react';
import { Brain, MessageSquare, Settings, Wrench, Menu, X } from 'lucide-react';
import { useAppStore } from '../store';
import clsx from 'clsx';

export const Sidebar: React.FC = () => {
    const { sidebarOpen, toggleSidebar, useCompass, toggleCompass, currentView, setCurrentView } = useAppStore();


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
                    <div className="flex items-center gap-3 mb-8">
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary-500 to-purple-600 flex items-center justify-center">
                            <Brain className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h1 className="font-bold text-lg gradient-text">COMPASS</h1>
                            <p className="text-xs text-slate-400">Cognitive AI System</p>
                        </div>
                    </div>

                    {/* Navigation */}
                    <nav className="flex-1 space-y-2">
                        <button
                            onClick={() => setCurrentView('chat')}
                            className={clsx(
                                'w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200',
                                currentView === 'chat'
                                    ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                                    : 'text-slate-300 hover:bg-white/5'
                            )}
                        >
                            <MessageSquare className="w-5 h-5" />
                            <span className="font-medium">Chat</span>
                        </button>

                        <button
                            onClick={toggleCompass}
                            className={clsx(
                                'w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200',
                                useCompass
                                    ? 'bg-primary-500/10 text-primary-300'
                                    : 'text-slate-300 hover:bg-white/5'
                            )}
                        >
                            <Brain className="w-5 h-5" />
                            <span className="font-medium">COMPASS</span>
                            <div className={clsx(
                                'ml-auto w-2 h-2 rounded-full',
                                useCompass ? 'bg-green-500 animate-pulse' : 'bg-slate-600'
                            )} />
                        </button>

                        <button
                            onClick={() => setCurrentView('mcp')}
                            className={clsx(
                                'w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200',
                                currentView === 'mcp'
                                    ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                                    : 'text-slate-300 hover:bg-white/5'
                            )}
                        >
                            <Wrench className="w-5 h-5" />
                            <span className="font-medium">MCP Tools</span>
                        </button>

                        <button
                            onClick={() => setCurrentView('settings')}
                            className={clsx(
                                'w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200',
                                currentView === 'settings'
                                    ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                                    : 'text-slate-300 hover:bg-white/5'
                            )}
                        >
                            <Settings className="w-5 h-5" />
                            <span className="font-medium">Settings</span>
                        </button>
                    </nav>

                    {/* Footer */}
                    <div className="pt-4 border-t border-white/10">
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
