
import { Sidebar } from './components/Sidebar';
import { ChatInterface } from './components/ChatInterface';
import { ProviderSelector } from './components/ProviderSelector';
import { MCPTools } from './components/MCPTools';
import { useAppStore } from './store';
import './index.css';

function App() {
    const { useCompass, toggleCompass, currentView } = useAppStore();

    return (
        <div className="flex h-screen">
            <Sidebar />

            <div className="flex-1 flex flex-col lg:flex-row">
                {/* Main Content Area */}
                <div className="flex-1 flex flex-col h-full overflow-hidden">
                    {currentView === 'chat' && <ChatInterface />}

                    {currentView === 'mcp' && <MCPTools />}

                    {currentView === 'settings' && (
                        <div className="flex-1 p-8 overflow-y-auto">
                            <h2 className="text-2xl font-bold text-white mb-6">Settings</h2>
                            <div className="glass-card p-6">
                                <h3 className="text-lg font-semibold text-white mb-4">Application Settings</h3>
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                                        <div>
                                            <p className="font-medium text-slate-200">COMPASS Auto-Configuration</p>
                                            <p className="text-xs text-slate-400">Automatically optimize framework parameters</p>
                                        </div>
                                        <div className="w-10 h-6 bg-primary-600 rounded-full relative">
                                            <div className="absolute right-1 top-1 w-4 h-4 bg-white rounded-full"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Panel - Only show in chat view */}
                {currentView === 'chat' && (
                    <div className="w-full lg:w-80 border-l border-white/10 p-4 space-y-4 overflow-y-auto custom-scrollbar bg-slate-900/30">
                        <ProviderSelector />

                        {/* COMPASS Toggle */}
                        <div className="glass-card p-4">
                            <div className="flex items-center justify-between mb-2">
                                <h3 className="text-sm font-semibold text-slate-300">COMPASS</h3>
                                <button
                                    onClick={toggleCompass}
                                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${useCompass ? 'bg-primary-600' : 'bg-slate-600'
                                        }`}
                                >
                                    <span
                                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${useCompass ? 'translate-x-6' : 'translate-x-1'
                                            }`}
                                    />
                                </button>
                            </div>
                            <p className="text-xs text-slate-400">
                                {useCompass
                                    ? 'Auto-configuring cognitive frameworks based on task complexity'
                                    : 'Using direct LLM responses'}
                            </p>
                        </div>

                        {/* COMPASS Trace / Info */}
                        <div className="glass-card p-4 flex-1 overflow-hidden flex flex-col min-h-0">
                            <h3 className="text-sm font-semibold text-slate-300 mb-2 flex items-center justify-between">
                                <div className="flex flex-col">
                                    <span>System Trace</span>
                                    {useAppStore.getState().currentModel && (
                                        <span className="text-[10px] text-slate-500 font-normal">
                                            Model: {useAppStore.getState().currentModel}
                                        </span>
                                    )}
                                </div>
                                {useAppStore.getState().compassTrace.length > 0 && (
                                    <span className="text-xs text-primary-400 animate-pulse">● Live</span>
                                )}
                            </h3>

                            {useAppStore.getState().compassTrace.length > 0 ? (
                                <div className="flex-1 overflow-y-auto custom-scrollbar space-y-2 pr-2">
                                    {useAppStore.getState().compassTrace.map((step, idx) => (
                                        <div key={idx} className="text-xs font-mono text-slate-400 border-l-2 border-slate-700 pl-2 py-0.5 animate-fadeIn">
                                            {step}
                                        </div>
                                    ))}
                                    <div ref={(el) => el?.scrollIntoView({ behavior: 'smooth' })} />
                                </div>
                            ) : (
                                <div className="text-xs text-slate-500 italic">
                                    Waiting for task execution...
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
