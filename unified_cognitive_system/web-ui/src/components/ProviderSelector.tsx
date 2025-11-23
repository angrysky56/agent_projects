import React, { useEffect, useState } from 'react';
import { Check, Wifi } from 'lucide-react';
import { useAppStore } from '../store';
import { api } from '../services/api';
import type { ProviderType, ProviderInfo } from '../types';
import clsx from 'clsx';

const providerNames: Record<ProviderType, string> = {
    ollama: 'Ollama',
    lm_studio: 'LM Studio',
    openai: 'OpenAI',
    anthropic: 'Anthropic',
};

const providerIcons: Record<ProviderType, string> = {
    ollama: '🦙',
    lm_studio: '💻',
    openai: '🤖',
    anthropic: '🏛️',
};

export const ProviderSelector: React.FC = () => {
    const { currentProvider, currentModel, providers, setCurrentProvider, setCurrentModel, setProviders } = useAppStore();
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadProviders();
    }, []);

    const loadProviders = async () => {
        try {
            const data = await api.getProviders();
            setProviders(data as Record<ProviderType, ProviderInfo>);
        } catch (error) {
            console.error('Failed to load providers:', error);
        } finally {
            setLoading(false);
        }
    };

    const currentProviderInfo = providers[currentProvider];
    const availableModels = currentProviderInfo?.models || [];

    return (
        <div className="glass-card p-4 space-y-4">
            <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                <Wifi className="w-4 h-4" />
                LLM Provider
            </h3>

            {loading ? (
                <div className="text-center py-4 text-slate-400">Loading providers...</div>
            ) : (
                <>
                    {/* Provider Grid */}
                    <div className="grid grid-cols-2 gap-2">
                        {(Object.keys(providerNames) as ProviderType[]).map((type) => {
                            const info = providers[type];
                            const isSelected = currentProvider === type;
                            const isAvailable = info?.available || false;

                            return (
                                <button
                                    key={type}
                                    onClick={() => isAvailable && setCurrentProvider(type)}
                                    disabled={!isAvailable}
                                    className={clsx(
                                        'relative p-3 rounded-lg border transition-all duration-200',
                                        isSelected
                                            ? 'border-primary-500 bg-primary-500/20'
                                            : 'border-white/10 bg-white/5 hover:bg-white/10',
                                        !isAvailable && 'opacity-50 cursor-not-allowed'
                                    )}
                                >
                                    <div className="flex items-center justify-between mb-1">
                                        <span className="text-2xl">{providerIcons[type]}</span>
                                        {isAvailable ? (
                                            <div className="status-indicator status-online" />
                                        ) : (
                                            <div className="status-indicator status-offline" />
                                        )}
                                    </div>
                                    <div className="text-sm font-medium text-slate-200">
                                        {providerNames[type]}
                                    </div>
                                    {isAvailable && (
                                        <div className="text-xs text-slate-400 mt-1">
                                            {info.models.length} model{info.models.length !== 1 ? 's' : ''}
                                        </div>
                                    )}
                                    {isSelected && (
                                        <div className="absolute top-2 right-2">
                                            <Check className="w-4 h-4 text-primary-400" />
                                        </div>
                                    )}
                                </button>
                            );
                        })}
                    </div>

                    {/* Model Selector */}
                    {availableModels.length > 0 && (
                        <div>
                            <label className="block text-xs text-slate-400 mb-2">Model</label>
                            <select
                                value={currentModel || ''}
                                onChange={(e) => setCurrentModel(e.target.value)}
                                className="input-field text-sm"
                            >
                                <option value="">Default</option>
                                {availableModels.map((model) => (
                                    <option key={model} value={model}>
                                        {model}
                                    </option>
                                ))}
                            </select>
                        </div>
                    )}
                </>
            )}
        </div>
    );
};
