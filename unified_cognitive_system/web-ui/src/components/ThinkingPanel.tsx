import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Brain, Zap } from 'lucide-react';

interface ThinkingStep {
    timestamp: string | null;
    content: string;
}

interface ThinkingPanelProps {
    steps: ThinkingStep[];
}

export const ThinkingPanel: React.FC<ThinkingPanelProps> = ({ steps }) => {
    const [isExpanded, setIsExpanded] = useState(true);

    if (!steps || steps.length === 0) return null;

    return (
        <div className="my-4 border border-primary-500/30 rounded-lg overflow-hidden bg-slate-900/50 backdrop-blur-sm">
            {/* Header */}
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-slate-800/50 transition-colors"
            >
                <div className="flex items-center gap-2">
                    <Brain className="w-4 h-4 text-primary-400" />
                    <span className="font-medium text-sm text-primary-300">
                        COMPASS Thinking Process
                    </span>
                    <span className="text-xs text-slate-400">
                        ({steps.length} steps)
                    </span>
                </div>
                {isExpanded ? (
                    <ChevronDown className="w-4 h-4 text-slate-400" />
                ) : (
                    <ChevronRight className="w-4 h-4 text-slate-400" />
                )}
            </button>

            {/* Content */}
            {isExpanded && (
                <div className="px-4 py-3 border-t border-white/10">
                    <div className="space-y-2">
                        {steps.map((step, idx) => (
                            <div
                                key={idx}
                                className="flex gap-3 text-sm"
                            >
                                <div className="flex-shrink-0 mt-1">
                                    <Zap className="w-3 h-3 text-primary-500" />
                                </div>
                                <div className="flex-1">
                                    <div className="text-slate-300 font-mono text-xs leading-relaxed">
                                        {step.content}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};
