import React from 'react';
import type { RiskLevel } from '../hooks/useToolPermissions';

interface ToolAuthorizationModalProps {
    isOpen: boolean;
    toolName: string;
    serverName: string;
    arguments: Record<string, any>;
    riskLevel: RiskLevel;
    onDeny: () => void;
    onAllowOnce: () => void;
    onAllowAlways: () => void;
}

const getRiskColor = (risk: RiskLevel) => {
    switch (risk) {
        case 'SAFE':
            return 'text-green-400 bg-green-500/10 border-green-500/30';
        case 'MODERATE':
            return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30';
        case 'DANGEROUS':
            return 'text-red-400 bg-red-500/10 border-red-500/30';
    }
};

const getRiskIcon = (risk: RiskLevel) => {
    switch (risk) {
        case 'SAFE':
            return '🟢';
        case 'MODERATE':
            return '🟡';
        case 'DANGEROUS':
            return '🔴';
    }
};

const getRiskWarning = (risk: RiskLevel) => {
    switch (risk) {
        case 'SAFE':
            return 'This operation is read-only and has no side effects.';
        case 'MODERATE':
            return 'This operation will modify data but changes are typically reversible.';
        case 'DANGEROUS':
            return '⚠️ WARNING: This operation may have irreversible consequences!';
    }
};

export const ToolAuthorizationModal: React.FC<ToolAuthorizationModalProps> = ({
    isOpen,
    toolName,
    serverName,
    arguments: args,
    riskLevel,
    onDeny,
    onAllowOnce,
    onAllowAlways,
}) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
            <div className="bg-slate-800 border border-slate-700 rounded-lg shadow-2xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-hidden flex flex-col">
                {/* Header */}
                <div className="px-6 py-4 border-b border-slate-700">
                    <h2 className="text-xl font-semibold text-slate-100 flex items-center gap-2">
                        🔐 Tool Authorization Required
                    </h2>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto custom-scrollbar px-6 py-4 space-y-4">
                    {/* Risk Badge */}
                    <div className={`flex items-center gap-2 px-4 py-3 rounded-lg border ${getRiskColor(riskLevel)}`}>
                        <span className="text-2xl">{getRiskIcon(riskLevel)}</span>
                        <div className="flex-1">
                            <div className="font-semibold text-sm uppercase tracking-wide">
                                {riskLevel} OPERATION
                            </div>
                            <div className="text-xs opacity-90 mt-1">
                                {getRiskWarning(riskLevel)}
                            </div>
                        </div>
                    </div>

                    {/* Tool Info */}
                    <div className="space-y-2">
                        <div className="flex items-baseline gap-2">
                            <span className="text-sm text-slate-400">Tool:</span>
                            <span className="text-base font-mono text-primary-300">{toolName}</span>
                        </div>
                        <div className="flex items-baseline gap-2">
                            <span className="text-sm text-slate-400">Server:</span>
                            <span className="text-sm font-mono text-slate-300">{serverName}</span>
                        </div>
                    </div>

                    {/* Arguments */}
                    <div>
                        <div className="text-sm text-slate-400 mb-2">Arguments:</div>
                        <div className="bg-slate-900 border border-slate-700 rounded p-3 font-mono text-xs text-slate-300 max-h-48 overflow-y-auto custom-scrollbar">
                            {Object.keys(args).length > 0 ? (
                                <pre className="whitespace-pre-wrap">
                                    {JSON.stringify(args, null, 2)}
                                </pre>
                            ) : (
                                <span className="text-slate-500 italic">No arguments</span>
                            )}
                        </div>
                    </div>
                </div>

                {/* Actions */}
                <div className="px-6 py-4 border-t border-slate-700 flex gap-3">
                    <button
                        onClick={onDeny}
                        className="flex-1 px-4 py-2.5 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg font-medium transition-colors"
                    >
                        ❌ Deny
                    </button>
                    <button
                        onClick={onAllowOnce}
                        className="flex-1 px-4 py-2.5 bg-primary-600 hover:bg-primary-500 text-white rounded-lg font-medium transition-colors"
                    >
                        ✅ Allow Once
                    </button>
                    <button
                        onClick={onAllowAlways}
                        className="flex-1 px-4 py-2.5 bg-green-600 hover:bg-green-500 text-white rounded-lg font-medium transition-colors"
                    >
                        🔓 Always Allow
                    </button>
                </div>
            </div>
        </div>
    );
};
