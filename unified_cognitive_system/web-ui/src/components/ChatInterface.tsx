import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Brain, Sparkles, ChevronDown, ChevronUp, Check } from 'lucide-react';
import { useAppStore } from '../store';
import { api } from '../services/api';
import type { Message, ProcessingUpdate, COMPASSResult } from '../types';
import ReactMarkdown from 'react-markdown';

interface ThinkingStep {
    timestamp: string | null;
    content: string;
}

export const ChatInterface: React.FC = () => {
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [processingUpdate, setProcessingUpdate] = useState<ProcessingUpdate | null>(null);
    const [conversationId, setConversationId] = useState<string | null>(null);
    const [currentThinking, setCurrentThinking] = useState<ThinkingStep[]>([]);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const {
        messages,
        addMessage,
        currentProvider,
        currentModel,
        useCompass,
        allowAllTools,
        addCompassStep,
        clearCompassTrace
    } = useAppStore();

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, processingUpdate, currentThinking]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage: Message = {
            role: 'user',
            content: input.trim(),
        };

        addMessage(userMessage);
        setInput('');
        setIsLoading(true);
        setProcessingUpdate(null);
        setCurrentThinking([]);
        clearCompassTrace(); // Clear previous trace

        try {
            const request = {
                messages: [...messages, userMessage],
                provider: currentProvider,
                model: currentModel || undefined,
                stream: true,
                temperature: 0.7,
                use_compass: useCompass,
                allow_all_tools: allowAllTools,
                conversation_id: conversationId || undefined,
            };

            const responseBody = await api.chatCompletion(request);

            if (responseBody) {
                const reader = responseBody.getReader();
                const decoder = new TextDecoder();
                let assistantContent = '';
                let rawContent = '';
                let reasoning: any = {};
                let thinking: ThinkingStep[] = [];
                let inThinkingBlock = false;
                let thinkingBuffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') break;

                            try {
                                const parsed = JSON.parse(data);

                                if (parsed.type === 'content') {
                                    const contentChunk = parsed.content;
                                    rawContent += contentChunk;

                                    // Robust <thinking> tag parsing
                                    // Check for opening tag
                                    if (contentChunk.includes('<thinking>')) {
                                        inThinkingBlock = true;
                                        const parts = contentChunk.split('<thinking>');
                                        assistantContent += parts[0];
                                        thinkingBuffer += parts[1] || '';
                                    }
                                    // Check for closing tag
                                    else if (contentChunk.includes('</thinking>')) {
                                        inThinkingBlock = false;
                                        const parts = contentChunk.split('</thinking>');
                                        thinkingBuffer += parts[0];
                                        assistantContent += parts[1] || '';

                                        // Add parsed thinking to steps
                                        if (thinkingBuffer.trim()) {
                                            // Split by newlines to create distinct steps if possible
                                            const bufferSteps = thinkingBuffer.split('\n').filter(s => s.trim());
                                            const newSteps = bufferSteps.map(s => ({
                                                timestamp: new Date().toISOString(),
                                                content: s.trim()
                                            }));

                                            thinking.push(...newSteps);
                                            setCurrentThinking(prev => [...prev, ...newSteps]);
                                            thinkingBuffer = '';
                                        }
                                    }
                                    // Inside thinking block
                                    else if (inThinkingBlock) {
                                        thinkingBuffer += contentChunk;
                                        // Optional: Update live preview of thinking if we want to show it streaming
                                        // For now, we batch it by lines or just wait for close tag to avoid UI jitter
                                    }
                                    // Normal content
                                    else {
                                        assistantContent += contentChunk;
                                    }

                                } else if (parsed.type === 'update') {
                                    setProcessingUpdate(parsed.data as ProcessingUpdate);
                                    if (!reasoning.updates) reasoning.updates = [];
                                    reasoning.updates.push(parsed.data);
                                } else if (parsed.type === 'result') {
                                    reasoning.result = parsed.data as COMPASSResult;
                                } else if (parsed.type === 'thinking') {
                                    // Handle COMPASS structured thinking events -> Send to Sidebar Trace
                                    const newSteps = parsed.steps;
                                    newSteps.forEach((step: ThinkingStep) => {
                                        addCompassStep(step.content);
                                    });
                                    // Do NOT add to main thinking panel unless we want logs there too
                                    // User requested separation, so we skip adding to 'thinking' array here
                                } else if (parsed.type === 'conversation_id') {
                                    setConversationId(parsed.conversation_id);
                                }
                            } catch (e) {
                                console.error('Parse error:', e);
                            }
                        }
                    }
                }

                // Pre-process markdown to ensure headers have newlines
                // This fixes the "text### Header" issue
                const processedContent = assistantContent.replace(/([^\n])(#{1,6}\s)/g, '$1\n\n$2');

                addMessage({
                    role: 'assistant',
                    content: processedContent,
                    reasoning: Object.keys(reasoning).length > 0 ? reasoning : undefined,
                    thinking, // Only contains actual model thinking now
                });
            }
        } catch (error) {
            console.error('Chat error:', error);
            addMessage({
                role: 'assistant',
                content: 'Sorry, an error occurred while processing your request.',
            });
        } finally {
            setIsLoading(false);
            setProcessingUpdate(null);
            setCurrentThinking([]);
        }
    };

    return (
        <div className="flex flex-col h-full bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100 font-sans">
            {/* Messages Container */}
            <div className="flex-1 overflow-y-auto custom-scrollbar px-4 py-6 space-y-8">
                {messages.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-full text-center max-w-2xl mx-auto">
                        <div className="relative mb-8">
                            <div className="absolute inset-0 bg-primary-500/20 blur-3xl rounded-full"></div>
                            <Brain className="w-20 h-20 text-primary-400 relative animate-pulse" />
                        </div>
                        <h2 className="text-3xl font-bold gradient-text mb-3 tracking-tight">
                            COMPASS Cognitive System
                        </h2>
                        <p className="text-slate-400 text-lg mb-8 max-w-md leading-relaxed">
                            Advanced AI reasoning with metacognitive control and transparent thinking process
                        </p>
                        <div className="grid grid-cols-2 gap-3 w-full max-w-lg text-sm">
                            {['Complex problem solving', 'Strategic planning', 'Multi-domain analysis', 'Adaptive reasoning'].map((feature, idx) => (
                                <div key={idx} className="flex items-center gap-2 px-4 py-3 bg-slate-800/40 rounded-xl border border-slate-700/40 backdrop-blur-sm">
                                    <Check className="w-4 h-4 text-primary-400" />
                                    <span className="text-slate-300">{feature}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <MessageBubble key={idx} message={msg} />
                ))}

                {/* Processing Indicator */}
                {isLoading && (
                    <div className="flex justify-start max-w-4xl mx-auto w-full px-4">
                        <div className="flex-1 max-w-3xl">
                            {/* Thinking Panel for Current Message */}
                            {currentThinking.length > 0 && (
                                <ThinkingPanel steps={currentThinking} isStreaming={true} />
                            )}

                            {/* Processing Status */}
                            {processingUpdate && (
                                <div className="mt-4 bg-slate-900/80 backdrop-blur-md border border-slate-700/50 rounded-xl p-4 shadow-lg max-w-md">
                                    <div className="flex items-center gap-3">
                                        <Loader2 className="w-4 h-4 text-primary-400 animate-spin" />
                                        <div className="flex-1 min-w-0">
                                            <div className="text-sm font-medium text-slate-200 truncate">
                                                {processingUpdate.stage}
                                            </div>
                                            <div className="text-xs text-slate-400 truncate">{processingUpdate.message}</div>
                                        </div>
                                        <div className="text-xs font-mono text-primary-400">
                                            {Math.round(processingUpdate.progress * 100)}%
                                        </div>
                                    </div>
                                    {/* Progress Bar */}
                                    <div className="mt-3 h-1 bg-slate-800 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-gradient-to-r from-primary-500 to-primary-400 transition-all duration-300 ease-out"
                                            style={{ width: `${processingUpdate.progress * 100}%` }}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="border-t border-slate-800/50 bg-slate-950/80 backdrop-blur-xl p-4">
                <div className="max-w-4xl mx-auto">
                    <div className="relative flex items-end gap-3">
                        <div className="flex-1 relative">
                            <textarea
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && !e.shiftKey) {
                                        e.preventDefault();
                                        handleSend();
                                    }
                                }}
                                placeholder="Ask anything..."
                                className="w-full px-5 py-4 bg-slate-900/50 border border-slate-700/50 rounded-2xl
                                         text-slate-100 placeholder-slate-500 resize-none
                                         focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-500/50
                                         transition-all duration-200 max-h-40 leading-relaxed"
                                rows={1}
                                style={{
                                    minHeight: '60px',
                                    maxHeight: '160px',
                                    overflowY: input.split('\\n').length > 3 ? 'auto' : 'hidden'
                                }}
                            />
                        </div>
                        <button
                            onClick={handleSend}
                            disabled={!input.trim() || isLoading}
                            className="flex items-center justify-center w-14 h-[60px] bg-gradient-to-br from-primary-500 to-primary-600
                                     hover:from-primary-600 hover:to-primary-700 disabled:from-slate-800 disabled:to-slate-800
                                     text-white rounded-2xl transition-all duration-200 shadow-lg hover:shadow-primary-500/25
                                     disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none group"
                        >
                            {isLoading ? (
                                <Loader2 className="w-6 h-6 animate-spin text-slate-400" />
                            ) : (
                                <Send className="w-6 h-6 group-hover:scale-110 transition-transform" />
                            )}
                        </button>
                    </div>

                    {/* Helper Text */}
                    <div className="flex items-center justify-between mt-3 px-2 text-xs text-slate-500 font-medium">
                        <span>Press Enter to send, Shift+Enter for new line</span>
                        {useCompass && (
                            <span className="flex items-center gap-1.5 text-primary-400/80">
                                <Sparkles className="w-3.5 h-3.5" />
                                COMPASS enabled
                            </span>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

// Message Bubble Component
const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
    const isUser = message.role === 'user';

    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} max-w-4xl mx-auto w-full px-4`}>
            <div className={`flex-1 ${isUser ? 'max-w-2xl' : 'max-w-3xl'}`}>
                {/* User Message */}
                {isUser ? (
                    <div className="bg-gradient-to-br from-primary-600 to-primary-700 text-white
                                  rounded-2xl rounded-tr-sm px-6 py-4 shadow-lg">
                        <div className="leading-relaxed whitespace-pre-wrap text-[15px]">
                            {message.content}
                        </div>
                    </div>
                ) : (
                    /* Assistant Message */
                    <div className="space-y-4">
                        {/* Thinking Panel */}
                        {message.thinking && message.thinking.length > 0 && (
                            <ThinkingPanel steps={message.thinking} />
                        )}

                        {/* Response Content */}
                        <div className="bg-slate-900/40 border border-slate-800/50 rounded-2xl rounded-tl-sm px-8 py-6 shadow-sm">
                            <MarkdownContent content={message.content} />

                            {/* Reasoning Metrics */}
                            {message.reasoning?.result && (
                                <div className="mt-6 pt-4 border-t border-slate-800/50 flex flex-wrap items-center gap-6 text-xs font-medium text-slate-500">
                                    <div className="flex items-center gap-2 text-primary-400">
                                        <Sparkles className="w-3.5 h-3.5" />
                                        <span>Score: {message.reasoning.result.score.toFixed(2)}</span>
                                    </div>
                                    <div>Iterations: {message.reasoning.result.iterations}</div>
                                    <div>Resources: {message.reasoning.result.resources_used.toFixed(1)}</div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

// Thinking Panel Component
const ThinkingPanel: React.FC<{ steps: ThinkingStep[], isStreaming?: boolean }> = ({ steps, isStreaming = false }) => {
    const [isExpanded, setIsExpanded] = useState(isStreaming);

    return (
        <div className="bg-slate-900/60 border border-indigo-500/20 rounded-xl overflow-hidden mb-4">
            {/* Header */}
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-indigo-500/5
                         transition-colors group cursor-pointer"
            >
                <div className="flex items-center gap-3">
                    <div className="relative flex items-center justify-center">
                        <Brain className={`w-4 h-4 text-indigo-400 ${isStreaming ? 'animate-pulse' : ''}`} />
                        {isStreaming && (
                            <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-indigo-400 rounded-full animate-ping"></span>
                        )}
                    </div>
                    <div className="text-left flex items-center gap-2">
                        <span className="font-medium text-sm text-indigo-300">
                            Thinking Process
                        </span>
                        <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400/80 border border-indigo-500/10">
                            {steps.length} steps
                        </span>
                    </div>
                </div>
                {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-indigo-400/70 group-hover:text-indigo-300" />
                ) : (
                    <ChevronDown className="w-4 h-4 text-indigo-400/70 group-hover:text-indigo-300" />
                )}
            </button>

            {/* Content */}
            {isExpanded && (
                <div className="px-4 py-3 border-t border-indigo-500/10 bg-slate-950/30 max-h-96 overflow-y-auto custom-scrollbar">
                    <div className="space-y-3">
                        {steps.map((step, idx) => {
                            if (step.content.startsWith('SLAP_DETAILED_PLAN:')) {
                                try {
                                    const plan = JSON.parse(step.content.replace('SLAP_DETAILED_PLAN:', ''));
                                    return <SLAPAnalysisView key={idx} plan={plan} />;
                                } catch (e) {
                                    return (
                                        <div key={idx} className="text-red-400 text-xs">
                                            Failed to parse SLAP plan
                                        </div>
                                    );
                                }
                            }
                            return (
                                <div
                                    key={idx}
                                    className="flex gap-3 text-sm animate-fadeIn group"
                                    style={{ animationDelay: `${idx * 30}ms` }}
                                >
                                    <div className="flex-shrink-0 mt-1.5 opacity-50 group-hover:opacity-100 transition-opacity">
                                        <div className="w-1 h-1 rounded-full bg-indigo-400"></div>
                                    </div>
                                    <div className="flex-1 text-slate-300/90 font-mono text-xs leading-relaxed break-words">
                                        {step.content}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
};

const SLAPAnalysisView: React.FC<{ plan: any }> = ({ plan }) => {
    const [expanded, setExpanded] = useState(false);

    return (
        <div className="mt-2 mb-4 border border-indigo-500/30 rounded-lg bg-indigo-900/10 overflow-hidden">
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full px-3 py-2 flex items-center justify-between bg-indigo-900/20 hover:bg-indigo-900/30 transition-colors"
            >
                <div className="flex items-center gap-2">
                    <Sparkles className="w-3 h-3 text-indigo-300" />
                    <span className="text-xs font-semibold text-indigo-200">SLAP Reasoning Analysis</span>
                    <span className="text-[10px] bg-indigo-500/20 text-indigo-300 px-1.5 py-0.5 rounded">
                        Score: {plan.advancement_score?.toFixed(2)}
                    </span>
                </div>
                {expanded ? <ChevronUp className="w-3 h-3 text-indigo-400" /> : <ChevronDown className="w-3 h-3 text-indigo-400" />}
            </button>

            {expanded && (
                <div className="p-3 space-y-3 text-xs font-mono text-slate-300">
                    {/* Conceptualization */}
                    <div className="space-y-1">
                        <div className="text-indigo-400 font-semibold">1. Conceptualization</div>
                        <div className="pl-2 border-l border-indigo-500/20">
                            <div>Primary: <span className="text-slate-200">{plan.conceptualization?.primary_concept}</span></div>
                            <div>Domain: <span className="text-slate-400">{plan.conceptualization?.domain}</span></div>
                        </div>
                    </div>

                    {/* Scrutiny */}
                    <div className="space-y-1">
                        <div className="text-indigo-400 font-semibold">2. Scrutiny (Score: {plan.scrutiny?.score})</div>
                        <div className="pl-2 border-l border-indigo-500/20">
                            {plan.scrutiny?.weaknesses?.length > 0 && (
                                <div className="text-amber-400/80">Weaknesses: {plan.scrutiny.weaknesses.join(', ')}</div>
                            )}
                            {plan.scrutiny?.gaps?.length > 0 && (
                                <div className="text-amber-400/80">Gaps: {plan.scrutiny.gaps.join(', ')}</div>
                            )}
                        </div>
                    </div>

                    {/* Model */}
                    <div className="space-y-1">
                        <div className="text-indigo-400 font-semibold">3. Model (Completeness: {plan.model?.completeness})</div>
                        <div className="pl-2 border-l border-indigo-500/20">
                            <div>Structure: {plan.model?.structure}</div>
                            <div>Components: {plan.model?.components?.join(', ')}</div>
                        </div>
                    </div>

                    {/* Full JSON Toggle */}
                    <details className="pt-2">
                        <summary className="cursor-pointer text-indigo-400/60 hover:text-indigo-400 transition-colors">View Full JSON</summary>
                        <pre className="mt-2 p-2 bg-slate-950 rounded overflow-x-auto text-[10px] text-slate-400">
                            {JSON.stringify(plan, null, 2)}
                        </pre>
                    </details>
                </div>
            )}
        </div>
    );
};

// Markdown Content Component
const MarkdownContent: React.FC<{ content: string }> = ({ content }) => (
    <div className="markdown-content text-slate-200">
        <ReactMarkdown
            components={{
                p: ({ children }) => <p className="mb-5 last:mb-0 text-[15px] leading-7 text-slate-300">{children}</p>,
                ul: ({ children }) => <ul className="list-disc list-outside mb-5 space-y-2 ml-5 text-slate-300">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal list-outside mb-5 space-y-2 ml-5 text-slate-300">{children}</ol>,
                li: ({ children }) => <li className="text-[15px] leading-7 pl-1 marker:text-slate-500">{children}</li>,
                h1: ({ children }) => <h1 className="text-2xl font-bold mt-8 mb-4 first:mt-0 text-slate-100 tracking-tight">{children}</h1>,
                h2: ({ children }) => <h2 className="text-xl font-semibold mt-6 mb-3 first:mt-0 text-slate-100 tracking-tight">{children}</h2>,
                h3: ({ children }) => <h3 className="text-lg font-medium mt-5 mb-2 first:mt-0 text-slate-200">{children}</h3>,
                code: ({ inline, children }: any) =>
                    inline ? (
                        <code className="bg-slate-800 px-1.5 py-0.5 rounded text-primary-300 text-[13px] font-mono border border-slate-700/50">
                            {children}
                        </code>
                    ) : (
                        <code className="block bg-slate-950 p-4 rounded-xl my-4 text-[13px] font-mono overflow-x-auto
                                       border border-slate-800 text-slate-300 leading-6 shadow-inner">
                            {children}
                        </code>
                    ),
                strong: ({ children }) => <strong className="font-semibold text-slate-100">{children}</strong>,
                em: ({ children }) => <em className="italic text-slate-400">{children}</em>,
                hr: () => <hr className="my-8 border-slate-800" />,
                blockquote: ({ children }) => (
                    <blockquote className="border-l-2 border-primary-500/50 pl-4 py-1 my-6 italic text-slate-400">
                        {children}
                    </blockquote>
                ),
                a: ({ children, href }) => (
                    <a href={href} className="text-primary-400 hover:text-primary-300 underline underline-offset-4 decoration-primary-400/30 hover:decoration-primary-400" target="_blank" rel="noopener noreferrer">
                        {children}
                    </a>
                ),
            }}
        >
            {content}
        </ReactMarkdown>
    </div>
);
