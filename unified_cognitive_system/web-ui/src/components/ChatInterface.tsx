import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Brain, Zap } from 'lucide-react';
import { useAppStore } from '../store';
import { api } from '../services/api';
import type { Message, ProcessingUpdate, COMPASSResult } from '../types';
import ReactMarkdown from 'react-markdown';

export const ChatInterface: React.FC = () => {
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [processingUpdate, setProcessingUpdate] = useState<ProcessingUpdate | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const {
        messages,
        addMessage,
        currentProvider,
        currentModel,
        useCompass
    } = useAppStore();

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, processingUpdate]);

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

        try {
            const request = {
                messages: [...messages, userMessage],
                provider: currentProvider,
                model: currentModel || undefined,
                stream: true,
                temperature: 0.7,
                use_compass: useCompass,
            };

            const responseBody = await api.chatCompletion(request);

            if (responseBody) {
                const reader = responseBody.getReader();
                const decoder = new TextDecoder();
                let assistantContent = '';
                let reasoning: any = {};

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
                                    assistantContent += parsed.content;
                                } else if (parsed.type === 'update') {
                                    setProcessingUpdate(parsed.data as ProcessingUpdate);
                                    if (!reasoning.updates) reasoning.updates = [];
                                    reasoning.updates.push(parsed.data);
                                } else if (parsed.type === 'result') {
                                    reasoning.result = parsed.data as COMPASSResult;
                                    // Don't overwrite content, wait for LLM stream
                                }
                            } catch (e) {
                                console.error('Parse error:', e);
                            }
                        }
                    }
                }

                addMessage({
                    role: 'assistant',
                    content: assistantContent,
                    reasoning: Object.keys(reasoning).length > 0 ? reasoning : undefined,
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
        }
    };

    return (
        <div className="flex flex-col h-full">
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-4">
                {messages.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-full text-center">
                        <Brain className="w-16 h-16 text-primary-500 mb-4" />
                        <h2 className="text-2xl font-semibold gradient-text mb-2">
                            COMPASS Cognitive System
                        </h2>
                        <p className="text-slate-400 max-w-md">
                            Advanced AI reasoning with metacognitive control. Ask me anything!
                        </p>
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`message-bubble ${msg.role === 'user' ? 'message-user' : 'message-assistant'}`}>
                            <div className="prose prose-invert max-w-none">
                                <ReactMarkdown>{msg.content}</ReactMarkdown>
                            </div>

                            {msg.reasoning && (
                                <div className="mt-3 pt-3 border-t border-white/10">
                                    <div className="flex items-center gap-2 text-xs text-slate-400 mb-2">
                                        <Zap className="w-3 h-3" />
                                        <span>COMPASS Reasoning</span>
                                    </div>
                                    {msg.reasoning.result && (
                                        <div className="text-xs space-y-1">
                                            <div>Score: {msg.reasoning.result.score.toFixed(2)}</div>
                                            <div>Iterations: {msg.reasoning.result.iterations}</div>
                                            <div>Resources: {msg.reasoning.result.resources_used.toFixed(1)}</div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                ))}

                {processingUpdate && (
                    <div className="flex justify-start">
                        <div className="message-bubble message-assistant">
                            <div className="flex items-center gap-3">
                                <Loader2 className="w-4 h-4 animate-spin text-primary-400" />
                                <div>
                                    <div className="font-medium text-sm">{processingUpdate.stage}</div>
                                    <div className="text-xs text-slate-400">{processingUpdate.message}</div>
                                    <div className="mt-2 w-full bg-slate-700 rounded-full h-1.5">
                                        <div
                                            className="bg-primary-500 h-1.5 rounded-full transition-all duration-300"
                                            style={{ width: `${processingUpdate.progress * 100}%` }}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-white/10 bg-slate-900/50 backdrop-blur-sm">
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                        placeholder="Ask anything..."
                        className="input-field flex-1"
                        disabled={isLoading}
                    />
                    <button
                        onClick={handleSend}
                        disabled={isLoading || !input.trim()}
                        className="btn-primary px-5"
                    >
                        {isLoading ? (
                            <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                            <Send className="w-5 h-5" />
                        )}
                    </button>
                </div>

                {useCompass && (
                    <div className="mt-2 flex items-center gap-2 text-xs text-primary-400">
                        <Brain className="w-3 h-3" />
                        <span>COMPASS auto-configuration enabled</span>
                    </div>
                )}
            </div>
        </div>
    );
};
