// Type definitions for the COMPASS Web UI

export type ProviderType = 'ollama' | 'lm_studio' | 'openai' | 'anthropic';

export interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp?: number;
    reasoning?: {
        updates?: ProcessingUpdate[];
        result?: COMPASSResult;
    };
    thinking?: Array<{
        timestamp: string | null;
        content: string;
    }>;
}

export interface ProviderInfo {
    type: ProviderType;
    available: boolean;
    models: string[];
}

export interface COMPASSConfig {
    omcd: {
        R: number;
        max_resources: number;
    };
    slap: {
        alpha: number;
        beta: number;
    };
    self_discover: {
        max_trials: number;
    };
}

export interface FrameworkActivation {
    shape: boolean;
    slap: boolean;
    smart: boolean;
    omcd: boolean;
    self_discover: boolean;
    intelligence: boolean;
}

export interface ProcessingUpdate {
    stage: string;
    message: string;
    progress: number;
    data?: any;
}

export interface COMPASSResult {
    success: boolean;
    solution: string;
    score: number;
    iterations: number;
    resources_used: number;
    reflections: any[];
    trajectory: any[];
    config_used: COMPASSConfig;
    activated_frameworks: FrameworkActivation;
}

export interface ReasoningTrace {
    updates: ProcessingUpdate[];
    result?: COMPASSResult;
}

export interface MCPTool {
    name: string;
    description: string;
    server_name: string;
}

export interface MCPServer {
    name: string;
    connected: boolean;
}

export interface ChatRequest {
    messages: Message[];
    provider: ProviderType;
    model?: string;
    stream: boolean;
    temperature: number;
    use_compass: boolean;
    context?: Record<string, any>;
}
