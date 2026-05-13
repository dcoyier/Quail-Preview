import type { ThinkingLevel } from "@mariozechner/pi-agent-core";
import type { Api, Model } from "@mariozechner/pi-ai";
export declare const OLLAMA_PROVIDER = "ollama";
export declare const DEFAULT_OLLAMA_MODEL = "qwen3-coder";
export declare const OLLAMA_API_KEY = "ollama";
export type OllamaModel = Model<Api> & {
    thinkingLevels?: ThinkingLevel[];
};
interface OllamaModelInfo {
    id: string;
    capabilities?: string[];
    contextWindow?: number;
    modelFamily?: string;
    modelFamilies?: string[];
    parser?: string;
    renderer?: string;
}
export declare function getOllamaBaseUrl(): string;
export declare function getOllamaProviderRequestConfig(): {
    apiKey: string;
};
export declare function getOllamaDefaults(): {
    api: Api;
    baseUrl: string;
};
export declare function createOllamaModel(model: string | OllamaModelInfo): OllamaModel;
export declare function loadOllamaModels(): Model<Api>[];
export {};
