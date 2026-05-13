import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
export const OLLAMA_PROVIDER = "ollama";
export const DEFAULT_OLLAMA_MODEL = "qwen3-coder";
export const OLLAMA_API_KEY = "ollama";
const DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434";
const OLLAMA_CONFIG_PATH_ENV = "QUAIL_OLLAMA_CONFIG_PATH";
const OLLAMA_REASONING_EFFORT_MAP = {
    minimal: "low",
    low: "low",
    medium: "medium",
    high: "high",
    xhigh: "high",
};
const OLLAMA_COMPAT = {
    supportsStore: false,
    supportsDeveloperRole: false,
    supportsReasoningEffort: true,
    reasoningEffortMap: OLLAMA_REASONING_EFFORT_MAP,
    thinkingFormat: "openrouter",
    supportsUsageInStreaming: true,
    maxTokensField: "max_tokens",
    supportsStrictMode: false,
    supportsLongCacheRetention: false,
};
function stripTrailingSlashes(value) {
    return value.replace(/\/+$/, "");
}
function ensureV1BaseUrl(value) {
    const stripped = stripTrailingSlashes(value);
    return stripped.endsWith("/v1") ? stripped : `${stripped}/v1`;
}
export function getOllamaBaseUrl() {
    const explicitBaseUrl = process.env.OLLAMA_BASE_URL?.trim();
    if (explicitBaseUrl)
        return ensureV1BaseUrl(explicitBaseUrl);
    const host = process.env.OLLAMA_HOST?.trim();
    if (host) {
        const withScheme = /^https?:\/\//i.test(host) ? host : `http://${host}`;
        return ensureV1BaseUrl(withScheme);
    }
    return `${DEFAULT_OLLAMA_HOST}/v1`;
}
export function getOllamaProviderRequestConfig() {
    return { apiKey: OLLAMA_API_KEY };
}
export function getOllamaDefaults() {
    return {
        api: "openai-completions",
        baseUrl: getOllamaBaseUrl(),
    };
}
function isRecord(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
function normalizeModelId(value) {
    if (typeof value !== "string")
        return undefined;
    const id = value.trim();
    if (!id || /\s/.test(id))
        return undefined;
    if (id.toLowerCase().includes("embedding"))
        return undefined;
    return id;
}
function getOllamaConfigPath() {
    return process.env[OLLAMA_CONFIG_PATH_ENV] || join(homedir(), ".ollama", "config.json");
}
function readOllamaLaunchModels() {
    const configPath = getOllamaConfigPath();
    if (!existsSync(configPath))
        return [];
    try {
        const parsed = JSON.parse(readFileSync(configPath, "utf-8"));
        if (!isRecord(parsed))
            return [];
        const models = [];
        const integrations = parsed.integrations;
        if (isRecord(integrations)) {
            for (const key of ["quail", "pi"]) {
                const integration = integrations[key];
                if (!isRecord(integration) || !Array.isArray(integration.models))
                    continue;
                for (const model of integration.models) {
                    const id = normalizeModelId(model);
                    if (id)
                        models.push({ id });
                }
            }
        }
        const lastSelection = normalizeModelId(parsed.last_selection);
        const lastModel = normalizeModelId(parsed.last_model);
        if ((lastSelection === "quail" || lastSelection === "pi") && lastModel) {
            models.push({ id: lastModel });
        }
        return models;
    }
    catch {
        return [];
    }
}
function getOllamaModelsRoot() {
    return process.env.OLLAMA_MODELS || join(homedir(), ".ollama", "models");
}
function getOllamaModelsDir() {
    return join(getOllamaModelsRoot(), "manifests");
}
function walkFiles(dir) {
    if (!existsSync(dir))
        return [];
    const files = [];
    try {
        for (const entry of readdirSync(dir)) {
            if (entry.startsWith("."))
                continue;
            const path = join(dir, entry);
            const stats = statSync(path);
            if (stats.isDirectory()) {
                files.push(...walkFiles(path));
            }
            else if (stats.isFile()) {
                files.push(path);
            }
        }
    }
    catch {
        return files;
    }
    return files;
}
function modelIdFromManifestPath(manifestsDir, manifestPath) {
    const relativePath = manifestPath.slice(manifestsDir.length + 1);
    const parts = relativePath.split(/[\\/]/).filter(Boolean);
    if (parts.length < 4)
        return undefined;
    const tag = parts.at(-1);
    const modelName = parts.at(-2);
    const namespace = parts.at(-3);
    if (!tag || !modelName || !namespace)
        return undefined;
    const prefix = namespace === "library" ? modelName : `${namespace}/${modelName}`;
    return normalizeModelId(`${prefix}:${tag}`);
}
function blobPathFromDigest(digest) {
    const match = /^sha256:(.+)$/.exec(digest);
    if (!match)
        return undefined;
    return join(getOllamaModelsRoot(), "blobs", `sha256-${match[1]}`);
}
function readModelInfoFromManifest(manifestsDir, manifestPath) {
    const id = modelIdFromManifestPath(manifestsDir, manifestPath);
    if (!id)
        return undefined;
    try {
        const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
        if (!isRecord(manifest) || !isRecord(manifest.config) || typeof manifest.config.digest !== "string") {
            return { id };
        }
        const blobPath = blobPathFromDigest(manifest.config.digest);
        if (!blobPath || !existsSync(blobPath))
            return { id };
        const config = JSON.parse(readFileSync(blobPath, "utf-8"));
        if (!isRecord(config))
            return { id };
        return {
            id,
            capabilities: Array.isArray(config.capabilities)
                ? config.capabilities.filter((capability) => typeof capability === "string")
                : undefined,
            contextWindow: typeof config.context_length === "number" && config.context_length > 0 ? config.context_length : undefined,
            modelFamily: typeof config.model_family === "string" ? config.model_family : undefined,
            modelFamilies: Array.isArray(config.model_families)
                ? config.model_families.filter((family) => typeof family === "string")
                : undefined,
            parser: typeof config.parser === "string" ? config.parser : undefined,
            renderer: typeof config.renderer === "string" ? config.renderer : undefined,
        };
    }
    catch {
        return { id };
    }
}
function readInstalledOllamaModels() {
    const manifestsDir = getOllamaModelsDir();
    return walkFiles(manifestsDir)
        .map((path) => readModelInfoFromManifest(manifestsDir, path))
        .filter((info) => info !== undefined);
}
function getOllamaModelInfos() {
    const byId = new Map();
    for (const info of [...readOllamaLaunchModels(), ...readInstalledOllamaModels()]) {
        const current = byId.get(info.id);
        byId.set(info.id, { ...current, ...info });
    }
    if (byId.size === 0) {
        byId.set(DEFAULT_OLLAMA_MODEL, { id: DEFAULT_OLLAMA_MODEL });
    }
    return [...byId.values()];
}
function findOllamaModelInfo(id) {
    return getOllamaModelInfos().find((info) => info.id === id);
}
function getModelDescriptor(info) {
    return [
        info.id,
        info.modelFamily,
        ...(info.modelFamilies ?? []),
        info.parser,
        info.renderer,
        info.capabilities?.join(" "),
    ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
}
function isVisionModel(info) {
    if (info.capabilities?.includes("vision"))
        return true;
    const lower = getModelDescriptor(info);
    return lower.includes("vl") || lower.includes("vision") || lower.includes("llava") || lower.includes("moondream");
}
function isGptOssModel(info) {
    return getModelDescriptor(info).includes("gpt-oss") || getModelDescriptor(info).includes("gptoss");
}
function getOllamaThinkingLevels(info) {
    if (info.capabilities) {
        if (!info.capabilities.includes("thinking"))
            return undefined;
        return isGptOssModel(info) ? ["low", "medium", "high"] : ["off", "low", "medium", "high"];
    }
    const descriptor = getModelDescriptor(info);
    const likelyThinkingModel = (!descriptor.includes("qwen3-coder") && /\bqwen3(\b|[.:])/.test(descriptor)) ||
        descriptor.includes("qwen35") ||
        descriptor.includes("qwen3.5") ||
        descriptor.includes("qwen36") ||
        descriptor.includes("qwen3.6") ||
        descriptor.includes("deepseek-r1") ||
        descriptor.includes("deepseek-v3.1") ||
        descriptor.includes("deepseek-v4") ||
        descriptor.includes("glm-5") ||
        descriptor.includes("thinking");
    if (!likelyThinkingModel)
        return undefined;
    return isGptOssModel(info) ? ["low", "medium", "high"] : ["off", "low", "medium", "high"];
}
export function createOllamaModel(model) {
    const info = typeof model === "string" ? (findOllamaModelInfo(model) ?? { id: model }) : model;
    const defaults = getOllamaDefaults();
    const thinkingLevels = getOllamaThinkingLevels(info);
    return {
        id: info.id,
        name: info.id,
        api: defaults.api,
        provider: OLLAMA_PROVIDER,
        baseUrl: defaults.baseUrl,
        reasoning: !!thinkingLevels,
        input: isVisionModel(info) ? ["text", "image"] : ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: info.contextWindow ?? 128000,
        maxTokens: 16384,
        headers: undefined,
        compat: OLLAMA_COMPAT,
        thinkingLevels,
    };
}
export function loadOllamaModels() {
    return getOllamaModelInfos().map(createOllamaModel);
}
//# sourceMappingURL=ollama-provider.js.map