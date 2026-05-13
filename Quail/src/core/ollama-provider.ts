import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ThinkingLevel } from "@mariozechner/pi-agent-core";
import type { Api, Model, OpenAICompletionsCompat } from "@mariozechner/pi-ai";

export const OLLAMA_PROVIDER = "ollama";
export const DEFAULT_OLLAMA_MODEL = "qwen3-coder";
export const OLLAMA_API_KEY = "ollama";

const DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434";
const OLLAMA_CONFIG_PATH_ENV = "QUAIL_OLLAMA_CONFIG_PATH";

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

const OLLAMA_REASONING_EFFORT_MAP: OpenAICompletionsCompat["reasoningEffortMap"] = {
	minimal: "low",
	low: "low",
	medium: "medium",
	high: "high",
	xhigh: "high",
};

const OLLAMA_COMPAT: OpenAICompletionsCompat = {
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

function stripTrailingSlashes(value: string): string {
	return value.replace(/\/+$/, "");
}

function ensureV1BaseUrl(value: string): string {
	const stripped = stripTrailingSlashes(value);
	return stripped.endsWith("/v1") ? stripped : `${stripped}/v1`;
}

export function getOllamaBaseUrl(): string {
	const explicitBaseUrl = process.env.OLLAMA_BASE_URL?.trim();
	if (explicitBaseUrl) return ensureV1BaseUrl(explicitBaseUrl);

	const host = process.env.OLLAMA_HOST?.trim();
	if (host) {
		const withScheme = /^https?:\/\//i.test(host) ? host : `http://${host}`;
		return ensureV1BaseUrl(withScheme);
	}

	return `${DEFAULT_OLLAMA_HOST}/v1`;
}

export function getOllamaProviderRequestConfig(): { apiKey: string } {
	return { apiKey: OLLAMA_API_KEY };
}

export function getOllamaDefaults(): { api: Api; baseUrl: string } {
	return {
		api: "openai-completions",
		baseUrl: getOllamaBaseUrl(),
	};
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeModelId(value: unknown): string | undefined {
	if (typeof value !== "string") return undefined;
	const id = value.trim();
	if (!id || /\s/.test(id)) return undefined;
	if (id.toLowerCase().includes("embedding")) return undefined;
	return id;
}

function getOllamaConfigPath(): string {
	return process.env[OLLAMA_CONFIG_PATH_ENV] || join(homedir(), ".ollama", "config.json");
}

function readOllamaLaunchModels(): OllamaModelInfo[] {
	const configPath = getOllamaConfigPath();
	if (!existsSync(configPath)) return [];

	try {
		const parsed = JSON.parse(readFileSync(configPath, "utf-8")) as unknown;
		if (!isRecord(parsed)) return [];

		const models: OllamaModelInfo[] = [];
		const integrations = parsed.integrations;
		if (isRecord(integrations)) {
			for (const key of ["quail", "pi"]) {
				const integration = integrations[key];
				if (!isRecord(integration) || !Array.isArray(integration.models)) continue;
				for (const model of integration.models) {
					const id = normalizeModelId(model);
					if (id) models.push({ id });
				}
			}
		}

		const lastSelection = normalizeModelId(parsed.last_selection);
		const lastModel = normalizeModelId(parsed.last_model);
		if ((lastSelection === "quail" || lastSelection === "pi") && lastModel) {
			models.push({ id: lastModel });
		}

		return models;
	} catch {
		return [];
	}
}

function getOllamaModelsRoot(): string {
	return process.env.OLLAMA_MODELS || join(homedir(), ".ollama", "models");
}

function getOllamaModelsDir(): string {
	return join(getOllamaModelsRoot(), "manifests");
}

function walkFiles(dir: string): string[] {
	if (!existsSync(dir)) return [];

	const files: string[] = [];
	try {
		for (const entry of readdirSync(dir)) {
			if (entry.startsWith(".")) continue;
			const path = join(dir, entry);
			const stats = statSync(path);
			if (stats.isDirectory()) {
				files.push(...walkFiles(path));
			} else if (stats.isFile()) {
				files.push(path);
			}
		}
	} catch {
		return files;
	}
	return files;
}

function modelIdFromManifestPath(manifestsDir: string, manifestPath: string): string | undefined {
	const relativePath = manifestPath.slice(manifestsDir.length + 1);
	const parts = relativePath.split(/[\\/]/).filter(Boolean);
	if (parts.length < 4) return undefined;

	const tag = parts.at(-1);
	const modelName = parts.at(-2);
	const namespace = parts.at(-3);
	if (!tag || !modelName || !namespace) return undefined;

	const prefix = namespace === "library" ? modelName : `${namespace}/${modelName}`;
	return normalizeModelId(`${prefix}:${tag}`);
}

function blobPathFromDigest(digest: string): string | undefined {
	const match = /^sha256:(.+)$/.exec(digest);
	if (!match) return undefined;
	return join(getOllamaModelsRoot(), "blobs", `sha256-${match[1]}`);
}

function readModelInfoFromManifest(manifestsDir: string, manifestPath: string): OllamaModelInfo | undefined {
	const id = modelIdFromManifestPath(manifestsDir, manifestPath);
	if (!id) return undefined;

	try {
		const manifest = JSON.parse(readFileSync(manifestPath, "utf-8")) as unknown;
		if (!isRecord(manifest) || !isRecord(manifest.config) || typeof manifest.config.digest !== "string") {
			return { id };
		}

		const blobPath = blobPathFromDigest(manifest.config.digest);
		if (!blobPath || !existsSync(blobPath)) return { id };

		const config = JSON.parse(readFileSync(blobPath, "utf-8")) as unknown;
		if (!isRecord(config)) return { id };

		return {
			id,
			capabilities: Array.isArray(config.capabilities)
				? config.capabilities.filter((capability): capability is string => typeof capability === "string")
				: undefined,
			contextWindow: typeof config.context_length === "number" && config.context_length > 0 ? config.context_length : undefined,
			modelFamily: typeof config.model_family === "string" ? config.model_family : undefined,
			modelFamilies: Array.isArray(config.model_families)
				? config.model_families.filter((family): family is string => typeof family === "string")
				: undefined,
			parser: typeof config.parser === "string" ? config.parser : undefined,
			renderer: typeof config.renderer === "string" ? config.renderer : undefined,
		};
	} catch {
		return { id };
	}
}

function readInstalledOllamaModels(): OllamaModelInfo[] {
	const manifestsDir = getOllamaModelsDir();
	return walkFiles(manifestsDir)
		.map((path) => readModelInfoFromManifest(manifestsDir, path))
		.filter((info): info is OllamaModelInfo => info !== undefined);
}

function getOllamaModelInfos(): OllamaModelInfo[] {
	const byId = new Map<string, OllamaModelInfo>();
	for (const info of [...readOllamaLaunchModels(), ...readInstalledOllamaModels()]) {
		const current = byId.get(info.id);
		byId.set(info.id, { ...current, ...info });
	}

	if (byId.size === 0) {
		byId.set(DEFAULT_OLLAMA_MODEL, { id: DEFAULT_OLLAMA_MODEL });
	}

	return [...byId.values()];
}

function findOllamaModelInfo(id: string): OllamaModelInfo | undefined {
	return getOllamaModelInfos().find((info) => info.id === id);
}

function getModelDescriptor(info: OllamaModelInfo): string {
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

function isVisionModel(info: OllamaModelInfo): boolean {
	if (info.capabilities?.includes("vision")) return true;
	const lower = getModelDescriptor(info);
	return lower.includes("vl") || lower.includes("vision") || lower.includes("llava") || lower.includes("moondream");
}

function isGptOssModel(info: OllamaModelInfo): boolean {
	return getModelDescriptor(info).includes("gpt-oss") || getModelDescriptor(info).includes("gptoss");
}

function getOllamaThinkingLevels(info: OllamaModelInfo): ThinkingLevel[] | undefined {
	if (info.capabilities) {
		if (!info.capabilities.includes("thinking")) return undefined;
		return isGptOssModel(info) ? ["low", "medium", "high"] : ["off", "low", "medium", "high"];
	}

	const descriptor = getModelDescriptor(info);
	const likelyThinkingModel =
		(!descriptor.includes("qwen3-coder") && /\bqwen3(\b|[.:])/.test(descriptor)) ||
		descriptor.includes("qwen35") ||
		descriptor.includes("qwen3.5") ||
		descriptor.includes("qwen36") ||
		descriptor.includes("qwen3.6") ||
		descriptor.includes("deepseek-r1") ||
		descriptor.includes("deepseek-v3.1") ||
		descriptor.includes("deepseek-v4") ||
		descriptor.includes("glm-5") ||
		descriptor.includes("thinking");

	if (!likelyThinkingModel) return undefined;
	return isGptOssModel(info) ? ["low", "medium", "high"] : ["off", "low", "medium", "high"];
}

export function createOllamaModel(model: string | OllamaModelInfo): OllamaModel {
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
	} as OllamaModel;
}

export function loadOllamaModels(): Model<Api>[] {
	return getOllamaModelInfos().map(createOllamaModel);
}
