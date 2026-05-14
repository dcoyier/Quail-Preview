#!/usr/bin/env node
import http from "node:http";

const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = "11435";
const DEFAULT_MODEL = "qwen/qwen3-embedding-8b";
const DEFAULT_BASE_URL = "https://openrouter.ai/api/v1";

function argValue(name) {
	const index = process.argv.indexOf(name);
	return index >= 0 ? process.argv[index + 1] : undefined;
}

function envValue(name) {
	const value = process.env[name]?.trim();
	return value || undefined;
}

const host = argValue("--host") ?? envValue("QUAIL_OPENROUTER_EMBED_PROXY_HOST") ?? DEFAULT_HOST;
const port = Number(argValue("--port") ?? envValue("QUAIL_OPENROUTER_EMBED_PROXY_PORT") ?? DEFAULT_PORT);
const defaultModel = argValue("--model") ?? envValue("QUAIL_EMBEDDING_MODEL") ?? DEFAULT_MODEL;
const baseUrl = (argValue("--base-url") ?? envValue("QUAIL_OPENROUTER_BASE_URL") ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
const providerOnly = argValue("--provider-only") ?? envValue("QUAIL_OPENROUTER_PROVIDER_ONLY");
const apiKey =
	envValue("QUAIL_OPENROUTER_API_KEY") ??
	envValue(argValue("--api-key-env") ?? "") ??
	envValue("OPENROUTER_API_KEY");

function sendJson(res, status, value) {
	const body = JSON.stringify(value);
	res.writeHead(status, {
		"content-type": "application/json",
		"content-length": Buffer.byteLength(body),
	});
	res.end(body);
}

function sendError(res, status, message) {
	sendJson(res, status, { error: message });
}

async function readJson(req) {
	const chunks = [];
	let size = 0;
	for await (const chunk of req) {
		size += chunk.length;
		if (size > 32 * 1024 * 1024) throw new Error("Request body is larger than 32 MB");
		chunks.push(chunk);
	}
	const text = Buffer.concat(chunks).toString("utf8").trim();
	return text ? JSON.parse(text) : {};
}

function vectorFrom(value) {
	if (!Array.isArray(value)) return undefined;
	const vector = value.map((item) => Number(item));
	return vector.every(Number.isFinite) ? vector : undefined;
}

function parseOpenRouterEmbeddings(payload) {
	if (Array.isArray(payload?.data)) {
		return payload.data
			.map((row, fallbackIndex) => ({
				index: Number.isInteger(row?.index) ? row.index : fallbackIndex,
				embedding: vectorFrom(row?.embedding),
			}))
			.filter((row) => row.embedding)
			.sort((a, b) => a.index - b.index)
			.map((row) => row.embedding);
	}
	if (Array.isArray(payload?.embeddings)) return payload.embeddings.map(vectorFrom).filter(Boolean);
	const embedding = vectorFrom(payload?.embedding);
	return embedding ? [embedding] : [];
}

async function embed(inputs, model) {
	if (!apiKey) throw new Error("Set OPENROUTER_API_KEY or QUAIL_OPENROUTER_API_KEY before starting the proxy.");
	const body = { model, input: inputs };
	if (providerOnly) body.provider = { only: [providerOnly] };
	const response = await fetch(`${baseUrl}/embeddings`, {
		method: "POST",
		headers: {
			"authorization": `Bearer ${apiKey}`,
			"content-type": "application/json",
			"http-referer": "https://github.com/quail",
			"x-title": "Quail v0.7",
		},
		body: JSON.stringify(body),
	});
	if (!response.ok) {
		const text = await response.text().catch(() => "");
		throw new Error(`OpenRouter embed failed (${response.status}): ${text || response.statusText}`);
	}
	const embeddings = parseOpenRouterEmbeddings(await response.json());
	if (embeddings.length !== inputs.length) {
		throw new Error(`OpenRouter returned ${embeddings.length} embedding(s) for ${inputs.length} input(s).`);
	}
	return embeddings;
}

function normalizeInputs(body) {
	if (Array.isArray(body.input)) return body.input.map((value) => String(value ?? ""));
	if (body.input !== undefined) return [String(body.input ?? "")];
	if (body.prompt !== undefined) return [String(body.prompt ?? "")];
	return [""];
}

const server = http.createServer(async (req, res) => {
	try {
		const url = new URL(req.url ?? "/", `http://${host}:${port}`);
		if (req.method === "GET" && url.pathname === "/api/tags") {
			sendJson(res, 200, { models: [{ name: defaultModel, model: defaultModel, modified_at: new Date(0).toISOString() }] });
			return;
		}
		if (req.method !== "POST" || (url.pathname !== "/api/embed" && url.pathname !== "/api/embeddings")) {
			sendError(res, 404, "Use POST /api/embed or POST /api/embeddings.");
			return;
		}
		const body = await readJson(req);
		const model = String(body.model ?? defaultModel);
		const inputs = normalizeInputs(body);
		const embeddings = await embed(inputs, model);
		if (url.pathname === "/api/embeddings") {
			sendJson(res, 200, { model, embedding: embeddings[0] ?? [] });
			return;
		}
		sendJson(res, 200, {
			model,
			embeddings,
			total_duration: 0,
			load_duration: 0,
			prompt_eval_count: inputs.length,
		});
	} catch (error) {
		sendError(res, 500, error instanceof Error ? error.message : String(error));
	}
});

server.listen(port, host, () => {
	console.error(`OpenRouter Ollama-compatible embedding proxy listening at http://${host}:${port}/api/embed`);
	console.error(`Model: ${defaultModel}${providerOnly ? ` via provider ${providerOnly}` : ""}`);
});
