import { createServer, type IncomingMessage } from "node:http";
import type { AddressInfo } from "node:net";
import { readFileSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	defaultEmbeddingBatchSize,
	defaultEmbeddingConcurrency,
	defaultEmbeddingModel,
	embedTexts,
	embeddingBackendCacheKey,
	embeddingBackendDescription,
	processDataset,
} from "../src/quail/dataset-store.js";
import { getQuailDatasetsDir } from "../src/quail/paths.js";

const ENV_KEYS = [
	"OPENROUTER_API_KEY",
	"QUAIL_OPENROUTER_API_KEY",
	"QUAIL_OPENROUTER_EMBED_URL",
	"QUAIL_OPENROUTER_BASE_URL",
	"QUAIL_OPENROUTER_PROVIDER_ONLY",
	"QUAIL_OLLAMA_EMBED_URL",
	"QUAIL_EMBEDDING_PROVIDER",
	"QUAIL_EMBEDDING_MODEL",
	"QUAIL_EMBEDDING_BATCH_SIZE",
	"QUAIL_EMBEDDING_CONCURRENCY",
	"QUAIL_EMBEDDING_MAX_RETRIES",
	"QUAIL_EMBEDDING_RETRY_BASE_MS",
	"QUAIL_DISABLE_LOCAL_EMBEDDING_ENV",
	"QUAIL_WORKSPACE_PATH",
] as const;

type CapturedRequest = {
	url: string;
	headers: IncomingMessage["headers"];
	body: Record<string, unknown>;
};

async function readRequestBody(req: IncomingMessage): Promise<Record<string, unknown>> {
	const chunks: Buffer[] = [];
	for await (const chunk of req) chunks.push(Buffer.from(chunk));
	const text = Buffer.concat(chunks).toString("utf8");
	return text ? (JSON.parse(text) as Record<string, unknown>) : {};
}

async function startEmbeddingServer(
	handler: (request: CapturedRequest) => unknown | Promise<unknown>,
): Promise<{ url: string; requests: CapturedRequest[]; close: () => Promise<void> }> {
	const requests: CapturedRequest[] = [];
	const server = createServer(async (req, res) => {
		try {
			const captured = { url: req.url ?? "/", headers: req.headers, body: await readRequestBody(req) };
			requests.push(captured);
			const body = JSON.stringify(await handler(captured));
			res.writeHead(200, { "content-type": "application/json" });
			res.end(body);
		} catch (error) {
			res.writeHead(500, { "content-type": "application/json" });
			res.end(JSON.stringify({ error: error instanceof Error ? error.message : String(error) }));
		}
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const { port } = server.address() as AddressInfo;
	return {
		url: `http://127.0.0.1:${port}`,
		requests,
		close: () => new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve()))),
	};
}

describe("quail embedding provider", () => {
	let previousEnv: Record<string, string | undefined> = {};
	let closeServer: (() => Promise<void>) | undefined;

	beforeEach(() => {
		previousEnv = {};
		for (const key of ENV_KEYS) {
			previousEnv[key] = process.env[key];
			delete process.env[key];
		}
		process.env.QUAIL_DISABLE_LOCAL_EMBEDDING_ENV = "1";
	});

	afterEach(async () => {
		if (closeServer) {
			await closeServer();
			closeServer = undefined;
		}
		for (const key of ENV_KEYS) {
			if (previousEnv[key] === undefined) delete process.env[key];
			else process.env[key] = previousEnv[key];
		}
	});

	it("uses OpenRouter qwen embeddings and batch size 256 by default", async () => {
		const server = await startEmbeddingServer(() => ({
			data: [
				{ index: 1, embedding: [0, 4] },
				{ index: 0, embedding: [3, 0] },
			],
		}));
		closeServer = server.close;
		process.env.OPENROUTER_API_KEY = "test-key";
		process.env.QUAIL_OPENROUTER_EMBED_URL = `${server.url}/embeddings`;

		const index = await embedTexts(["alpha", "beta"]);

		expect(defaultEmbeddingModel()).toBe("qwen/qwen3-embedding-8b");
		expect(defaultEmbeddingBatchSize()).toBe(256);
		expect(defaultEmbeddingConcurrency()).toBe(20);
		expect(index.model).toBe("qwen/qwen3-embedding-8b");
		expect(index.dimensions).toBe(2);
		expect(Array.from(index.vectors["0"])).toEqual([1, 0]);
		expect(Array.from(index.vectors["1"])).toEqual([0, 1]);
		expect(server.requests).toHaveLength(1);
		expect(server.requests[0].body).toEqual({ model: "qwen/qwen3-embedding-8b", input: ["alpha", "beta"] });
		expect(server.requests[0].headers.authorization).toBe("Bearer test-key");
		expect(embeddingBackendDescription()).toContain("OpenRouter model qwen/qwen3-embedding-8b");
	});

	it("includes OpenRouter provider routing in requests and cache keys", async () => {
		const server = await startEmbeddingServer(() => ({ data: [{ index: 0, embedding: [1, 0] }] }));
		closeServer = server.close;
		process.env.OPENROUTER_API_KEY = "test-key";
		process.env.QUAIL_OPENROUTER_EMBED_URL = `${server.url}/embeddings`;
		process.env.QUAIL_OPENROUTER_PROVIDER_ONLY = "deepinfra";

		await embedTexts(["alpha"], { model: "qwen/qwen3-embedding-8b" });

		expect(server.requests[0].body.provider).toEqual({ only: ["deepinfra"] });
		expect(embeddingBackendCacheKey("qwen/qwen3-embedding-8b")).toContain("openrouter");
		expect(embeddingBackendCacheKey("qwen/qwen3-embedding-8b")).toContain("deepinfra");
	});

	it("sends embedding batches concurrently while preserving vector positions", async () => {
		let inFlight = 0;
		let maxInFlight = 0;
		const vectors: Record<string, number[]> = {
			alpha: [1, 0],
			beta: [0, 1],
			gamma: [-1, 0],
			delta: [0, -1],
		};
		const server = await startEmbeddingServer(async (request) => {
			inFlight++;
			maxInFlight = Math.max(maxInFlight, inFlight);
			await new Promise((resolve) => setTimeout(resolve, 40));
			inFlight--;
			const input = request.body.input as string[];
			return { data: input.map((text, index) => ({ index, embedding: vectors[text] })) };
		});
		closeServer = server.close;
		process.env.OPENROUTER_API_KEY = "test-key";
		process.env.QUAIL_OPENROUTER_EMBED_URL = `${server.url}/embeddings`;

		const index = await embedTexts(["alpha", "beta", "gamma", "delta"], { batchSize: 1, concurrency: 2 });

		expect(server.requests).toHaveLength(4);
		expect(maxInFlight).toBe(2);
		expect(Array.from(index.vectors["0"])).toEqual([1, 0]);
		expect(Array.from(index.vectors["1"])).toEqual([0, 1]);
		expect(Array.from(index.vectors["2"])).toEqual([-1, 0]);
		expect(Array.from(index.vectors["3"])).toEqual([0, -1]);
	});

	it("retries transient malformed OpenRouter embedding responses", async () => {
		let calls = 0;
		const server = await startEmbeddingServer(() => {
			calls++;
			return calls === 1 ? { data: [] } : { data: [{ index: 0, embedding: [3, 4] }] };
		});
		closeServer = server.close;
		process.env.OPENROUTER_API_KEY = "test-key";
		process.env.QUAIL_OPENROUTER_EMBED_URL = `${server.url}/embeddings`;
		process.env.QUAIL_EMBEDDING_MAX_RETRIES = "1";
		process.env.QUAIL_EMBEDDING_RETRY_BASE_MS = "0";

		const index = await embedTexts(["alpha"], { batchSize: 1 });

		expect(calls).toBe(2);
		expect(index.vectors["0"][0]).toBeCloseTo(0.6);
		expect(index.vectors["0"][1]).toBeCloseTo(0.8);
	});

	it("can use an Ollama-compatible embedding URL for the OpenRouter proxy", async () => {
		const server = await startEmbeddingServer(() => ({ embeddings: [[3, 4]] }));
		closeServer = server.close;
		process.env.QUAIL_EMBEDDING_PROVIDER = "ollama";
		process.env.QUAIL_OLLAMA_EMBED_URL = `${server.url}/api/embed`;

		const index = await embedTexts(["alpha"], { model: "proxy-model" });

		expect(server.requests[0].body).toEqual({ model: "proxy-model", input: ["alpha"] });
		expect(index.vectors["0"][0]).toBeCloseTo(0.6);
		expect(index.vectors["0"][1]).toBeCloseTo(0.8);
		expect(embeddingBackendCacheKey("proxy-model")).toContain("ollama");
		expect(embeddingBackendCacheKey("proxy-model")).toContain(`${server.url}/api/embed`);
	});

	it("records the new embedding defaults in processed dataset manifests", async () => {
		const cwd = mkdtempSync(join(tmpdir(), "quail-embedding-provider-"));
		try {
			process.env.QUAIL_WORKSPACE_PATH = join(cwd, "workspace");
			const inputPath = join(cwd, "dataset.csv");
			writeFileSync(inputPath, "text\nhello world\n", "utf8");

			await processDataset({ cwd, inputPath, name: "Defaults Check", skipEmbeddings: true });

			const manifest = JSON.parse(
				readFileSync(join(getQuailDatasetsDir(cwd), "defaults-check", "manifest.json"), "utf8"),
			) as Record<string, unknown>;
			expect(manifest.embeddingModel).toBe("qwen/qwen3-embedding-8b");
			expect(manifest.batchSize).toBe(256);
		} finally {
			rmSync(cwd, { recursive: true, force: true });
		}
	});

	it("loads openrouter.env from the processing cwd when hatch is launched from the Quail v0.7 root", async () => {
		const cwd = mkdtempSync(join(tmpdir(), "quail-embedding-local-env-"));
		try {
			delete process.env.QUAIL_DISABLE_LOCAL_EMBEDDING_ENV;
			process.env.QUAIL_WORKSPACE_PATH = join(cwd, "workspace");
			writeFileSync(
				join(cwd, "openrouter.env"),
				[
					"QUAIL_EMBEDDING_MODEL=local-env-model",
					"QUAIL_EMBEDDING_BATCH_SIZE=129",
					"QUAIL_EMBEDDING_CONCURRENCY=3",
				].join("\n"),
				"utf8",
			);
			const inputPath = join(cwd, "dataset.csv");
			writeFileSync(inputPath, "text\nhello world\n", "utf8");

			await processDataset({ cwd, inputPath, name: "Local Env Check", skipEmbeddings: true });

			const manifest = JSON.parse(
				readFileSync(join(getQuailDatasetsDir(cwd), "local-env-check", "manifest.json"), "utf8"),
			) as Record<string, unknown>;
			expect(manifest.embeddingModel).toBe("local-env-model");
			expect(manifest.batchSize).toBe(129);
			expect(defaultEmbeddingConcurrency()).toBe(3);
		} finally {
			rmSync(cwd, { recursive: true, force: true });
		}
	});

	it("fails clearly if OpenRouter credentials are missing", async () => {
		const server = await startEmbeddingServer(() => ({ data: [{ index: 0, embedding: [1, 0] }] }));
		closeServer = server.close;
		process.env.QUAIL_OPENROUTER_EMBED_URL = `${server.url}/embeddings`;

		await expect(embedTexts(["alpha"])).rejects.toThrow("OpenRouter embeddings require OPENROUTER_API_KEY");
		expect(server.requests).toHaveLength(0);
	});
});
