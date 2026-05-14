import type http from "node:http";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createEmptyAnalysisState } from "../src/quail/analysis-state.js";
import { processDataset } from "../src/quail/dataset-store.js";
import { startQuailDslExecutorServer } from "../src/quail/dsl-executor-server.js";

describe("Quail DSL executor server", () => {
	let cwd: string;
	let server: http.Server;
	let baseUrl: string;
	let previousWorkspacePath: string | undefined;
	let previousWorkspaceScope: string | undefined;
	let previousExecutorDisable: string | undefined;

	beforeEach(async () => {
		previousWorkspacePath = process.env.QUAIL_WORKSPACE_PATH;
		previousWorkspaceScope = process.env.QUAIL_WORKSPACE_SCOPE;
		previousExecutorDisable = process.env.QUAIL_DSL_EXECUTOR_DISABLE;
		cwd = mkdtempSync(join(tmpdir(), "quail-executor-"));
		process.env.QUAIL_WORKSPACE_PATH = join(cwd, "workspace");
		delete process.env.QUAIL_WORKSPACE_SCOPE;
		delete process.env.QUAIL_DSL_EXECUTOR_DISABLE;

		const inputPath = join(cwd, "dataset.csv");
		writeFileSync(inputPath, ["text,kind", "Alpha apple,fruit", "Beta banana,fruit"].join("\n"), "utf8");
		await processDataset({ cwd, inputPath, name: "Executor Check", skipEmbeddings: true });

		server = await startQuailDslExecutorServer({
			port: 0,
			maxBodyBytes: 2048,
			log: () => {},
		});
		const address = server.address();
		if (!address || typeof address === "string") throw new Error("Expected executor to listen on a TCP port");
		baseUrl = `http://127.0.0.1:${address.port}`;
	});

	afterEach(async () => {
		await closeServer(server);
		if (previousWorkspacePath === undefined) delete process.env.QUAIL_WORKSPACE_PATH;
		else process.env.QUAIL_WORKSPACE_PATH = previousWorkspacePath;
		if (previousWorkspaceScope === undefined) delete process.env.QUAIL_WORKSPACE_SCOPE;
		else process.env.QUAIL_WORKSPACE_SCOPE = previousWorkspaceScope;
		if (previousExecutorDisable === undefined) delete process.env.QUAIL_DSL_EXECUTOR_DISABLE;
		else process.env.QUAIL_DSL_EXECUTOR_DISABLE = previousExecutorDisable;
		rmSync(cwd, { recursive: true, force: true });
	});

	async function request(path: string, init?: RequestInit): Promise<{ status: number; body: Record<string, unknown> }> {
		const response = await fetch(`${baseUrl}${path}`, init);
		return { status: response.status, body: await response.json() as Record<string, unknown> };
	}

	it("serves health, status, cache clear, prewarm, and execute endpoints", async () => {
		const health = await request("/health");
		expect(health.status).toBe(200);
		expect(health.body.ok).toBe(true);

		const prewarm = await request("/quail/prewarm", {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({ version: 1, cwd, datasets: ["Executor Check"] }),
		});
		expect(prewarm.status).toBe(200);
		expect(prewarm.body.ok).toBe(true);
		expect(prewarm.body.errors).toEqual([]);

		const execute = await request("/quail/execute", {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({
				version: 1,
				cwd,
				state: createEmptyAnalysisState(),
				blocks: [{
					datasets: ["Executor Check"],
					code: 'print(count(G0))\nprint(count(temp(contains: ["text": "apple"])))',
					raw: "",
				}],
			}),
		});
		expect(execute.status).toBe(200);
		expect((execute.body.result as { output: string }).output.trim().split("\n")).toEqual(["2", "1"]);

		const status = await request("/status");
		expect(status.status).toBe(200);
		expect(status.body.ok).toBe(true);
		expect(status.body.recent).toEqual(expect.arrayContaining([
			expect.objectContaining({ status: "completed", errors: 0 }),
		]));

		const clear = await request("/cache/clear", { method: "POST" });
		expect(clear.status).toBe(200);
		expect(clear.body.ok).toBe(true);
	});

	it("serializes concurrent executions while sharing optimized runtime caches", async () => {
		const clear = await request("/cache/clear", { method: "POST" });
		expect(clear.status).toBe(200);

		const body = JSON.stringify({
			version: 1,
			cwd,
			state: createEmptyAnalysisState(),
			blocks: [{
				datasets: ["Executor Check"],
				code: [
					`for threshold in [0.0, 0.0]:`,
					`    print(count(entries of (scope: G0, ([text] BM25 similarity to "apple" > threshold))))`,
				].join("\n"),
				raw: "",
			}],
		});

		const [first, second] = await Promise.all([
			request("/quail/execute", { method: "POST", headers: { "content-type": "application/json" }, body }),
			request("/quail/execute", { method: "POST", headers: { "content-type": "application/json" }, body }),
		]);
		expect(first.status).toBe(200);
		expect(second.status).toBe(200);
		expect((first.body.result as { output: string }).output.trim().split("\n")).toEqual(["1", "1"]);
		expect((second.body.result as { output: string }).output.trim().split("\n")).toEqual(["1", "1"]);

		const status = await request("/status");
		const recent = status.body.recent as Array<{ status: string; startedAt: number; completedAt: number; errors: number }>;
		expect(recent).toHaveLength(2);
		expect(recent).toEqual([
			expect.objectContaining({ status: "completed", errors: 0 }),
			expect.objectContaining({ status: "completed", errors: 0 }),
		]);
		expect(recent[1].startedAt).toBeGreaterThanOrEqual(recent[0].completedAt);

		const cache = status.body.cache as { scoreVectorMisses: number; scoreVectorHits: number; thresholdIdSetMisses: number; thresholdIdSetHits: number };
		expect(cache.scoreVectorMisses).toBe(1);
		expect(cache.scoreVectorHits).toBeGreaterThan(0);
		expect(cache.thresholdIdSetMisses).toBe(1);
		expect(cache.thresholdIdSetHits).toBeGreaterThan(0);
	});

	it("returns structured 4xx errors for invalid requests", async () => {
		const badJson = await request("/quail/execute", {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: "{",
		});
		expect(badJson.status).toBe(400);
		expect(badJson.body).toMatchObject({ ok: false, code: "E_BAD_JSON" });

		const badPayload = await request("/quail/execute", {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({ version: 1, cwd, state: {}, blocks: [{ datasets: ["Executor Check"] }] }),
		});
		expect(badPayload.status).toBe(400);
		expect(badPayload.body).toMatchObject({ ok: false, code: "E_BAD_EXECUTE_PAYLOAD" });

		const wrongMethod = await request("/quail/execute");
		expect(wrongMethod.status).toBe(405);
		expect(wrongMethod.body).toMatchObject({ ok: false, code: "E_METHOD_NOT_ALLOWED" });

		const missing = await request("/missing");
		expect(missing.status).toBe(404);
		expect(missing.body).toMatchObject({ ok: false, code: "E_NOT_FOUND" });
	});

	it("rejects oversized bodies without crashing the executor", async () => {
		const tinyServer = await startQuailDslExecutorServer({
			port: 0,
			maxBodyBytes: 16,
			log: () => {},
		});
		try {
			const address = tinyServer.address();
			if (!address || typeof address === "string") throw new Error("Expected executor to listen on a TCP port");
			const tinyUrl = `http://127.0.0.1:${address.port}`;
			const response = await fetch(`${tinyUrl}/quail/execute`, {
				method: "POST",
				headers: { "content-type": "application/json" },
				body: JSON.stringify({ thisBodyIs: "too large" }),
			});
			expect(response.status).toBe(413);
			expect(await response.json()).toMatchObject({ ok: false, code: "E_BODY_TOO_LARGE" });

			const health = await fetch(`${tinyUrl}/health`);
			expect(health.status).toBe(200);
		} finally {
			await closeServer(tinyServer);
		}
	});

	it("restores QUAIL_DSL_EXECUTOR_DISABLE when the server closes", async () => {
		await closeServer(server);
		expect(process.env.QUAIL_DSL_EXECUTOR_DISABLE).toBe(previousExecutorDisable);

		server = await startQuailDslExecutorServer({ port: 0, log: () => {} });
		expect(process.env.QUAIL_DSL_EXECUTOR_DISABLE).toBe("1");
	});
});

function closeServer(server: http.Server | undefined): Promise<void> {
	if (!server || !server.listening) return Promise.resolve();
	return new Promise((resolve, reject) => {
		server.close((error) => {
			if (error) reject(error);
			else resolve();
		});
	});
}
