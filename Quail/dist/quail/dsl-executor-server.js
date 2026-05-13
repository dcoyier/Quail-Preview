import http from "node:http";
import { clearQuailDslRuntimeCaches, executeQuailCallBlocks, getQuailDslRuntimeCacheStats, } from "./dsl.js";
function defaultLog(event, payload = {}) {
    console.log(JSON.stringify({ timestamp: new Date().toISOString(), event, ...payload }));
}
function summarizeBlocks(blocks) {
    return blocks.map((block) => ({
        datasets: block.datasets,
        codeBytes: typeof block.code === "string" ? Buffer.byteLength(block.code) : 0,
        codePreview: typeof block.code === "string" ? block.code.slice(0, 140).replace(/\s+/g, " ") : "",
    }));
}
function readBody(req, maxBodyBytes) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        let size = 0;
        req.on("data", (chunk) => {
            size += chunk.length;
            if (size > maxBodyBytes) {
                reject(new Error(`request body exceeds ${maxBodyBytes} bytes`));
                req.destroy();
                return;
            }
            chunks.push(chunk);
        });
        req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
        req.on("error", reject);
    });
}
function sendJson(res, status, payload) {
    const body = JSON.stringify(payload);
    res.writeHead(status, {
        "content-type": "application/json; charset=utf-8",
        "content-length": Buffer.byteLength(body),
    });
    res.end(body);
}
function isExecutionPayload(value) {
    if (!value || typeof value !== "object")
        return false;
    const payload = value;
    return payload.version === 1 &&
        typeof payload.cwd === "string" &&
        !!payload.state &&
        typeof payload.state === "object" &&
        Array.isArray(payload.blocks);
}
export async function startQuailDslExecutorServer(options = {}) {
    const host = options.host ?? process.env.QUAIL_DSL_EXECUTOR_HOST ?? "127.0.0.1";
    const port = options.port ?? Number(process.env.QUAIL_DSL_EXECUTOR_PORT || 0);
    const maxBodyBytes = options.maxBodyBytes ?? Number(process.env.QUAIL_DSL_EXECUTOR_MAX_BODY_BYTES || 128 * 1024 * 1024);
    const log = options.log ?? defaultLog;
    process.env.QUAIL_DSL_EXECUTOR_DISABLE = "1";
    let queue = Promise.resolve();
    let nextRequestId = 1;
    const requests = new Map();
    const statusPayload = () => {
        const values = [...requests.values()];
        return {
            ok: true,
            queued: values.filter((item) => item.status === "queued").length,
            running: values.filter((item) => item.status === "running").length,
            cache: getQuailDslRuntimeCacheStats(),
            recent: values.slice(-20),
        };
    };
    const handleExecute = async (req, res) => {
        const body = await readBody(req, maxBodyBytes);
        const payload = JSON.parse(body || "{}");
        if (!isExecutionPayload(payload))
            throw new Error("Expected payload { version: 1, cwd, state, blocks }");
        const requestId = nextRequestId++;
        const queuedAt = Date.now();
        const blocks = summarizeBlocks(payload.blocks);
        requests.set(requestId, { id: requestId, status: "queued", queuedAt, blocks });
        log("request_queued", { requestId, queued: requests.size, blocks });
        const run = async () => {
            const startedAt = Date.now();
            const record = requests.get(requestId);
            if (record) {
                record.status = "running";
                record.startedAt = startedAt;
            }
            log("request_started", { requestId, queuedMs: startedAt - queuedAt });
            try {
                const result = await executeQuailCallBlocks({
                    cwd: payload.cwd,
                    state: payload.state,
                    blocks: payload.blocks,
                });
                const completedAt = Date.now();
                if (record) {
                    record.status = "completed";
                    record.completedAt = completedAt;
                    record.errors = result.errors.length;
                    record.outputBytes = Buffer.byteLength(result.output || "");
                }
                log("request_completed", {
                    requestId,
                    runMs: completedAt - startedAt,
                    totalMs: completedAt - queuedAt,
                    errors: result.errors.length,
                    outputBytes: Buffer.byteLength(result.output || ""),
                    cache: getQuailDslRuntimeCacheStats(),
                });
                return result;
            }
            catch (error) {
                const completedAt = Date.now();
                if (record) {
                    record.status = "failed";
                    record.completedAt = completedAt;
                    record.error = error instanceof Error ? error.message : String(error);
                }
                log("request_failed", {
                    requestId,
                    runMs: completedAt - startedAt,
                    totalMs: completedAt - queuedAt,
                    error: error instanceof Error ? error.message : String(error),
                });
                throw error;
            }
            finally {
                setTimeout(() => requests.delete(requestId), 10 * 60 * 1000).unref();
            }
        };
        const result = await (queue = queue.then(run, run));
        sendJson(res, 200, { ok: true, result });
    };
    const server = http.createServer((req, res) => {
        (async () => {
            if (req.method === "GET" && req.url === "/health") {
                sendJson(res, 200, { ok: true, cache: getQuailDslRuntimeCacheStats() });
                return;
            }
            if (req.method === "GET" && req.url === "/status") {
                sendJson(res, 200, statusPayload());
                return;
            }
            if (req.method === "POST" && req.url === "/cache/clear") {
                clearQuailDslRuntimeCaches();
                sendJson(res, 200, { ok: true, cache: getQuailDslRuntimeCacheStats() });
                return;
            }
            if (req.method === "POST" && req.url === "/quail/execute") {
                await handleExecute(req, res);
                return;
            }
            sendJson(res, 404, { ok: false, error: "not found" });
        })().catch((error) => {
            sendJson(res, 500, { ok: false, error: error instanceof Error ? error.message : String(error) });
        });
    });
    await new Promise((resolve, reject) => {
        server.once("error", reject);
        server.listen(port, host, () => {
            server.off("error", reject);
            const address = server.address();
            const actualPort = typeof address === "object" && address ? address.port : port;
            log("executor_listening", { url: `http://${host}:${actualPort}` });
            resolve();
        });
    });
    return server;
}
//# sourceMappingURL=dsl-executor-server.js.map