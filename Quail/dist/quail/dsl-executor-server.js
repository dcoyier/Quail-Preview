import http from "node:http";
import { clearQuailDslRuntimeCaches, executeQuailCallBlocks, getQuailDslRuntimeCacheStats, } from "./dsl.js";
import { createEmptyAnalysisState } from "./analysis-state.js";
class ExecutorHttpError extends Error {
    status;
    code;
    constructor(status, code, message) {
        super(message);
        this.status = status;
        this.code = code;
        this.name = "ExecutorHttpError";
    }
}
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
        let settled = false;
        req.on("data", (chunk) => {
            if (settled)
                return;
            size += chunk.length;
            if (size > maxBodyBytes) {
                settled = true;
                req.resume();
                reject(new ExecutorHttpError(413, "E_BODY_TOO_LARGE", `Request body exceeds ${maxBodyBytes} bytes.`));
                return;
            }
            chunks.push(chunk);
        });
        req.on("end", () => {
            if (settled)
                return;
            settled = true;
            resolve(Buffer.concat(chunks).toString("utf8"));
        });
        req.on("error", (error) => {
            if (settled)
                return;
            settled = true;
            reject(error);
        });
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
function parseJsonBody(body) {
    try {
        return JSON.parse(body || "{}");
    }
    catch (error) {
        throw new ExecutorHttpError(400, "E_BAD_JSON", error instanceof Error ? error.message : String(error));
    }
}
function isQuailCallBlock(value) {
    if (!value || typeof value !== "object")
        return false;
    const block = value;
    return Array.isArray(block.datasets) &&
        block.datasets.every((item) => typeof item === "string") &&
        typeof block.code === "string" &&
        (block.raw === undefined || typeof block.raw === "string");
}
function isExecutionPayload(value) {
    if (!value || typeof value !== "object")
        return false;
    const payload = value;
    return payload.version === 1 &&
        typeof payload.cwd === "string" &&
        !!payload.state &&
        typeof payload.state === "object" &&
        Array.isArray(payload.blocks) &&
        payload.blocks.every(isQuailCallBlock);
}
function isPrewarmPayload(value) {
    if (!value || typeof value !== "object")
        return false;
    const payload = value;
    return payload.version === 1 &&
        typeof payload.cwd === "string" &&
        Array.isArray(payload.datasets) &&
        payload.datasets.every((item) => typeof item === "string");
}
export async function startQuailDslExecutorServer(options = {}) {
    const host = options.host ?? process.env.QUAIL_DSL_EXECUTOR_HOST ?? "127.0.0.1";
    const port = options.port ?? Number(process.env.QUAIL_DSL_EXECUTOR_PORT || 0);
    const maxBodyBytes = options.maxBodyBytes ?? Number(process.env.QUAIL_DSL_EXECUTOR_MAX_BODY_BYTES || 128 * 1024 * 1024);
    const log = options.log ?? defaultLog;
    const previousExecutorDisable = process.env.QUAIL_DSL_EXECUTOR_DISABLE;
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
        const payload = parseJsonBody(body);
        if (!isExecutionPayload(payload)) {
            throw new ExecutorHttpError(400, "E_BAD_EXECUTE_PAYLOAD", "Expected payload { version: 1, cwd, state, blocks }, where blocks contain datasets and code.");
        }
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
    const handlePrewarm = async (req, res) => {
        const body = await readBody(req, maxBodyBytes);
        const payload = parseJsonBody(body);
        if (!isPrewarmPayload(payload)) {
            throw new ExecutorHttpError(400, "E_BAD_PREWARM_PAYLOAD", "Expected payload { version: 1, cwd, datasets }.");
        }
        const startedAt = Date.now();
        const result = await executeQuailCallBlocks({
            cwd: payload.cwd,
            state: createEmptyAnalysisState(),
            blocks: [{ datasets: payload.datasets, code: "", raw: "" }],
        });
        const completedAt = Date.now();
        log("prewarm_completed", {
            datasets: payload.datasets,
            runMs: completedAt - startedAt,
            errors: result.errors.length,
            cache: getQuailDslRuntimeCacheStats(),
        });
        sendJson(res, 200, { ok: true, errors: result.errors, cache: getQuailDslRuntimeCacheStats() });
    };
    const handleError = (res, error) => {
        if (res.headersSent) {
            res.end();
            return;
        }
        if (error instanceof ExecutorHttpError) {
            sendJson(res, error.status, { ok: false, code: error.code, error: error.message });
            return;
        }
        sendJson(res, 500, {
            ok: false,
            code: "E_EXECUTOR_INTERNAL",
            error: error instanceof Error ? error.message : String(error),
        });
    };
    const server = http.createServer((req, res) => {
        (async () => {
            if (!req.url) {
                sendJson(res, 400, { ok: false, code: "E_BAD_REQUEST", error: "Missing request URL." });
                return;
            }
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
            if (req.method === "POST" && req.url === "/quail/prewarm") {
                await handlePrewarm(req, res);
                return;
            }
            const postOnly = new Set(["/cache/clear", "/quail/execute", "/quail/prewarm"]);
            const getOnly = new Set(["/health", "/status"]);
            if (postOnly.has(req.url) || getOnly.has(req.url)) {
                sendJson(res, 405, { ok: false, code: "E_METHOD_NOT_ALLOWED", error: `Method ${req.method} is not allowed for ${req.url}.` });
                return;
            }
            sendJson(res, 404, { ok: false, code: "E_NOT_FOUND", error: "Not found." });
        })().catch((error) => {
            handleError(res, error);
        });
    });
    server.on("close", () => {
        if (previousExecutorDisable === undefined)
            delete process.env.QUAIL_DSL_EXECUTOR_DISABLE;
        else
            process.env.QUAIL_DSL_EXECUTOR_DISABLE = previousExecutorDisable;
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