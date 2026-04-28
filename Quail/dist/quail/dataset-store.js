import { closeSync, existsSync, mkdirSync, openSync, readdirSync, readFileSync, rmSync, writeFileSync, writeSync, } from "node:fs";
import { basename, extname, join } from "node:path";
import { ensureQuailWorkspace, getQuailDatasetsDir } from "./paths.js";
import { normalizeContainsText, slugifyDatasetName, stableEntryId, tokenize } from "./text.js";
const ENTRIES_FILE = "entries.jsonl";
const BM25_FILE = "bm25.json";
const EMBEDDINGS_FILE = "embeddings.json";
const EMBEDDINGS_VECTOR_FILE = "embeddings.f32";
const MANIFEST_FILE = "manifest.json";
const ROOT_MANIFEST_FILE = "manifest.json";
const DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest";
const DEFAULT_BATCH_SIZE = 64;
function datasetDir(cwd, slug) {
    return join(getQuailDatasetsDir(cwd), slug);
}
function readJson(path) {
    return JSON.parse(readFileSync(path, "utf8"));
}
function writeJson(path, value) {
    writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}
function isBinaryEmbeddingIndexFile(value) {
    if (typeof value !== "object" || value === null)
        return false;
    const record = value;
    return record.format === "float32-binary-v1" && typeof record.model === "string";
}
function writeEmbeddingIndex(dir, fileName, index) {
    const ids = Object.keys(index.vectors);
    const dimensions = index.dimensions;
    const vectorsFile = EMBEDDINGS_VECTOR_FILE;
    const vectorPath = join(dir, vectorsFile);
    if (ids.length === 0 || dimensions === 0) {
        writeFileSync(vectorPath, "");
    }
    else {
        const vectorsPerChunk = Math.max(1, Math.min(256, Math.floor((8 * 1024 * 1024) / (dimensions * 4))));
        const buffer = Buffer.allocUnsafe(vectorsPerChunk * dimensions * 4);
        const fd = openSync(vectorPath, "w");
        try {
            for (let start = 0; start < ids.length; start += vectorsPerChunk) {
                const count = Math.min(vectorsPerChunk, ids.length - start);
                let offset = 0;
                for (let row = 0; row < count; row++) {
                    const vector = index.vectors[ids[start + row]];
                    for (let dimension = 0; dimension < dimensions; dimension++) {
                        const value = Number(vector?.[dimension] ?? 0);
                        buffer.writeFloatLE(Number.isFinite(value) ? value : 0, offset);
                        offset += 4;
                    }
                }
                writeSync(fd, buffer, 0, offset);
            }
        }
        finally {
            closeSync(fd);
        }
    }
    writeJson(join(dir, fileName), {
        format: "float32-binary-v1",
        model: index.model,
        dimensions,
        count: ids.length,
        ids,
        vectorsFile,
    });
}
function loadEmbeddingIndex(dir, fileName) {
    const raw = readJson(join(dir, fileName));
    if (!isBinaryEmbeddingIndexFile(raw))
        return raw;
    const dimensions = Math.max(0, Math.floor(Number(raw.dimensions) || 0));
    const ids = Array.isArray(raw.ids) ? raw.ids.map(String) : [];
    const data = readFileSync(join(dir, raw.vectorsFile || EMBEDDINGS_VECTOR_FILE));
    const vectors = {};
    const bytesPerVector = dimensions * 4;
    const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
    for (let i = 0; i < ids.length; i++) {
        const byteOffset = i * bytesPerVector;
        if (dimensions === 0 || byteOffset + bytesPerVector > data.byteLength) {
            vectors[ids[i]] = [];
            continue;
        }
        const absoluteOffset = data.byteOffset + byteOffset;
        if (littleEndian && absoluteOffset % 4 === 0) {
            vectors[ids[i]] = new Float32Array(data.buffer, absoluteOffset, dimensions);
            continue;
        }
        const vector = new Float32Array(dimensions);
        for (let dimension = 0; dimension < dimensions; dimension++)
            vector[dimension] = data.readFloatLE(byteOffset + dimension * 4);
        vectors[ids[i]] = vector;
    }
    return { model: raw.model, dimensions, vectors };
}
function splitDelimitedLine(line, delimiter) {
    const out = [];
    let current = "";
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') {
            if (quoted && line[i + 1] === '"') {
                current += '"';
                i++;
            }
            else {
                quoted = !quoted;
            }
            continue;
        }
        if (!quoted && ch === delimiter) {
            out.push(current);
            current = "";
            continue;
        }
        current += ch;
    }
    out.push(current);
    return out.map((cell) => cell.trim());
}
function parseDelimited(content, delimiter, textColumn) {
    const lines = content.replace(/^\uFEFF/, "").split(/\r?\n/).filter((line) => line.trim().length > 0);
    if (lines.length === 0)
        return [];
    const headers = splitDelimitedLine(lines[0], delimiter).map((h) => h.trim());
    const lowerHeaders = headers.map((h) => h.toLowerCase());
    const requested = textColumn ? lowerHeaders.indexOf(textColumn.toLowerCase()) : -1;
    const preferredNames = ["text", "response", "content", "answer", "comment", "body"];
    let textIndex = requested;
    if (textIndex < 0) {
        textIndex = preferredNames.map((name) => lowerHeaders.indexOf(name)).find((index) => index >= 0) ?? -1;
    }
    if (textIndex < 0)
        textIndex = 0;
    return lines.slice(1).flatMap((line) => {
        const cells = splitDelimitedLine(line, delimiter);
        const text = (cells[textIndex] ?? "").trim();
        if (!text)
            return [];
        const tags = {};
        for (let i = 0; i < headers.length; i++) {
            if (i === textIndex)
                continue;
            const value = (cells[i] ?? "").trim();
            if (value)
                tags[headers[i]] = value;
        }
        return [{ text, tags }];
    });
}
function parseJsonCorpus(content, textColumn) {
    const parsed = JSON.parse(content);
    const array = Array.isArray(parsed)
        ? parsed
        : typeof parsed === "object" && parsed !== null
            ? parsed.entries ||
                parsed.responses ||
                parsed.data
            : undefined;
    if (!Array.isArray(array)) {
        throw new Error("JSON corpus must be an array or an object with entries/responses/data array");
    }
    const preferred = [textColumn, "text", "response", "content", "answer", "comment", "body"].filter((value) => typeof value === "string" && value.length > 0);
    const addTag = (tags, key, value, fallbackPrefix) => {
        if (value === undefined || value === null)
            return;
        if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
            const tagKey = tags[key] === undefined ? key : fallbackPrefix ? `${fallbackPrefix}.${key}` : key;
            tags[tagKey] = String(value);
        }
    };
    const addTagsFromValue = (tags, key, value, options) => {
        if (typeof value === "object" && value !== null && !Array.isArray(value)) {
            for (const [nestedKey, nestedValue] of Object.entries(value)) {
                addTag(tags, options?.flatten ? nestedKey : `${key}.${nestedKey}`, nestedValue, key);
            }
            return;
        }
        addTag(tags, key, value);
    };
    return array.flatMap((item) => {
        if (typeof item === "string") {
            const text = item.trim();
            return text ? [{ text, tags: {} }] : [];
        }
        if (typeof item !== "object" || item === null)
            return [];
        const record = item;
        const textKey = preferred.find((key) => typeof record[key] === "string" && String(record[key]).trim());
        if (!textKey)
            return [];
        const text = String(record[textKey]).trim();
        const tags = {};
        for (const [key, value] of Object.entries(record)) {
            if (key === textKey || value === undefined || value === null)
                continue;
            addTagsFromValue(tags, key, value, { flatten: key === "metadata" || key === "tags" });
        }
        return [{ text, tags }];
    });
}
function parseJsonlCorpus(content, textColumn) {
    return content
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .flatMap((line) => parseJsonCorpus(`[${line}]`, textColumn));
}
function parseTextCorpus(content) {
    const normalized = content.replace(/\r\n/g, "\n").trim();
    if (!normalized)
        return [];
    const paragraphEntries = normalized.split(/\n\s*\n+/).map((part) => part.trim()).filter(Boolean);
    const entries = paragraphEntries.length > 1 ? paragraphEntries : normalized.split(/\n+/).map((part) => part.trim()).filter(Boolean);
    return entries.map((text) => ({ text, tags: {} }));
}
export function parseCorpusFile(inputPath, options) {
    const content = readFileSync(inputPath, "utf8");
    const ext = extname(inputPath).toLowerCase();
    const format = (options?.format ?? "auto").toLowerCase();
    if (format === "jsonl" || (format === "auto" && ext === ".jsonl"))
        return parseJsonlCorpus(content, options?.textColumn);
    if (format === "json" || (format === "auto" && ext === ".json"))
        return parseJsonCorpus(content, options?.textColumn);
    if (format === "csv" || (format === "auto" && ext === ".csv"))
        return parseDelimited(content, ",", options?.textColumn);
    if (format === "tsv" || (format === "auto" && ext === ".tsv"))
        return parseDelimited(content, "\t", options?.textColumn);
    return parseTextCorpus(content);
}
export function buildBm25Index(entries) {
    const k1 = 1.5;
    const b = 0.75;
    const docLengths = {};
    const docFreq = {};
    const termFreq = {};
    let totalLength = 0;
    for (const entry of entries) {
        const tokens = tokenize(entry.text);
        docLengths[entry.id] = tokens.length;
        totalLength += tokens.length;
        const tf = {};
        for (const token of tokens)
            tf[token] = (tf[token] ?? 0) + 1;
        termFreq[entry.id] = tf;
        for (const token of new Set(tokens))
            docFreq[token] = (docFreq[token] ?? 0) + 1;
    }
    return {
        k1,
        b,
        avgDocLength: entries.length > 0 ? totalLength / entries.length : 0,
        docCount: entries.length,
        docLengths,
        docFreq,
        termFreq,
    };
}
export function bm25Score(index, entryId, query) {
    const terms = tokenize(query);
    if (terms.length === 0 || index.docCount === 0)
        return 0;
    const tf = index.termFreq[entryId] ?? {};
    const dl = index.docLengths[entryId] ?? 0;
    const avgdl = index.avgDocLength || 1;
    let score = 0;
    for (const term of terms) {
        const f = tf[term] ?? 0;
        if (f <= 0)
            continue;
        const n = index.docFreq[term] ?? 0;
        const idf = Math.log(1 + (index.docCount - n + 0.5) / (n + 0.5));
        const denom = f + index.k1 * (1 - index.b + index.b * (dl / avgdl));
        score += idf * ((f * (index.k1 + 1)) / denom);
    }
    return score;
}
function l2Normalize(vector) {
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    if (!Number.isFinite(norm) || norm === 0)
        return vector;
    return vector.map((value) => value / norm);
}
export function cosineSimilarity(a, b) {
    const n = Math.min(a.length, b.length);
    let sum = 0;
    for (let i = 0; i < n; i++)
        sum += (a[i] ?? 0) * (b[i] ?? 0);
    return sum;
}
async function embedBatch(model, inputs) {
    const response = await fetch("http://127.0.0.1:11434/api/embed", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, input: inputs }),
    });
    if (!response.ok) {
        const text = await response.text().catch(() => "");
        throw new Error(`Ollama embed failed (${response.status}): ${text || response.statusText}`);
    }
    const data = (await response.json());
    if (Array.isArray(data.embeddings))
        return data.embeddings.map(l2Normalize);
    if (Array.isArray(data.embedding))
        return [l2Normalize(data.embedding)];
    throw new Error("Ollama embed response did not include embeddings");
}
export async function embedTexts(texts, options) {
    const model = options?.model ?? DEFAULT_EMBEDDING_MODEL;
    const batchSize = options?.batchSize ?? DEFAULT_BATCH_SIZE;
    const vectors = {};
    let dimensions = 0;
    for (let start = 0; start < texts.length; start += batchSize) {
        const batch = texts.slice(start, start + batchSize);
        options?.onProgress?.(`Embedding entries ${start + 1}-${start + batch.length} of ${texts.length}`);
        const embeddings = await embedBatch(model, batch);
        for (let i = 0; i < embeddings.length; i++) {
            vectors[String(start + i)] = embeddings[i];
            dimensions = Math.max(dimensions, embeddings[i].length);
        }
    }
    return { model, dimensions, vectors };
}
export function datasetExists(cwd, name) {
    const slug = slugifyDatasetName(name);
    return existsSync(join(datasetDir(cwd, slug), MANIFEST_FILE));
}
export function listDatasets(cwd) {
    ensureQuailWorkspace(cwd);
    const rootManifestPath = join(getQuailDatasetsDir(cwd), ROOT_MANIFEST_FILE);
    if (!existsSync(rootManifestPath))
        return [];
    const manifest = readJson(rootManifestPath);
    return [...(manifest.datasets ?? [])].sort((a, b) => a.name.localeCompare(b.name));
}
function writeRootManifest(cwd) {
    const datasetsDir = getQuailDatasetsDir(cwd);
    const datasets = [];
    if (!existsSync(datasetsDir))
        mkdirSync(datasetsDir, { recursive: true });
    for (const name of readdirSync(datasetsDir)) {
        const path = join(datasetsDir, name, MANIFEST_FILE);
        if (!existsSync(path))
            continue;
        const manifest = readJson(path);
        datasets.push({
            name: manifest.name,
            slug: manifest.slug,
            entryCount: manifest.entryCount,
            embeddingModel: manifest.embeddingModel,
            metadataFields: manifest.metadataFields,
            createdAt: manifest.createdAt,
        });
    }
    writeJson(join(datasetsDir, ROOT_MANIFEST_FILE), { version: 1, datasets });
}
export function removeDataset(cwd, name) {
    ensureQuailWorkspace(cwd);
    const slug = slugifyDatasetName(name);
    const dir = datasetDir(cwd, slug);
    if (!existsSync(dir))
        return false;
    rmSync(dir, { recursive: true, force: true });
    writeRootManifest(cwd);
    return true;
}
export function loadDataset(cwd, name) {
    const slug = slugifyDatasetName(name);
    const dir = datasetDir(cwd, slug);
    if (!existsSync(join(dir, MANIFEST_FILE))) {
        throw new Error(`Dataset "${name}" is not processed in this Quail workspace`);
    }
    const manifest = readJson(join(dir, MANIFEST_FILE));
    const entries = readFileSync(join(dir, manifest.files.entries), "utf8")
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => JSON.parse(line));
    const bm25 = readJson(join(dir, manifest.files.bm25));
    const embeddings = loadEmbeddingIndex(dir, manifest.files.embeddings);
    return { manifest, entries, bm25, embeddings };
}
export function loadDatasets(cwd, names) {
    return names.map((name) => loadDataset(cwd, name));
}
export async function processDataset(options) {
    const cwd = options.cwd;
    const name = options.name.trim();
    if (!name)
        throw new Error("Dataset name is required");
    ensureQuailWorkspace(cwd);
    const slug = slugifyDatasetName(name);
    const dir = datasetDir(cwd, slug);
    if (existsSync(join(dir, MANIFEST_FILE)) && !options.overwrite) {
        throw new Error(`Dataset name must be unique. "${name}" already exists.`);
    }
    options.onProgress?.("[1/6] Reading and structuring corpus");
    const parsed = parseCorpusFile(options.inputPath, { format: options.format, textColumn: options.textColumn });
    if (parsed.length === 0)
        throw new Error("No entries found in corpus");
    const globalTags = options.globalTags ?? {};
    const entries = parsed.map((entry, index) => ({
        id: stableEntryId(slug, index),
        dataset: name,
        ordinal: index + 1,
        text: entry.text,
        tags: { ...entry.tags, ...globalTags },
        contains: normalizeContainsText(entry.text),
    }));
    const metadataFields = Array.from(new Set(entries.flatMap((entry) => Object.keys(entry.tags)))).sort();
    options.onProgress?.("[2/6] Building BM25 preprocessing index");
    const bm25 = buildBm25Index(entries);
    options.onProgress?.("[3/6] Preparing exact contains search text");
    // Exact string search uses the per-entry normalized `contains` field written below.
    const embeddingModel = options.model ?? DEFAULT_EMBEDDING_MODEL;
    const batchSize = options.batchSize ?? DEFAULT_BATCH_SIZE;
    let embeddingIndex;
    if (options.skipEmbeddings) {
        options.onProgress?.("[4/6] Skipping embeddings by request");
        embeddingIndex = { model: embeddingModel, dimensions: 0, vectors: {} };
    }
    else {
        options.onProgress?.(`[4/6] Embedding with Ollama model ${embeddingModel} at batch size ${batchSize}`);
        const byPosition = await embedTexts(entries.map((entry) => entry.text), { model: embeddingModel, batchSize, onProgress: options.onProgress });
        embeddingIndex = {
            model: byPosition.model,
            dimensions: byPosition.dimensions,
            vectors: Object.fromEntries(entries.map((entry, index) => [entry.id, byPosition.vectors[String(index)] ?? []])),
        };
    }
    options.onProgress?.("[5/6] Writing dataset files into workspace/datasets");
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, ENTRIES_FILE), `${entries.map((entry) => JSON.stringify(entry)).join("\n")}\n`, "utf8");
    writeJson(join(dir, BM25_FILE), bm25);
    writeEmbeddingIndex(dir, EMBEDDINGS_FILE, embeddingIndex);
    const now = new Date().toISOString();
    const manifest = {
        name,
        slug,
        createdAt: now,
        updatedAt: now,
        entryCount: entries.length,
        metadataFields,
        embeddingModel,
        embeddingDimensions: embeddingIndex.dimensions,
        batchSize,
        source: { fileName: basename(options.inputPath), format: options.format ?? "auto" },
        files: { entries: ENTRIES_FILE, bm25: BM25_FILE, embeddings: EMBEDDINGS_FILE },
    };
    writeJson(join(dir, MANIFEST_FILE), manifest);
    options.onProgress?.("[6/6] Updating dataset registry");
    writeRootManifest(cwd);
    return manifest;
}
//# sourceMappingURL=dataset-store.js.map