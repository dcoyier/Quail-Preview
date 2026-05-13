import { closeSync, existsSync, mkdirSync, openSync, readdirSync, readFileSync, readSync, rmSync, statSync, writeFileSync, writeSync, } from "node:fs";
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
const MAX_EAGER_VECTOR_FILE_BYTES = 1.5 * 1024 * 1024 * 1024;
const FIELD_DOC_SEPARATOR = "\u0000";
const LEGACY_TEXT_FIELD = "text";
const loadedDatasetCache = new Map();
function datasetDir(cwd, slug) {
    return join(getQuailDatasetsDir(cwd), slug);
}
function readJson(path) {
    return JSON.parse(readFileSync(path, "utf8"));
}
function writeJson(path, value) {
    writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}
function getMtimeMs(path) {
    return statSync(path).mtimeMs;
}
export function fieldDocumentId(entryId, field) {
    return `${entryId}${FIELD_DOC_SEPARATOR}${field}`;
}
export function fieldValueToText(value) {
    if (value === undefined || value === null)
        return "";
    if (typeof value === "string")
        return value;
    if (typeof value === "number" || typeof value === "boolean")
        return String(value);
    if (Array.isArray(value))
        return value.map(fieldValueToText).filter(Boolean).join(" ");
    if (typeof value === "object")
        return JSON.stringify(value);
    return String(value);
}
function isBinaryEmbeddingIndexFile(value) {
    if (typeof value !== "object" || value === null)
        return false;
    const record = value;
    return record.format === "float32-binary-v1" && typeof record.model === "string";
}
function hasDotProduct(vector) {
    return typeof vector.dotProduct === "function";
}
class FileBackedEmbeddingVector {
    store;
    index;
    length;
    constructor(store, index, dimensions) {
        this.store = store;
        this.index = index;
        this.length = dimensions;
    }
    dotProduct(other) {
        if (other instanceof FileBackedEmbeddingVector)
            return this.store.dotProduct(this.index, other.toFloat32Array());
        return this.store.dotProduct(this.index, other);
    }
    toFloat32Array() {
        return this.store.readVector(this.index);
    }
}
class FileBackedEmbeddingStore {
    fd;
    dimensions;
    scratch;
    constructor(fd, dimensions) {
        this.fd = fd;
        this.dimensions = dimensions;
        this.scratch = Buffer.allocUnsafe(dimensions * 4);
    }
    dotProduct(index, other) {
        readSync(this.fd, this.scratch, 0, this.scratch.byteLength, index * this.dimensions * 4);
        const n = Math.min(other.length, this.dimensions);
        let sum = 0;
        for (let i = 0; i < n; i++)
            sum += (other[i] ?? 0) * this.scratch.readFloatLE(i * 4);
        return sum;
    }
    readVector(index) {
        const buffer = Buffer.allocUnsafe(this.dimensions * 4);
        readSync(this.fd, buffer, 0, buffer.byteLength, index * this.dimensions * 4);
        return new Float32Array(buffer.buffer, buffer.byteOffset, this.dimensions);
    }
    vector(index) {
        return new FileBackedEmbeddingVector(this, index, this.dimensions);
    }
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
    const vectorPath = join(dir, raw.vectorsFile || EMBEDDINGS_VECTOR_FILE);
    const vectorSize = statSync(vectorPath).size;
    const vectors = {};
    const bytesPerVector = dimensions * 4;
    if (dimensions > 0 && vectorSize > MAX_EAGER_VECTOR_FILE_BYTES) {
        const fd = openSync(vectorPath, "r");
        const store = new FileBackedEmbeddingStore(fd, dimensions);
        for (let i = 0; i < ids.length; i++) {
            const byteOffset = i * bytesPerVector;
            vectors[ids[i]] = byteOffset + bytesPerVector <= vectorSize ? store.vector(i) : [];
        }
        return { model: raw.model, dimensions, vectors };
    }
    const data = readFileSync(vectorPath);
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
function splitDelimitedRecords(content, delimiter) {
    const rows = [];
    let row = [];
    let current = "";
    let quoted = false;
    const input = content.replace(/^\uFEFF/, "");
    for (let i = 0; i < input.length; i++) {
        const ch = input[i];
        if (ch === '"') {
            if (quoted && input[i + 1] === '"') {
                current += '"';
                i++;
            }
            else {
                quoted = !quoted;
            }
            continue;
        }
        if (!quoted && ch === delimiter) {
            row.push(current.trim());
            current = "";
            continue;
        }
        if (!quoted && (ch === "\n" || ch === "\r")) {
            row.push(current.trim());
            current = "";
            if (row.some((cell) => cell.length > 0))
                rows.push(row);
            row = [];
            if (ch === "\r" && input[i + 1] === "\n")
                i++;
            continue;
        }
        current += ch;
    }
    row.push(current.trim());
    if (row.some((cell) => cell.length > 0))
        rows.push(row);
    return rows;
}
function inferFieldValue(value) {
    const trimmed = value.trim();
    if (!trimmed)
        return "";
    if (/^(true|false)$/i.test(trimmed))
        return trimmed.toLowerCase() === "true";
    if (/^-?\d+$/.test(trimmed)) {
        const parsed = Number(trimmed);
        if (Number.isSafeInteger(parsed))
            return parsed;
    }
    if (/^-?(?:\d+\.\d+|\d+\.|\.\d+)(?:e[+-]?\d+)?$/i.test(trimmed) || /^-?\d+e[+-]?\d+$/i.test(trimmed)) {
        const parsed = Number(trimmed);
        if (Number.isFinite(parsed))
            return parsed;
    }
    return value.trim();
}
function normalizeFieldType(type) {
    const normalized = String(type).trim().toLowerCase();
    if (normalized === "boolean")
        return "bool";
    if (normalized === "string" ||
        normalized === "int" ||
        normalized === "float" ||
        normalized === "bool" ||
        normalized === "null" ||
        normalized === "list" ||
        normalized === "object" ||
        normalized === "mixed") {
        return normalized;
    }
    throw new Error(`Unknown field type "${type}". Expected string, int, float, bool, null, list, object, or mixed.`);
}
function inferStringType(value) {
    const trimmed = value.trim();
    if (!trimmed)
        return "null";
    if (/^(true|false)$/i.test(trimmed))
        return "bool";
    if (/^-?(?:0|[1-9]\d*)$/.test(trimmed))
        return "int";
    if (/^-?(?:\d+\.\d+|\d+\.|\.\d+)(?:e[+-]?\d+)?$/i.test(trimmed) || /^-?(?:0|[1-9]\d*)e[+-]?\d+$/i.test(trimmed))
        return "float";
    return "string";
}
function inferValueType(value) {
    if (value === null)
        return "null";
    if (typeof value === "string")
        return inferStringType(value);
    if (typeof value === "boolean")
        return "bool";
    if (typeof value === "number")
        return Number.isInteger(value) ? "int" : "float";
    if (Array.isArray(value))
        return "list";
    if (typeof value === "object")
        return "object";
    return "string";
}
function combineFieldTypes(types) {
    const observed = [...new Set([...types].filter((type) => type !== "null"))];
    if (observed.length === 0)
        return "null";
    if (observed.length === 1)
        return observed[0];
    if (observed.includes("mixed"))
        return "mixed";
    if (observed.every((type) => type === "int" || type === "float"))
        return "float";
    if (observed.includes("string") && observed.every((type) => type === "string" || type === "int" || type === "float" || type === "bool"))
        return "string";
    return "mixed";
}
function inferFieldTypes(entries, overrides) {
    const observed = new Map();
    for (const entry of entries) {
        for (const [field, value] of Object.entries(entry.fields)) {
            const types = observed.get(field) ?? [];
            types.push(inferValueType(value));
            observed.set(field, types);
        }
    }
    const fields = [...observed.keys()].sort();
    const normalizedOverrides = Object.fromEntries(Object.entries(overrides ?? {}).map(([field, type]) => [field, normalizeFieldType(type)]));
    const fieldTypes = {};
    for (const field of fields)
        fieldTypes[field] = normalizedOverrides[field] ?? combineFieldTypes(observed.get(field) ?? []);
    for (const field of Object.keys(normalizedOverrides)) {
        if (!(field in fieldTypes))
            fieldTypes[field] = normalizedOverrides[field];
    }
    return fieldTypes;
}
function coerceFieldValue(field, value, type) {
    if (value === null)
        return null;
    if (type === "mixed")
        return value;
    if (type === "string")
        return typeof value === "string" ? value.trim() : fieldValueToText(value);
    if (type === "list") {
        if (Array.isArray(value))
            return value;
        throw new Error(`Field "${field}" was set to list, but value ${JSON.stringify(value)} is not a list.`);
    }
    if (type === "object") {
        if (typeof value === "object" && !Array.isArray(value))
            return value;
        throw new Error(`Field "${field}" was set to object, but value ${JSON.stringify(value)} is not an object.`);
    }
    if (type === "bool") {
        if (typeof value === "boolean")
            return value;
        if (typeof value === "string" && /^(true|false)$/i.test(value.trim()))
            return value.trim().toLowerCase() === "true";
        throw new Error(`Field "${field}" was set to bool, but value ${JSON.stringify(value)} cannot be converted to bool.`);
    }
    const numericValue = typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value.trim()) : Number.NaN;
    if (!Number.isFinite(numericValue)) {
        throw new Error(`Field "${field}" was set to ${type}, but value ${JSON.stringify(value)} cannot be converted to ${type}.`);
    }
    if (type === "int") {
        if (!Number.isSafeInteger(numericValue))
            throw new Error(`Field "${field}" was set to int, but value ${JSON.stringify(value)} is not a safe integer.`);
        return numericValue;
    }
    if (type === "float")
        return numericValue;
    return null;
}
function coerceFields(fields, fieldTypes) {
    return Object.fromEntries(Object.entries(fields).map(([field, value]) => [field, coerceFieldValue(field, value, fieldTypes[field] ?? inferValueType(value))]));
}
function getStringFieldNames(fieldTypes) {
    return Object.entries(fieldTypes)
        .filter(([, type]) => type === "string")
        .map(([field]) => field)
        .sort();
}
function withGlobalFields(entries, globalFields) {
    return entries.map((entry) => ({
        ...entry,
        fields: {
            ...entry.fields,
            ...globalFields,
        },
    }));
}
function sampleFieldValues(entries, field, type) {
    const samples = [];
    const seen = new Set();
    for (const entry of entries) {
        if (!Object.prototype.hasOwnProperty.call(entry.fields, field))
            continue;
        const value = coerceFieldValue(field, entry.fields[field], type);
        const text = fieldValueToText(value).trim();
        if (!text || seen.has(text))
            continue;
        seen.add(text);
        samples.push(text.slice(0, 160));
        if (samples.length >= 3)
            break;
    }
    return samples;
}
function inspectParsedCorpus(entries, options) {
    const prepared = withGlobalFields(entries, options?.globalFields ?? {});
    const fieldTypes = inferFieldTypes(prepared, options?.fieldTypes);
    const embeddedFields = getStringFieldNames(fieldTypes);
    const fields = Object.keys(fieldTypes).sort().map((field) => {
        const type = fieldTypes[field];
        return {
            name: field,
            type,
            embedded: type === "string",
            nonEmptyCount: prepared.filter((entry) => fieldValueToText(entry.fields[field]).trim().length > 0).length,
            samples: sampleFieldValues(prepared, field, type),
        };
    });
    return { entryCount: prepared.length, fieldTypes, embeddedFields, fields };
}
function getPreferredTextField(headers, textColumn) {
    const lowerHeaders = headers.map((h) => h.toLowerCase());
    if (textColumn) {
        const index = lowerHeaders.indexOf(textColumn.toLowerCase());
        return index >= 0 ? headers[index] : undefined;
    }
    const preferredNames = ["text", "response", "content", "answer", "comment", "body"];
    const index = preferredNames.map((name) => lowerHeaders.indexOf(name)).find((candidate) => candidate >= 0) ?? -1;
    return index >= 0 ? headers[index] : undefined;
}
function inferTextFields(fields, preferred) {
    if (preferred && fieldValueToText(fields[preferred]).trim())
        return [preferred];
    return Object.entries(fields)
        .filter(([, value]) => typeof value === "string" && value.trim().length > 0)
        .map(([field]) => field);
}
function buildDefaultText(fields, textFields) {
    return textFields.map((field) => fieldValueToText(fields[field]).trim()).filter(Boolean).join("\n\n");
}
function parseDelimited(content, delimiter, textColumn) {
    const rows = splitDelimitedRecords(content, delimiter);
    if (rows.length === 0)
        return [];
    const headers = rows[0].map((h) => h.trim());
    const preferredTextField = getPreferredTextField(headers, textColumn);
    return rows.slice(1).flatMap((cells) => {
        const fields = {};
        for (let i = 0; i < headers.length; i++) {
            const value = (cells[i] ?? "").trim();
            if (value)
                fields[headers[i]] = value;
        }
        if (Object.keys(fields).length === 0)
            return [];
        const textFields = inferTextFields(fields, preferredTextField);
        return [{ text: buildDefaultText(fields, textFields), fields, tags: {}, textFields }];
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
    const toFieldValue = (value) => {
        if (value === undefined || value === null)
            return;
        if (typeof value === "string" || typeof value === "number" || typeof value === "boolean")
            return value;
        if (Array.isArray(value))
            return value.flatMap((item) => {
                const converted = toFieldValue(item);
                return converted === undefined ? [] : [converted];
            });
        if (typeof value === "object") {
            const out = {};
            for (const [key, nestedValue] of Object.entries(value)) {
                const converted = toFieldValue(nestedValue);
                if (converted !== undefined)
                    out[key] = converted;
            }
            return out;
        }
        return String(value);
    };
    const addFieldFromValue = (fields, key, value, options) => {
        if (typeof value === "object" && value !== null && !Array.isArray(value)) {
            for (const [nestedKey, nestedValue] of Object.entries(value)) {
                const converted = toFieldValue(nestedValue);
                if (converted !== undefined)
                    fields[options?.flatten ? nestedKey : `${key}.${nestedKey}`] = converted;
            }
            return;
        }
        const converted = toFieldValue(value);
        if (converted !== undefined)
            fields[key] = converted;
    };
    return array.flatMap((item) => {
        if (typeof item === "string") {
            const text = item.trim();
            return text ? [{ text, fields: { [LEGACY_TEXT_FIELD]: text }, tags: {}, textFields: [LEGACY_TEXT_FIELD] }] : [];
        }
        if (typeof item !== "object" || item === null)
            return [];
        const record = item;
        const textKey = preferred.find((key) => typeof record[key] === "string" && String(record[key]).trim());
        const fields = {};
        for (const [key, value] of Object.entries(record)) {
            if (value === undefined || value === null)
                continue;
            addFieldFromValue(fields, key, value, { flatten: key === "metadata" });
        }
        if (Object.keys(fields).length === 0)
            return [];
        const textFields = inferTextFields(fields, textKey);
        return [{ text: buildDefaultText(fields, textFields), fields, tags: {}, textFields }];
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
    return entries.map((text) => ({ text, fields: { [LEGACY_TEXT_FIELD]: text }, tags: {}, textFields: [LEGACY_TEXT_FIELD] }));
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
export function buildBm25Index(entries, fieldTypes) {
    const k1 = 1.5;
    const b = 0.75;
    const docLengths = {};
    const docFreq = {};
    const termFreq = {};
    let totalLength = 0;
    for (const [docId, text] of getBm25Documents(entries, fieldTypes)) {
        const tokens = tokenize(text);
        docLengths[docId] = tokens.length;
        totalLength += tokens.length;
        const tf = {};
        for (const token of tokens)
            tf[token] = (tf[token] ?? 0) + 1;
        termFreq[docId] = tf;
        for (const token of new Set(tokens))
            docFreq[token] = (docFreq[token] ?? 0) + 1;
    }
    const docCount = Object.keys(docLengths).length;
    return {
        k1,
        b,
        avgDocLength: docCount > 0 ? totalLength / docCount : 0,
        docCount,
        docLengths,
        docFreq,
        termFreq,
    };
}
function getBm25Documents(entries, fieldTypes) {
    return entries.flatMap((entry) => [
        [entry.id, entry.text],
        ...getStringFieldDocuments([entry], fieldTypes),
    ]);
}
function getStringFieldDocuments(entries, fieldTypes) {
    const stringFields = fieldTypes ? new Set(getStringFieldNames(fieldTypes)) : undefined;
    return entries.flatMap((entry) => {
        const docs = [];
        for (const [field, value] of Object.entries(entry.fields)) {
            if (stringFields && !stringFields.has(field))
                continue;
            if (!stringFields && typeof value !== "string")
                continue;
            const text = typeof value === "string" ? value : fieldValueToText(value);
            if (text.trim())
                docs.push([fieldDocumentId(entry.id, field), text]);
        }
        return docs;
    });
}
function getEmbeddingDocuments(entries, fieldTypes) {
    return getStringFieldDocuments(entries, fieldTypes).filter(([, text]) => text.trim().length > 0);
}
export function bm25Score(index, entryId, query) {
    return bm25ScoreTerms(index, entryId, tokenize(query));
}
export function bm25ScoreTerms(index, entryId, terms) {
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
    if (hasDotProduct(b))
        return b.dotProduct(a);
    if (hasDotProduct(a))
        return a.dotProduct(b);
    const n = Math.min(a.length, b.length);
    let sum = 0;
    for (let i = 0; i < n; i++)
        sum += (a[i] ?? 0) * (b[i] ?? 0);
    return sum;
}
async function embedBatch(model, inputs) {
    const embedUrl = process.env.QUAIL_OLLAMA_EMBED_URL ?? "http://127.0.0.1:11434/api/embed";
    const response = await fetch(embedUrl, {
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
    loadedDatasetCache.delete(`${cwd}\0${slug}`);
    writeRootManifest(cwd);
    return true;
}
function buildFieldContains(fields, fieldTypes) {
    return Object.fromEntries(Object.entries(fields)
        .filter(([field, value]) => fieldTypes[field] === "string" && typeof value === "string")
        .map(([field, value]) => [field, normalizeContainsText(value)]));
}
function normalizeLoadedEntry(raw, manifest) {
    const rawRecord = raw;
    const fields = rawRecord.fields && typeof rawRecord.fields === "object"
        ? { ...rawRecord.fields }
        : {
            ...(raw.text ? { [LEGACY_TEXT_FIELD]: raw.text } : {}),
            ...Object.fromEntries(Object.entries(raw.tags ?? {}).map(([field, value]) => [field, inferFieldValue(String(value))])),
        };
    const fieldTypes = manifest?.fieldTypes ?? inferFieldTypes([{ text: raw.text ?? "", fields, tags: raw.tags ?? {} }]);
    const textFields = getStringFieldNames(fieldTypes);
    const text = raw.text || buildDefaultText(fields, textFields);
    return {
        ...raw,
        text,
        fields,
        tags: raw.tags ?? {},
        contains: raw.contains ?? normalizeContainsText(text),
        fieldContains: rawRecord.fieldContains ?? buildFieldContains(fields, fieldTypes),
    };
}
function getFieldNames(entries) {
    return Array.from(new Set(entries.flatMap((entry) => Object.keys(entry.fields)))).sort();
}
function uniqueStrings(values) {
    return [...new Set(values)];
}
function getTextFieldNames(entries, fieldTypes) {
    if (fieldTypes)
        return getStringFieldNames(fieldTypes);
    return getFieldNames(entries).filter((field) => entries.some((entry) => typeof entry.fields[field] === "string" && String(entry.fields[field]).trim().length > 0));
}
export function loadDataset(cwd, name) {
    const slug = slugifyDatasetName(name);
    const dir = datasetDir(cwd, slug);
    const manifestPath = join(dir, MANIFEST_FILE);
    if (!existsSync(manifestPath)) {
        throw new Error(`Dataset "${name}" is not processed in this Quail workspace`);
    }
    const manifest = readJson(manifestPath);
    const entriesPath = join(dir, manifest.files.entries);
    const bm25Path = join(dir, manifest.files.bm25);
    const embeddingsPath = join(dir, manifest.files.embeddings);
    const vectorPath = join(dir, EMBEDDINGS_VECTOR_FILE);
    const cacheKey = `${cwd}\0${slug}`;
    const manifestMtimeMs = getMtimeMs(manifestPath);
    const entriesMtimeMs = getMtimeMs(entriesPath);
    const bm25MtimeMs = getMtimeMs(bm25Path);
    const embeddingsMtimeMs = getMtimeMs(embeddingsPath);
    const vectorMtimeMs = existsSync(vectorPath) ? getMtimeMs(vectorPath) : undefined;
    const cached = loadedDatasetCache.get(cacheKey);
    if (cached &&
        cached.manifestMtimeMs === manifestMtimeMs &&
        cached.entriesMtimeMs === entriesMtimeMs &&
        cached.bm25MtimeMs === bm25MtimeMs &&
        cached.embeddingsMtimeMs === embeddingsMtimeMs &&
        cached.vectorMtimeMs === vectorMtimeMs) {
        return cached.dataset;
    }
    const entries = readFileSync(entriesPath, "utf8")
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => normalizeLoadedEntry(JSON.parse(line), manifest));
    manifest.fieldNames ??= getFieldNames(entries);
    manifest.fieldTypes ??= inferFieldTypes(entries.map((entry) => ({ text: entry.text, fields: entry.fields, tags: entry.tags })));
    manifest.embeddedFields ??= getStringFieldNames(manifest.fieldTypes);
    manifest.textFields ??= manifest.embeddedFields;
    for (const entry of entries) {
        const stringFields = manifest.embeddedFields ?? [];
        entry.text = entry.text || buildDefaultText(entry.fields, stringFields);
        entry.contains = normalizeContainsText(entry.text);
        entry.fieldContains = buildFieldContains(entry.fields, manifest.fieldTypes);
    }
    const bm25 = readJson(bm25Path);
    const embeddings = loadEmbeddingIndex(dir, manifest.files.embeddings);
    const dataset = { manifest, entries, bm25, embeddings };
    loadedDatasetCache.set(cacheKey, {
        manifestMtimeMs,
        entriesMtimeMs,
        bm25MtimeMs,
        embeddingsMtimeMs,
        vectorMtimeMs,
        dataset,
    });
    return dataset;
}
export function loadDatasets(cwd, names) {
    return names.map((name) => loadDataset(cwd, name));
}
export function inspectDatasetFile(options) {
    const parsed = parseCorpusFile(options.inputPath, { format: options.format, textColumn: options.textColumn });
    return inspectParsedCorpus(parsed, { globalFields: options.globalTags, fieldTypes: options.fieldTypes });
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
    loadedDatasetCache.delete(`${cwd}\0${slug}`);
    options.onProgress?.("[1/6] Reading and structuring corpus");
    const parsed = parseCorpusFile(options.inputPath, { format: options.format, textColumn: options.textColumn });
    if (parsed.length === 0)
        throw new Error("No entries found in corpus");
    const globalTags = options.globalTags ?? {};
    const prepared = withGlobalFields(parsed, globalTags);
    const inspection = inspectParsedCorpus(prepared, { fieldTypes: options.fieldTypes });
    const fieldTypes = inspection.fieldTypes;
    const embeddedFields = inspection.embeddedFields;
    const entries = parsed.map((entry, index) => ({
        ...(() => {
            const rawFields = {
                ...entry.fields,
                ...globalTags,
            };
            const fields = coerceFields(rawFields, fieldTypes);
            const text = buildDefaultText(fields, embeddedFields);
            return {
                id: stableEntryId(slug, index),
                dataset: name,
                ordinal: index + 1,
                text,
                fields,
                tags: { ...entry.tags },
                contains: normalizeContainsText(text),
                fieldContains: buildFieldContains(fields, fieldTypes),
            };
        })(),
    }));
    const metadataFields = Array.from(new Set(entries.flatMap((entry) => Object.keys(entry.fields)))).sort();
    const textFields = getTextFieldNames(entries, fieldTypes);
    options.onProgress?.("[2/6] Building BM25 preprocessing index");
    const bm25 = buildBm25Index(entries, fieldTypes);
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
        const embeddingDocs = getEmbeddingDocuments(entries, fieldTypes);
        const byPosition = await embedTexts(embeddingDocs.map(([, text]) => text), { model: embeddingModel, batchSize, onProgress: options.onProgress });
        embeddingIndex = {
            model: byPosition.model,
            dimensions: byPosition.dimensions,
            vectors: Object.fromEntries(embeddingDocs.map(([docId], index) => [docId, byPosition.vectors[String(index)] ?? []])),
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
        fieldNames: metadataFields,
        textFields,
        fieldTypes,
        embeddedFields,
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