export interface CorpusEntryInput {
    text: string;
    fields: Record<string, FieldValue>;
    tags: Record<string, string>;
    textFields?: string[];
}
export type FieldValue = string | number | boolean | null | FieldValue[] | {
    [key: string]: FieldValue;
};
export type FieldType = "string" | "int" | "float" | "bool" | "null" | "list" | "object" | "mixed";
export type FieldTypeOverride = FieldType | "boolean";
export interface FieldInspection {
    name: string;
    type: FieldType;
    embedded: boolean;
    nonEmptyCount: number;
    samples: string[];
}
export interface DatasetInspection {
    entryCount: number;
    fieldTypes: Record<string, FieldType>;
    embeddedFields: string[];
    fields: FieldInspection[];
}
export interface QuailEntry {
    id: string;
    dataset: string;
    ordinal: number;
    text: string;
    fields: Record<string, FieldValue>;
    tags: Record<string, string>;
    contains: string;
    fieldContains: Record<string, string>;
}
export interface Bm25Index {
    k1: number;
    b: number;
    avgDocLength: number;
    docCount: number;
    docLengths: Record<string, number>;
    docFreq: Record<string, number>;
    termFreq: Record<string, Record<string, number>>;
}
export type EmbeddingVector = ArrayLike<number>;
export interface EmbeddingIndex {
    model: string;
    dimensions: number;
    vectors: Record<string, EmbeddingVector>;
}
export interface QuailDatasetManifest {
    name: string;
    slug: string;
    createdAt: string;
    updatedAt: string;
    entryCount: number;
    metadataFields: string[];
    fieldNames?: string[];
    textFields?: string[];
    fieldTypes?: Record<string, FieldType>;
    embeddedFields?: string[];
    embeddingModel: string;
    embeddingDimensions: number;
    batchSize: number;
    source: {
        fileName?: string;
        format: string;
    };
    files: {
        entries: string;
        bm25: string;
        embeddings: string;
    };
}
export interface LoadedQuailDataset {
    manifest: QuailDatasetManifest;
    workspaceDir: string;
    datasetDir: string;
    entries: QuailEntry[];
    bm25: Bm25Index;
    embeddings: EmbeddingIndex;
}
export interface ProcessDatasetOptions {
    cwd: string;
    inputPath: string;
    name: string;
    format?: string;
    textColumn?: string;
    model?: string;
    batchSize?: number;
    embeddingConcurrency?: number;
    globalTags?: Record<string, string>;
    fieldTypes?: Record<string, FieldTypeOverride>;
    skipEmbeddings?: boolean;
    overwrite?: boolean;
    onProgress?: (message: string) => void;
}
export interface DatasetListItem {
    name: string;
    slug: string;
    entryCount: number;
    embeddingModel: string;
    metadataFields: string[];
    createdAt: string;
}
export declare const DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b";
export declare const DEFAULT_EMBEDDING_PROVIDER = "openrouter";
export declare const DEFAULT_BATCH_SIZE = 256;
export declare const DEFAULT_OPENROUTER_EMBEDDING_CONCURRENCY = 20;
export declare const DEFAULT_OLLAMA_EMBEDDING_CONCURRENCY = 1;
export declare const DEFAULT_EMBEDDING_MAX_RETRIES = 6;
export declare function fieldDocumentId(entryId: string, field: string): string;
export declare function fieldValueToText(value: unknown): string;
export declare function loadLocalEmbeddingEnv(cwd?: string): void;
export declare function defaultEmbeddingModel(): string;
export declare function defaultEmbeddingBatchSize(): number;
export declare function defaultEmbeddingConcurrency(): number;
export declare function embeddingBackendCacheKey(model: string): string;
export declare function embeddingBackendDescription(model?: string): string;
export declare function parseCorpusFile(inputPath: string, options?: {
    format?: string;
    textColumn?: string;
}): CorpusEntryInput[];
export declare function buildBm25Index(entries: QuailEntry[], fieldTypes?: Record<string, FieldType>): Bm25Index;
export declare function bm25Score(index: Bm25Index, entryId: string, query: string): number;
export declare function bm25ScoreTerms(index: Bm25Index, entryId: string, terms: readonly string[]): number;
export declare function cosineSimilarity(a: EmbeddingVector, b: EmbeddingVector): number;
export declare function scoreEmbeddingVectorValues(vectors: ReadonlyArray<EmbeddingVector | undefined>, query: EmbeddingVector): Float32Array;
export declare function embedTexts(texts: string[], options?: {
    model?: string;
    batchSize?: number;
    concurrency?: number;
    onProgress?: (message: string) => void;
}): Promise<EmbeddingIndex>;
export declare function datasetExists(cwd: string, name: string): boolean;
export declare function listDatasets(cwd: string): DatasetListItem[];
export declare function removeDataset(cwd: string, name: string): boolean;
export declare function loadDataset(cwd: string, name: string): LoadedQuailDataset;
export declare function loadDatasets(cwd: string, names: string[]): LoadedQuailDataset[];
export declare function inspectDatasetFile(options: {
    inputPath: string;
    format?: string;
    textColumn?: string;
    globalTags?: Record<string, string>;
    fieldTypes?: Record<string, FieldTypeOverride>;
}): DatasetInspection;
export declare function processDataset(options: ProcessDatasetOptions): Promise<QuailDatasetManifest>;
