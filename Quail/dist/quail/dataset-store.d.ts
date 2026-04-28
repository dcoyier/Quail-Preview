export interface CorpusEntryInput {
    text: string;
    tags: Record<string, string>;
}
export interface QuailEntry {
    id: string;
    dataset: string;
    ordinal: number;
    text: string;
    tags: Record<string, string>;
    contains: string;
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
    globalTags?: Record<string, string>;
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
export declare function parseCorpusFile(inputPath: string, options?: {
    format?: string;
    textColumn?: string;
}): CorpusEntryInput[];
export declare function buildBm25Index(entries: QuailEntry[]): Bm25Index;
export declare function bm25Score(index: Bm25Index, entryId: string, query: string): number;
export declare function cosineSimilarity(a: EmbeddingVector, b: EmbeddingVector): number;
export declare function embedTexts(texts: string[], options?: {
    model?: string;
    batchSize?: number;
    onProgress?: (message: string) => void;
}): Promise<EmbeddingIndex>;
export declare function datasetExists(cwd: string, name: string): boolean;
export declare function listDatasets(cwd: string): DatasetListItem[];
export declare function removeDataset(cwd: string, name: string): boolean;
export declare function loadDataset(cwd: string, name: string): LoadedQuailDataset;
export declare function loadDatasets(cwd: string, names: string[]): LoadedQuailDataset[];
export declare function processDataset(options: ProcessDatasetOptions): Promise<QuailDatasetManifest>;
