import { type QuailAnalysisState } from "./analysis-state.js";
export interface QuailCallBlock {
    datasets: string[];
    code: string;
    raw: string;
}
export interface QuailExecutionResult {
    state: QuailAnalysisState;
    output: string;
    errors: QuailDslError[];
    blocks: number;
}
export interface QuailDslError {
    code: string;
    message: string;
    line?: number;
}
export interface DslRuntimeCacheStats {
    datasetContexts: number;
    scoreVectorEntries: number;
    thresholdIdSetEntries: number;
    fieldComparisonEntries: number;
    textFilterEntries: number;
    queryEmbeddingEntries: number;
    scoreVectorHits: number;
    scoreVectorMisses: number;
    thresholdIdSetHits: number;
    thresholdIdSetMisses: number;
    fieldComparisonHits: number;
    fieldComparisonMisses: number;
    textFilterHits: number;
    textFilterMisses: number;
    queryEmbeddingHits: number;
    queryEmbeddingMisses: number;
}
export declare function clearQuailDslRuntimeCaches(): void;
export declare function getQuailDslRuntimeCacheStats(): DslRuntimeCacheStats;
export declare function extractQuailCallBlocks(text: string): QuailCallBlock[];
export declare function formatQuailExecutionResult(result: QuailExecutionResult): string;
export declare function executeQuailCallBlocks(options: {
    cwd: string;
    state: QuailAnalysisState;
    blocks: QuailCallBlock[];
}): Promise<QuailExecutionResult>;
