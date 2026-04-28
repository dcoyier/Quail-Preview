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
export declare function extractQuailCallBlocks(text: string): QuailCallBlock[];
export declare function formatQuailExecutionResult(result: QuailExecutionResult): string;
export declare function executeQuailCallBlocks(options: {
    cwd: string;
    state: QuailAnalysisState;
    blocks: QuailCallBlock[];
}): Promise<QuailExecutionResult>;
