import type { SessionEntry } from "../core/session-manager.js";
export declare const QUAIL_ANALYSIS_STATE_ENTRY = "quail.analysis_state";
export declare const QUAIL_ANALYSIS_RESULT_MESSAGE = "quail.analysis_result";
export interface QuailGroupState {
    id: string;
    datasets: string[];
    spec: string;
    entryIds: string[];
    createdAt: string;
}
export interface QuailAnalysisState {
    version: 1;
    nextGroupNumber: number;
    groups: Record<string, QuailGroupState>;
    tagsByEntry: Record<string, Record<string, string | string[]>>;
    variables: Record<string, unknown>;
}
export declare function createEmptyAnalysisState(): QuailAnalysisState;
export declare function cloneAnalysisState(state: QuailAnalysisState): QuailAnalysisState;
export declare function getAnalysisStateFromBranch(entries: readonly SessionEntry[]): QuailAnalysisState;
