import type { SessionEntry } from "../core/session-manager.js";
export declare const QUAIL_ANALYSIS_STATE_ENTRY = "quail.analysis_state";
export declare const QUAIL_ANALYSIS_RESULT_MESSAGE = "quail.analysis_result";
export interface QuailGroupState {
    id: string;
    datasets: string[];
    scope: "entries" | "fields";
    spec: string;
    members: string[];
    entryIds?: string[];
    fieldNames?: string[];
    createdAt: string;
}
export interface QuailPythonBindingState {
    name: string;
    kind: "function" | "class";
    source: string;
}
export interface QuailAnalysisState {
    version: 1;
    nextGroupNumber: number;
    groups: Record<string, QuailGroupState>;
    tagsByEntry: Record<string, Record<string, unknown>>;
    variables: Record<string, unknown>;
    createdFields?: string[];
    pythonBindings?: QuailPythonBindingState[];
}
export declare function createEmptyAnalysisState(): QuailAnalysisState;
export declare function cloneAnalysisState(state: QuailAnalysisState): QuailAnalysisState;
export declare function getAnalysisStateFromBranch(entries: readonly SessionEntry[]): QuailAnalysisState;
