import { Type, type Static } from "typebox";
import type { SessionManager } from "../core/session-manager.js";
import type { ToolDefinition } from "../core/extensions/types.js";
declare const quailQuerySchema: Type.TObject<{
    datasets: Type.TArray<Type.TString>;
    code: Type.TString;
}>;
export type QuailQueryToolInput = Static<typeof quailQuerySchema>;
export declare const DEFAULT_QUAIL_QUERY_OUTPUT_PREVIEW_LINES = 10;
export interface QuailQueryToolOptions {
    /**
     * Number of result lines shown in the TUI preview before the user expands the
     * tool row. The model still receives the full tool result.
     */
    outputPreviewLines?: number;
}
export interface QuailQueryToolDetails {
    datasets: string[];
    blocks: number;
    hasErrors: boolean;
    outputPreviewLines: number;
    outputLineCount: number;
}
export declare function createQuailQueryToolDefinition(cwd: string, sessionManager: SessionManager, options?: QuailQueryToolOptions): ToolDefinition<typeof quailQuerySchema, QuailQueryToolDetails>;
export {};
