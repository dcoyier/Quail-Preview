import type { AssistantMessage } from "@mariozechner/pi-ai";
import type { SessionManager } from "../core/session-manager.js";
import { type QuailAnalysisState } from "./analysis-state.js";
import { type QuailCallBlock } from "./dsl.js";
export interface QuailAnalysisRunResult {
    blocks: QuailCallBlock[];
    state: QuailAnalysisState;
    message: string;
    hasErrors: boolean;
}
export declare function getAssistantText(message: AssistantMessage): string;
export declare function getQuailCallBlocksFromAssistant(message: AssistantMessage): QuailCallBlock[];
export declare function runQuailAnalysisCalls(options: {
    cwd: string;
    sessionManager: SessionManager;
    assistantMessage: AssistantMessage;
}): Promise<QuailAnalysisRunResult | undefined>;
export declare function createQuailAnalysisResultMessage(content: string): {
    role: "custom";
    customType: string;
    content: string;
    display: boolean;
    timestamp: number;
};
