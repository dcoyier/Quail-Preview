import type { AgentMessage } from "@mariozechner/pi-agent-core";
export interface ActiveDatasetInfo {
    name: string;
    entries: number;
}
export declare function getActivatedDatasetNamesFromMessages(messages: readonly AgentMessage[]): string[];
export declare function getActiveDatasetsForPrompt(cwd: string, messages: readonly AgentMessage[]): ActiveDatasetInfo[];
export declare function buildQuailMainSystemPrompt(options: {
    activeDatasets?: readonly ActiveDatasetInfo[];
}): string;
export declare function buildQuailProcessingSystemPrompt(cwd: string): string;
