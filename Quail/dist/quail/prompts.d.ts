import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { type DatasetListItem } from "./dataset-store.js";
export interface ActiveDatasetInfo {
    name: string;
    entries: number;
}
type DatasetActivationTarget = Pick<DatasetListItem, "name" | "slug">;
export declare function getDatasetMentionAliases(dataset: DatasetActivationTarget): string[];
export declare function getActivatedDatasetNamesFromMessages(messages: readonly AgentMessage[], datasets?: readonly DatasetActivationTarget[]): string[];
export declare function getActiveDatasetsForPrompt(cwd: string, messages: readonly AgentMessage[]): ActiveDatasetInfo[];
export declare function buildQuailMainSystemPrompt(options: {
    activeDatasets?: readonly ActiveDatasetInfo[];
}): string;
export declare function buildQuailProcessingSystemPrompt(cwd: string): string;
export {};
