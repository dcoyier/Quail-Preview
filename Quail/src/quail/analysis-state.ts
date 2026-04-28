import type { SessionEntry } from "../core/session-manager.js";

export const QUAIL_ANALYSIS_STATE_ENTRY = "quail.analysis_state";
export const QUAIL_ANALYSIS_RESULT_MESSAGE = "quail.analysis_result";

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

export function createEmptyAnalysisState(): QuailAnalysisState {
	return {
		version: 1,
		nextGroupNumber: 1,
		groups: {},
		tagsByEntry: {},
		variables: {},
	};
}

function isAnalysisState(value: unknown): value is QuailAnalysisState {
	return (
		typeof value === "object" &&
		value !== null &&
		(value as { version?: unknown }).version === 1 &&
		typeof (value as { nextGroupNumber?: unknown }).nextGroupNumber === "number" &&
		typeof (value as { groups?: unknown }).groups === "object"
	);
}

export function cloneAnalysisState(state: QuailAnalysisState): QuailAnalysisState {
	return JSON.parse(JSON.stringify(state)) as QuailAnalysisState;
}

export function getAnalysisStateFromBranch(entries: readonly SessionEntry[]): QuailAnalysisState {
	for (let i = entries.length - 1; i >= 0; i--) {
		const entry = entries[i];
		if (entry.type !== "custom" || entry.customType !== QUAIL_ANALYSIS_STATE_ENTRY) continue;
		if (isAnalysisState(entry.data)) return cloneAnalysisState(entry.data);
	}
	return createEmptyAnalysisState();
}
