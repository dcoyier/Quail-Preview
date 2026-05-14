import type { SessionEntry } from "../core/session-manager.js";

export const QUAIL_ANALYSIS_STATE_ENTRY = "quail.analysis_state";
export const QUAIL_ANALYSIS_RESULT_MESSAGE = "quail.analysis_result";

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

export function createEmptyAnalysisState(): QuailAnalysisState {
	return {
		version: 1,
		nextGroupNumber: 2,
		groups: {},
		tagsByEntry: {},
		variables: {},
		createdFields: [],
		pythonBindings: [],
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
