export const QUAIL_ANALYSIS_STATE_ENTRY = "quail.analysis_state";
export const QUAIL_ANALYSIS_RESULT_MESSAGE = "quail.analysis_result";
export function createEmptyAnalysisState() {
    return {
        version: 1,
        nextGroupNumber: 1,
        groups: {},
        tagsByEntry: {},
        variables: {},
    };
}
function isAnalysisState(value) {
    return (typeof value === "object" &&
        value !== null &&
        value.version === 1 &&
        typeof value.nextGroupNumber === "number" &&
        typeof value.groups === "object");
}
export function cloneAnalysisState(state) {
    return JSON.parse(JSON.stringify(state));
}
export function getAnalysisStateFromBranch(entries) {
    for (let i = entries.length - 1; i >= 0; i--) {
        const entry = entries[i];
        if (entry.type !== "custom" || entry.customType !== QUAIL_ANALYSIS_STATE_ENTRY)
            continue;
        if (isAnalysisState(entry.data))
            return cloneAnalysisState(entry.data);
    }
    return createEmptyAnalysisState();
}
//# sourceMappingURL=analysis-state.js.map