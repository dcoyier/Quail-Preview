import { getAnalysisStateFromBranch, QUAIL_ANALYSIS_RESULT_MESSAGE, QUAIL_ANALYSIS_STATE_ENTRY, } from "./analysis-state.js";
import { executeQuailCallBlocks, extractQuailCallBlocks, formatQuailExecutionResult, } from "./dsl.js";
export function getAssistantText(message) {
    return message.content
        .filter((part) => part.type === "text" && typeof part.text === "string")
        .map((part) => part.text)
        .join("\n");
}
export function getQuailCallBlocksFromAssistant(message) {
    return extractQuailCallBlocks(getAssistantText(message));
}
export async function runQuailAnalysisCalls(options) {
    const blocks = getQuailCallBlocksFromAssistant(options.assistantMessage);
    if (blocks.length === 0)
        return undefined;
    const state = getAnalysisStateFromBranch(options.sessionManager.getBranch());
    const result = await executeQuailCallBlocks({ cwd: options.cwd, state, blocks });
    options.sessionManager.appendCustomEntry(QUAIL_ANALYSIS_STATE_ENTRY, result.state);
    return {
        blocks,
        state: result.state,
        message: formatQuailExecutionResult(result),
        hasErrors: result.errors.length > 0,
    };
}
export function createQuailAnalysisResultMessage(content) {
    return {
        role: "custom",
        customType: QUAIL_ANALYSIS_RESULT_MESSAGE,
        content,
        display: true,
        timestamp: Date.now(),
    };
}
//# sourceMappingURL=analysis-runner.js.map