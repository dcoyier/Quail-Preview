import type { AssistantMessage } from "@mariozechner/pi-ai";
import type { SessionManager } from "../core/session-manager.js";
import {
	getAnalysisStateFromBranch,
	QUAIL_ANALYSIS_RESULT_MESSAGE,
	QUAIL_ANALYSIS_STATE_ENTRY,
	type QuailAnalysisState,
} from "./analysis-state.js";
import {
	executeQuailCallBlocks,
	extractQuailCallBlocks,
	formatQuailExecutionResult,
	type QuailCallBlock,
} from "./dsl.js";

export interface QuailAnalysisRunResult {
	blocks: QuailCallBlock[];
	state: QuailAnalysisState;
	message: string;
	hasErrors: boolean;
}

export function getAssistantText(message: AssistantMessage): string {
	return message.content
		.filter((part): part is { type: "text"; text: string } => part.type === "text" && typeof part.text === "string")
		.map((part) => part.text)
		.join("\n");
}

export function getQuailCallBlocksFromAssistant(message: AssistantMessage): QuailCallBlock[] {
	return extractQuailCallBlocks(getAssistantText(message));
}

export async function runQuailAnalysisCalls(options: {
	cwd: string;
	sessionManager: SessionManager;
	assistantMessage: AssistantMessage;
}): Promise<QuailAnalysisRunResult | undefined> {
	const blocks = getQuailCallBlocksFromAssistant(options.assistantMessage);
	if (blocks.length === 0) return undefined;
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

export function createQuailAnalysisResultMessage(content: string): {
	role: "custom";
	customType: string;
	content: string;
	display: boolean;
	timestamp: number;
} {
	return {
		role: "custom",
		customType: QUAIL_ANALYSIS_RESULT_MESSAGE,
		content,
		display: true,
		timestamp: Date.now(),
	};
}
