import { Text } from "@mariozechner/pi-tui";
import { Type, type Static } from "typebox";
import { keyHint } from "../modes/interactive/components/keybinding-hints.js";
import type { SessionManager } from "../core/session-manager.js";
import type { ToolDefinition, ToolRenderResultOptions } from "../core/extensions/types.js";
import { getTextOutput } from "../core/tools/render-utils.js";
import {
	getAnalysisStateFromBranch,
	QUAIL_ANALYSIS_STATE_ENTRY,
} from "./analysis-state.js";
import {
	executeQuailCallBlocks,
	formatQuailExecutionResult,
	type QuailCallBlock,
} from "./dsl.js";

const quailQuerySchema = Type.Object({
	datasets: Type.Array(Type.String(), {
		description: "Dataset names to query, usually the activated dataset names shown in the prompt.",
	}),
	code: Type.String({
		description:
			"Quail DSL code to execute. Do not include $ wrappers or an @ dataset line; pass datasets separately.",
	}),
});

export type QuailQueryToolInput = Static<typeof quailQuerySchema>;

export const DEFAULT_QUAIL_QUERY_OUTPUT_PREVIEW_LINES = 10;
const QUAIL_RESULT_CONTINUATION_HINT = "Use this result to continue. If there were parse/runtime errors, correct the code call before answering.";

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
	executionTimeMs: number;
	outputPreviewLines: number;
	outputLineCount: number;
}

function normalizeOutputPreviewLines(value: number | undefined): number {
	if (value === undefined || !Number.isFinite(value)) return DEFAULT_QUAIL_QUERY_OUTPUT_PREVIEW_LINES;
	return Math.max(1, Math.floor(value));
}

function formatDatasetList(datasets: unknown, theme: typeof import("../modes/interactive/theme/theme.js").theme): string {
	if (!Array.isArray(datasets)) return theme.fg("dim", "...");
	const names = datasets.filter((name): name is string => typeof name === "string" && name.trim().length > 0);
	if (names.length === 0) return theme.fg("dim", "...");
	return names.map((name) => theme.fg("accent", `"${name}"`)).join(theme.fg("muted", ", "));
}

function formatCodePreview(code: unknown, theme: typeof import("../modes/interactive/theme/theme.js").theme): string {
	if (typeof code !== "string" || code.length === 0) return theme.fg("dim", "...");
	return code
		.replace(/\r/g, "")
		.split("\n")
		.map((line) => theme.fg("toolOutput", line))
		.join("\n");
}

function formatQuailQueryCall(
	args: Partial<QuailQueryToolInput> | undefined,
	theme: typeof import("../modes/interactive/theme/theme.js").theme,
	context: { argsComplete: boolean; executionStarted: boolean },
): string {
	let text = `${theme.fg("toolTitle", theme.bold("quail"))} ${theme.fg("toolTitle", theme.bold("input"))}`;
	if (!context.argsComplete && !context.executionStarted) {
		text += theme.fg("dim", " streaming...");
	}
	text += `\n${theme.fg("muted", "datasets:")} ${formatDatasetList(args?.datasets, theme)}`;
	text += `\n${theme.fg("muted", "code:")}\n${formatCodePreview(args?.code, theme)}`;
	return text;
}

function formatQuailQueryResult(
	result: {
		content: Array<{ type: string; text?: string; data?: string; mimeType?: string }>;
		details?: QuailQueryToolDetails;
	},
	options: ToolRenderResultOptions,
	theme: typeof import("../modes/interactive/theme/theme.js").theme,
): string {
	const output = getTextOutput(result, false).trimEnd();
	if (!output) return "";

	let displayOutput = output.startsWith("Output:\n") ? output.slice("Output:\n".length) : output;
	displayOutput = displayOutput
		.split("\n")
		.filter((line) => !/^Call \d+ datasets: /.test(line) && line !== QUAIL_RESULT_CONTINUATION_HINT)
		.join("\n")
		.trimEnd();
	if (!displayOutput) return "";
	const lines = displayOutput.split("\n");
	const previewLines = normalizeOutputPreviewLines(result.details?.outputPreviewLines);
	const maxLines = options.expanded ? lines.length : previewLines;
	const displayLines = lines.slice(0, maxLines);
	const remaining = lines.length - maxLines;
	let text = `\n${theme.fg("toolTitle", theme.bold("output"))}`;
	text += `\n${displayLines.map((line) => theme.fg("toolOutput", line)).join("\n")}`;
	if (remaining > 0) {
		text += `${theme.fg("muted", `\n... (${remaining} more lines,`)} ${keyHint("app.tools.expand", "to expand")})`;
	}
	return text;
}

function countLines(text: string): number {
	return text.length === 0 ? 0 : text.split("\n").length;
}

export function createQuailQueryToolDefinition(
	cwd: string,
	sessionManager: SessionManager,
	options: QuailQueryToolOptions = {},
): ToolDefinition<typeof quailQuerySchema, QuailQueryToolDetails> {
	const outputPreviewLines = normalizeOutputPreviewLines(options.outputPreviewLines);

	return {
		name: "quail",
		label: "quail",
		description:
			"Execute Quail qualitative-analysis DSL against processed datasets. Use this for dataset metadata, grouping, retrieval, counts, distributions, and entry lookup. Pass dataset names in datasets and DSL statements in code; do not write $ blocks as assistant text.",
		promptSnippet: "Run Quail qualitative-analysis DSL against processed datasets",
		promptGuidelines: [
			"For questions about activated Quail datasets, call quail instead of writing DSL blocks as plain text.",
			"Pass dataset names through the datasets argument and DSL statements through code; do not include $ wrappers or @ dataset lines.",
			"Use print() for any values that should be returned in the tool result.",
			"When inspecting source fields for multiple entries, retrieve entry ids first and call get(ids) once; avoid loops that call get(id) per entry.",
		],
		parameters: quailQuerySchema,
		executionMode: "sequential",
		async execute(_toolCallId, params, signal) {
			if (signal?.aborted) {
				throw new Error("Operation aborted");
			}

			const datasets = params.datasets.map((name) => name.trim()).filter(Boolean);
			if (datasets.length === 0) {
				throw new Error("quail requires at least one dataset name");
			}
			const code = params.code.trim();
			if (!code) {
				throw new Error("quail requires non-empty code");
			}

			const state = getAnalysisStateFromBranch(sessionManager.getBranch());
			const block: QuailCallBlock = {
				datasets,
				code,
				raw: `$\n@${datasets.map((name) => `"${name}"`).join(", ")}\n${code}\n$`,
			};
				const startedAt = performance.now();
				const result = await executeQuailCallBlocks({ cwd, state, blocks: [block] });
				const executionTimeMs = Math.round(performance.now() - startedAt);
				sessionManager.appendCustomEntry(QUAIL_ANALYSIS_STATE_ENTRY, result.state);
				const output = formatQuailExecutionResult(result);

			return {
				content: [{ type: "text", text: output }],
				details: {
						datasets,
						blocks: result.blocks,
						hasErrors: result.errors.length > 0,
						executionTimeMs,
						outputPreviewLines,
						outputLineCount: countLines(output),
					},
			};
		},
		renderCall(args, theme, context) {
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			text.setText(formatQuailQueryCall(args, theme, context));
			return text;
		},
		renderResult(result, options, theme, context) {
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			text.setText(formatQuailQueryResult(result as any, options, theme));
			return text;
		},
	};
}
