import {
	type AutocompleteItem,
	type AutocompleteProvider,
	type Component,
	fuzzyFilter,
	Text,
} from "@mariozechner/pi-tui";
import { theme } from "../../modes/interactive/theme/theme.js";
import { listDatasets } from "../../quail/dataset-store.js";
import { getDatasetMentionAliases } from "../../quail/prompts.js";
import type { AppHeaderOptions, AppInteractiveAdapter, AppStartupContent } from "../types.js";

const QUAIL_ASCII = ["   ,", "  (')>", "/(V)", " ^ ^"];

class QuailHeader extends Text {
	constructor(
		private readonly title: string,
		private readonly compactInstructions: string,
		private readonly expandedInstructions: string,
		private readonly onboarding: string,
		private readonly footer: string,
		expanded = false,
	) {
		super("", 0, 0);
		this.setExpanded(expanded);
	}

	setExpanded(expanded: boolean): void {
		const lines = QUAIL_ASCII.map((line) => theme.fg("accent", `  ${line}`));
		lines.push("");
		lines.push(theme.fg("accent", `  ${this.title}`));
		lines.push("");
		lines.push(...this.indentLines(expanded ? this.expandedInstructions : this.compactInstructions));
		lines.push(...this.indentLines(this.onboarding));
		lines.push(...this.indentLines(this.footer));
		this.setText(lines.join("\n"));
	}

	private indentLines(text: string): string[] {
		return text.split("\n").map((line) => ` ${line}`);
	}
}

class QuailDatasetAutocompleteProvider implements AutocompleteProvider {
	constructor(
		private readonly delegate: AutocompleteProvider,
		private readonly getCwd: () => string,
	) {}

	async getSuggestions(
		lines: string[],
		cursorLine: number,
		cursorCol: number,
		options: { signal: AbortSignal; force?: boolean },
	): Promise<{ items: AutocompleteItem[]; prefix: string } | null> {
		const line = lines[cursorLine] ?? "";
		const beforeCursor = line.slice(0, cursorCol);
		const mention = getDatasetMentionCompletion(beforeCursor);
		if (mention) {
			const { query, prefix } = mention;
			const datasets = listDatasets(this.getCwd()).map((dataset) => ({
				value: `@\"${dataset.name}\"`,
				label: dataset.name,
				description: formatDatasetDescription(dataset.entryCount, getDatasetMentionAliases(dataset)),
				insertText: `@\"${dataset.name}\"`,
				aliases: getDatasetMentionAliases(dataset),
			}));
			const items = query ? fuzzyFilter(datasets, query, (item) => `${item.label} ${item.description ?? ""} ${item.aliases.join(" ")}`) : datasets;
			return items.length > 0 ? { items, prefix } : null;
		}
		return this.delegate.getSuggestions(lines, cursorLine, cursorCol, options);
	}

	applyCompletion(
		lines: string[],
		cursorLine: number,
		cursorCol: number,
		item: AutocompleteItem,
		prefix: string,
	): { lines: string[]; cursorLine: number; cursorCol: number } {
		if (prefix.startsWith("@")) {
			const nextLines = [...lines];
			const line = nextLines[cursorLine] ?? "";
			const start = Math.max(0, cursorCol - prefix.length);
			const insertText = "insertText" in item && typeof item.insertText === "string" ? item.insertText : item.value;
			nextLines[cursorLine] = `${line.slice(0, start)}${insertText}${line.slice(cursorCol)}`;
			return {
				lines: nextLines,
				cursorLine,
				cursorCol: start + insertText.length,
			};
		}
		return this.delegate.applyCompletion(lines, cursorLine, cursorCol, item, prefix);
	}
}

function getDatasetMentionCompletion(beforeCursor: string): { prefix: string; query: string } | undefined {
	const quoted = beforeCursor.match(/@\s*"([^"\n]*)$/);
	if (quoted) return { prefix: quoted[0], query: quoted[1] ?? "" };
	const atIndex = beforeCursor.lastIndexOf("@");
	if (atIndex < 0) return undefined;
	const previous = atIndex > 0 ? beforeCursor[atIndex - 1] : "";
	if (/[A-Za-z0-9._%+-]/.test(previous)) return undefined;
	const prefix = beforeCursor.slice(atIndex);
	if (prefix.includes("\"") || prefix.includes("\n")) return undefined;
	return { prefix, query: prefix.slice(1).trimStart() };
}

function formatDatasetDescription(entryCount: number, aliases: readonly string[]): string {
	const brightAliases = aliases
		.filter((alias) => alias.startsWith("bright "))
		.slice(0, 2)
		.map((alias) => `@${alias}`);
	return brightAliases.length > 0
		? `${entryCount} entries · ${brightAliases.join(", ")}`
		: `${entryCount} entries`;
}

function getStartupContent(options: { isProcessingThread: boolean }): AppStartupContent {
	return {
		title: options.isProcessingThread ? "Quail Process" : "Quail",
		compactInstructions: theme.fg("muted", "escape interrupt · ctrl+c/ctrl+d clear/exit · / commands"),
		onboarding: theme.fg(
			"muted",
			options.isProcessingThread
				? "Process, inspect, or remove a dataset here. Exit to return to the main Quail thread."
				: "activate a processed dataset with @\"Dataset Name\" or use /process to add one",
		),
		compactOnboarding: theme.fg("dim", options.isProcessingThread ? "temporary dataset processing thread" : "a pi fork"),
	};
}

function createHeader(options: AppHeaderOptions): Component {
	return new QuailHeader(
		options.title,
		options.compactInstructions,
		options.expandedInstructions,
		options.onboarding,
		options.footer,
		options.expanded,
	);
}

export const quailInteractive: AppInteractiveAdapter = {
	slashCommands: [
		{ name: "process", description: "Open the temporary Quail dataset processing thread" },
		{ name: "exit", description: "Exit the processing thread", processThreadOnly: true },
	],
	wrapAutocompleteProvider: (delegate, getCwd) => new QuailDatasetAutocompleteProvider(delegate, getCwd),
	getStartupContent,
	createHeader,
};
