import { fuzzyFilter, Text, } from "@mariozechner/pi-tui";
import { theme } from "../../modes/interactive/theme/theme.js";
import { listDatasets } from "../../quail/dataset-store.js";
const QUAIL_ASCII = ["   ,", "  (')>", "/(V)", " ^ ^"];
class QuailHeader extends Text {
    title;
    compactInstructions;
    expandedInstructions;
    onboarding;
    footer;
    constructor(title, compactInstructions, expandedInstructions, onboarding, footer, expanded = false) {
        super("", 0, 0);
        this.title = title;
        this.compactInstructions = compactInstructions;
        this.expandedInstructions = expandedInstructions;
        this.onboarding = onboarding;
        this.footer = footer;
        this.setExpanded(expanded);
    }
    setExpanded(expanded) {
        const lines = QUAIL_ASCII.map((line) => theme.fg("accent", `  ${line}`));
        lines.push("");
        lines.push(theme.fg("accent", `  ${this.title}`));
        lines.push("");
        lines.push(...this.indentLines(expanded ? this.expandedInstructions : this.compactInstructions));
        lines.push(...this.indentLines(this.onboarding));
        lines.push(...this.indentLines(this.footer));
        this.setText(lines.join("\n"));
    }
    indentLines(text) {
        return text.split("\n").map((line) => ` ${line}`);
    }
}
class QuailDatasetAutocompleteProvider {
    delegate;
    getCwd;
    constructor(delegate, getCwd) {
        this.delegate = delegate;
        this.getCwd = getCwd;
    }
    async getSuggestions(lines, cursorLine, cursorCol, options) {
        const line = lines[cursorLine] ?? "";
        const beforeCursor = line.slice(0, cursorCol);
        const match = beforeCursor.match(/@\"([^\"\n]*)$/) ?? beforeCursor.match(/@([^\s\"\n]*)$/);
        if (match) {
            const query = match[1] ?? "";
            const prefix = match[0];
            const datasets = listDatasets(this.getCwd()).map((dataset) => ({
                value: prefix.startsWith("@\"") ? `@\"${dataset.name}\"` : `@${dataset.name}`,
                label: dataset.name,
                description: `${dataset.entryCount} entries`,
                insertText: `@\"${dataset.name}\"`,
            }));
            const items = query ? fuzzyFilter(datasets, query, (item) => `${item.label} ${item.description ?? ""}`) : datasets;
            return items.length > 0 ? { items, prefix } : null;
        }
        return this.delegate.getSuggestions(lines, cursorLine, cursorCol, options);
    }
    applyCompletion(lines, cursorLine, cursorCol, item, prefix) {
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
function getStartupContent(options) {
    return {
        title: options.isProcessingThread ? "Quail Process" : "Quail",
        compactInstructions: theme.fg("muted", "escape interrupt · ctrl+c/ctrl+d clear/exit · / commands"),
        onboarding: theme.fg("muted", options.isProcessingThread
            ? "Process, inspect, or remove a dataset here. Exit to return to the main Quail thread."
            : "activate a processed dataset with @\"Dataset Name\" or use /process to add one"),
        compactOnboarding: theme.fg("dim", options.isProcessingThread ? "temporary dataset processing thread" : "a pi fork"),
    };
}
function createHeader(options) {
    return new QuailHeader(options.title, options.compactInstructions, options.expandedInstructions, options.onboarding, options.footer, options.expanded);
}
export const quailInteractive = {
    slashCommands: [
        { name: "process", description: "Open the temporary Quail dataset processing thread" },
        { name: "exit", description: "Exit the processing thread", processThreadOnly: true },
    ],
    wrapAutocompleteProvider: (delegate, getCwd) => new QuailDatasetAutocompleteProvider(delegate, getCwd),
    getStartupContent,
    createHeader,
};
//# sourceMappingURL=interactive.js.map