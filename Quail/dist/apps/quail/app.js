import path from "node:path";
import { createQuailAnalysisResultMessage, runQuailAnalysisCalls } from "../../quail/analysis-runner.js";
import { handleDatasetCommand } from "../../quail/dataset-cli.js";
import { handleExecutorCommand } from "../../quail/executor-cli.js";
import { buildQuailMainSystemPrompt, buildQuailProcessingSystemPrompt, getActiveDatasetsForPrompt, } from "../../quail/prompts.js";
import { createQuailQueryToolDefinition } from "../../quail/query-tool.js";
import { quailInteractive } from "./interactive.js";
function buildQuailSystemPromptOverride(options) {
    if (options.customPrompt) {
        return undefined;
    }
    let prompt = buildQuailMainSystemPrompt({
        activeDatasets: options.quailActiveDatasets,
    });
    if (options.appendSection) {
        prompt += options.appendSection;
    }
    if (options.contextFiles.length > 0) {
        prompt += "\n\n# Project Context\n\n";
        for (const { path: filePath, content } of options.contextFiles) {
            prompt += `## ${filePath}\n\n${content}\n\n`;
        }
    }
    return prompt;
}
export const quailApp = {
    id: "quail",
    title: "Quail",
    description: "qualitative analysis harness with processed corpus search",
    defaultActiveToolNames: ["quail"],
    appendDateToCustomPrompt: false,
    cliHelpCommands: [{ usage: "executor start", description: "Start a warm Quail DSL executor server" }],
    interactive: quailInteractive,
    processingThread: {
        isActive: () => process.env.QUAIL_PROCESS_THREAD === "1",
        buildSystemPrompt: buildQuailProcessingSystemPrompt,
        buildEnvironment: (cwd) => ({
            ...process.env,
            QUAIL_PROCESS_THREAD: "1",
            PI_SKIP_VERSION_CHECK: "1",
            PATH: `${cwd}${path.delimiter}${process.env.PATH ?? ""}`,
        }),
    },
    suppressUpstreamVersionCheck: true,
    suppressUpstreamChangelog: true,
    changelogReplacementMessage: "Quail does not show Pi's upstream changelog. Quail-specific release notes are not bundled yet.",
    configureProcessEnvironment: () => {
        process.env.QUAIL_CODING_AGENT = "true";
    },
    handleCliCommand: async (args) => {
        if (await handleDatasetCommand(args)) {
            return true;
        }
        if (await handleExecutorCommand(args)) {
            return true;
        }
        return false;
    },
    createToolDefinitions: ({ cwd, sessionManager }) => [
        createQuailQueryToolDefinition(cwd, sessionManager),
    ],
    buildSystemPromptOverride: buildQuailSystemPromptOverride,
    getSystemPromptContext: ({ cwd, messages }) => ({
        quailActiveDatasets: getActiveDatasetsForPrompt(cwd, messages),
    }),
    shouldRebuildSystemPromptForUserMessage: true,
    afterAssistantMessage: async ({ cwd, sessionManager, assistantMessage }) => {
        const result = await runQuailAnalysisCalls({
            cwd,
            sessionManager,
            assistantMessage,
        });
        if (!result)
            return undefined;
        if (!result.message.trim())
            return undefined;
        return [createQuailAnalysisResultMessage(result.message)];
    },
};
//# sourceMappingURL=app.js.map