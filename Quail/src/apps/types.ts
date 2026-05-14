import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { AssistantMessage } from "@mariozechner/pi-ai";
import type { AutocompleteProvider, Component } from "@mariozechner/pi-tui";
import type { ToolDefinition } from "../core/extensions/index.js";
import type { SessionManager } from "../core/session-manager.js";

export interface AppSlashCommand {
	name: string;
	description: string;
	processThreadOnly?: boolean;
}

export interface AppHeaderOptions {
	title: string;
	compactInstructions: string;
	expandedInstructions: string;
	onboarding: string;
	footer: string;
	expanded: boolean;
}

export interface AppStartupContent {
	title: string;
	compactInstructions: string;
	onboarding: string;
	compactOnboarding: string;
}

export interface AppInteractiveAdapter {
	slashCommands?: readonly AppSlashCommand[];
	wrapAutocompleteProvider?: (delegate: AutocompleteProvider, getCwd: () => string) => AutocompleteProvider;
	getStartupContent?: (options: { isProcessingThread: boolean }) => AppStartupContent;
	createHeader?: (options: AppHeaderOptions) => Component;
}

export interface AppProcessingThreadAdapter {
	isActive: () => boolean;
	buildSystemPrompt: (cwd: string) => string;
	buildEnvironment: (cwd: string) => NodeJS.ProcessEnv;
}

export interface AppToolOptions {
	cwd: string;
	sessionManager: SessionManager;
}

export interface AppSystemPromptOverrideOptions {
	customPrompt?: string;
	appendSection: string;
	contextFiles: Array<{ path: string; content: string }>;
	quailActiveDatasets?: unknown[];
}

export interface AppPromptContextOptions {
	cwd: string;
	messages: readonly AgentMessage[];
}

export interface AppAssistantMessageOptions {
	cwd: string;
	sessionManager: SessionManager;
	assistantMessage: AssistantMessage;
}

export interface AppCliHelpCommand {
	usage: string;
	description: string;
}

export interface AppDefinition {
	id: string;
	title: string;
	description: string;
	defaultActiveToolNames?: readonly string[];
	appendDateToCustomPrompt?: boolean;
	cliHelpCommands?: readonly AppCliHelpCommand[];
	interactive?: AppInteractiveAdapter;
	processingThread?: AppProcessingThreadAdapter;
	suppressUpstreamVersionCheck?: boolean;
	suppressUpstreamChangelog?: boolean;
	changelogReplacementMessage?: string;
	configureProcessEnvironment?: () => void;
	handleCliCommand?: (args: string[]) => Promise<boolean>;
	createToolDefinitions?: (options: AppToolOptions) => ToolDefinition[];
	buildSystemPromptOverride?: (options: AppSystemPromptOverrideOptions) => string | undefined;
	getSystemPromptContext?: (options: AppPromptContextOptions) => Record<string, unknown>;
	shouldRebuildSystemPromptForUserMessage?: boolean;
	afterAssistantMessage?: (options: AppAssistantMessageOptions) => Promise<AgentMessage[] | undefined>;
}
