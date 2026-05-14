import type { SourceInfo } from "./source-info.js";
export type SlashCommandSource = "extension" | "prompt" | "skill";
export interface SlashCommandInfo {
    name: string;
    description?: string;
    source: SlashCommandSource;
    sourceInfo: SourceInfo;
}
export interface BuiltinSlashCommand {
    name: string;
    description: string;
    processThreadOnly?: boolean;
}
export declare function getBuiltinSlashCommands(): ReadonlyArray<BuiltinSlashCommand>;
