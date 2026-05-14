import { existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { CONFIG_DIR_NAME } from "../config.js";

export const QUAIL_WORKSPACE_DIR = "workspace";
export const QUAIL_DATASETS_DIR = "datasets";
export const QUAIL_STAGING_DIR = "staging";
export const QUAIL_WORKSPACE_PATH_ENV = "QUAIL_WORKSPACE_PATH";
export const QUAIL_WORKSPACE_SCOPE_ENV = "QUAIL_WORKSPACE_SCOPE";

function expandHomePath(path: string): string {
	if (path === "~") return homedir();
	if (path.startsWith("~/")) return join(homedir(), path.slice(2));
	return path;
}

function resolveWorkspaceOverride(cwd: string, value: string): string {
	const expanded = expandHomePath(value);
	return isAbsolute(expanded) ? expanded : resolve(cwd, expanded);
}

function useProjectWorkspace(): boolean {
	const scope = process.env[QUAIL_WORKSPACE_SCOPE_ENV]?.trim().toLowerCase();
	return scope === "project" || scope === "cwd" || scope === "local";
}

export function getQuailWorkspaceDir(cwd: string): string {
	const explicitPath = process.env[QUAIL_WORKSPACE_PATH_ENV]?.trim();
	if (explicitPath) {
		return resolveWorkspaceOverride(cwd, explicitPath);
	}
	if (useProjectWorkspace()) {
		return join(cwd, QUAIL_WORKSPACE_DIR);
	}
	return join(homedir(), CONFIG_DIR_NAME, QUAIL_WORKSPACE_DIR);
}

export function getQuailDatasetsDir(cwd: string): string {
	return join(getQuailWorkspaceDir(cwd), QUAIL_DATASETS_DIR);
}

export function getQuailStagingDir(cwd: string): string {
	return join(getQuailWorkspaceDir(cwd), QUAIL_STAGING_DIR);
}

export function ensureQuailWorkspace(cwd: string): void {
	for (const dir of [getQuailWorkspaceDir(cwd), getQuailDatasetsDir(cwd), getQuailStagingDir(cwd)]) {
		if (!existsSync(dir)) {
			mkdirSync(dir, { recursive: true });
		}
	}
}
