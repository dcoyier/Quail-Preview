import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { CONFIG_DIR_NAME } from "../src/config.js";
import {
	getQuailDatasetsDir,
	getQuailStagingDir,
	getQuailWorkspaceDir,
	QUAIL_DATASETS_DIR,
	QUAIL_STAGING_DIR,
	QUAIL_WORKSPACE_DIR,
	QUAIL_WORKSPACE_PATH_ENV,
	QUAIL_WORKSPACE_SCOPE_ENV,
} from "../src/quail/paths.js";

describe("quail workspace resolution", () => {
	const cwd = "/tmp/quail-launch-folder";
	let previousWorkspacePath: string | undefined;
	let previousWorkspaceScope: string | undefined;

	beforeEach(() => {
		previousWorkspacePath = process.env[QUAIL_WORKSPACE_PATH_ENV];
		previousWorkspaceScope = process.env[QUAIL_WORKSPACE_SCOPE_ENV];
		delete process.env[QUAIL_WORKSPACE_PATH_ENV];
		delete process.env[QUAIL_WORKSPACE_SCOPE_ENV];
	});

	afterEach(() => {
		if (previousWorkspacePath === undefined) delete process.env[QUAIL_WORKSPACE_PATH_ENV];
		else process.env[QUAIL_WORKSPACE_PATH_ENV] = previousWorkspacePath;
		if (previousWorkspaceScope === undefined) delete process.env[QUAIL_WORKSPACE_SCOPE_ENV];
		else process.env[QUAIL_WORKSPACE_SCOPE_ENV] = previousWorkspaceScope;
	});

	it("uses a global Quail workspace by default instead of the launch directory", () => {
		expect(getQuailWorkspaceDir(cwd)).toBe(join(homedir(), CONFIG_DIR_NAME, QUAIL_WORKSPACE_DIR));
		expect(getQuailDatasetsDir(cwd)).toBe(join(homedir(), CONFIG_DIR_NAME, QUAIL_WORKSPACE_DIR, QUAIL_DATASETS_DIR));
		expect(getQuailStagingDir(cwd)).toBe(join(homedir(), CONFIG_DIR_NAME, QUAIL_WORKSPACE_DIR, QUAIL_STAGING_DIR));
	});

	it("supports an explicit absolute workspace override", () => {
		process.env[QUAIL_WORKSPACE_PATH_ENV] = "/Volumes/research/quail-workspace";
		expect(getQuailWorkspaceDir(cwd)).toBe("/Volumes/research/quail-workspace");
	});

	it("resolves relative workspace overrides from the launch directory", () => {
		process.env[QUAIL_WORKSPACE_PATH_ENV] = "shared-quail-workspace";
		expect(getQuailWorkspaceDir(cwd)).toBe(resolve(cwd, "shared-quail-workspace"));
	});

	it("keeps project-local workspace mode available when explicitly requested", () => {
		process.env[QUAIL_WORKSPACE_SCOPE_ENV] = "project";
		expect(getQuailWorkspaceDir(cwd)).toBe(join(cwd, QUAIL_WORKSPACE_DIR));
	});
});
