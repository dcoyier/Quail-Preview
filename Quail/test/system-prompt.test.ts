import { describe, expect, test } from "vitest";
import { buildSystemPrompt } from "../src/core/system-prompt.js";
import {
	getActivatedDatasetNamesFromMessages,
	getDatasetMentionAliases,
} from "../src/quail/prompts.js";

describe("buildSystemPrompt", () => {
	test("uses Quail's DSL system prompt by default", () => {
		const prompt = buildSystemPrompt({
			selectedTools: [],
			contextFiles: [],
			skills: [],
			cwd: process.cwd(),
		});

		expect(prompt).toContain("You are an agent in a qualitative research harness.");
		expect(prompt).toContain("Quail DSL:");
		expect(prompt).toContain("These are the activated datasets:");
		expect(prompt).toContain("retrieve(DIRECTION AMOUNT UNIT.REGEX of GROUP-EXPR sorted by RANKING)");
		expect(prompt).toContain("g_save(GROUP-EXPR)");
		expect(prompt).toContain("- (none)");
		expect(prompt).not.toContain("Available tools:");
		expect(prompt).not.toContain("Current working directory:");
	});

	test("includes active dataset context in the Quail prompt", () => {
		const prompt = buildSystemPrompt({
			selectedTools: ["quail"],
			contextFiles: [],
			skills: [],
			cwd: process.cwd(),
			quailActiveDatasets: [{ name: "ASRS Primary Problem Narratives Tagged", entries: 231209 }],
		});

		expect(prompt).toContain('- "ASRS Primary Problem Narratives Tagged", 231209');
	});

	test("appends project context and explicit append text to the Quail prompt", () => {
		const prompt = buildSystemPrompt({
			selectedTools: ["quail"],
			contextFiles: [{ path: "/tmp/project/AGENTS.md", content: "Prefer compact evidence summaries." }],
			skills: [],
			cwd: process.cwd(),
			appendSystemPrompt: "Additional Quail instruction.",
		});

		expect(prompt).toContain("Additional Quail instruction.");
		expect(prompt).toContain("# Project Context");
		expect(prompt).toContain("## /tmp/project/AGENTS.md");
		expect(prompt).toContain("Prefer compact evidence summaries.");
	});

	test("lets an explicit custom prompt bypass the Quail DSL prompt", () => {
		const prompt = buildSystemPrompt({
			customPrompt: "Custom research harness prompt.",
			selectedTools: ["quail"],
			contextFiles: [],
			skills: [],
			cwd: "/tmp/quail-project",
			appendSystemPrompt: "Extra instruction.",
		});

		expect(prompt).toContain("Custom research harness prompt.");
		expect(prompt).toContain("Extra instruction.");
		expect(prompt).not.toContain("Quail DSL:");
		expect(prompt).not.toContain("Current working directory:");
	});

	test("resolves BRIGHT-style dataset mentions to processed retrieval dataset names", () => {
		const datasets = [
			{ name: "retrieval-biology", slug: "retrieval-biology" },
			{ name: "US Exec SOTU 2025", slug: "us-exec-sotu-2025" },
		];

		expect(getDatasetMentionAliases(datasets[0])).toContain("bright biology");
		expect(getActivatedDatasetNamesFromMessages([
			{ role: "user", content: "Use @ bright biology and compare it with @US Exec SOTU 2025." } as any,
		], datasets)).toEqual(["retrieval-biology", "US Exec SOTU 2025"]);
		expect(getActivatedDatasetNamesFromMessages([
			{ role: "user", content: "Use @\"BRIGHT Biology\"." } as any,
		], datasets)).toEqual(["retrieval-biology"]);
		expect(getActivatedDatasetNamesFromMessages([
			{ role: "user", content: "This email-like string x@ bright biology should not activate." } as any,
		], datasets)).toEqual([]);
	});
});
