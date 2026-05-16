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
		expect(prompt).toContain("unless directed to.");
		expect(prompt).toContain("Quail DSL:");
		expect(prompt).toContain("These are the activated datasets:");
		expect(prompt).toContain("retrieve(DIRECTION AMOUNT UNIT.REGEX of GROUP-EXPR sorted by RANKING)");
		expect(prompt).toContain("g_save(GROUP-EXPR)");
		expect(prompt).toContain("Fundamentally, groups are just sets of either entries or fields; there are two core groups related to this. G0 is the set of all entires and G1 is the set of all fields");
		expect(prompt).toContain("These are the two possible overall scopes for groups. Groups are just subsets of one of these two.");
		expect(prompt).toContain("this can be used for relational retrieval *if* processing was sequential, which should not be assumed, but can be tested;");
		expect(prompt).toContain("Scores are computed from raw BM25 and cosine similarity for embed. Strict BM25 is the default");
		expect(prompt).toContain("- only what is within print( ... ) will be returned in the results of the quail tool call");
		expect(prompt).toContain("Usage must be var = g_save(...), where var is the variable where you are saving this group. This variable can then be plugged in as a group across tool executions.");
		expect(prompt).toContain("g_save(...) cannot be used bare or inline as a GROUP-EXPR and does not print a group ID");
		expect(prompt).not.toContain("Substituting g_save(GROUP) for a GROUP is the standard usage.");
		expect(prompt).toContain("Be careful with syntax.");
		expect(prompt).toContain("And most importantly: The quail tool is complex and incredibly versatile. Do not limit yourself in how you use it.");
		expect(prompt).not.toContain("This command will print a group ID");
		expect(prompt).not.toContain("This is the only command that automatically prints something.");
		expect(prompt).not.toContain("besides:");
		expect(prompt).not.toContain("group_expr(GROUP-EXPR)");
		expect(prompt).not.toContain("create an unsaved first-class group expression value");
		expect(prompt).not.toContain("Final essential notes:");
		expect(prompt).not.toContain("The quail tool is diverse, do not limit yourself.");
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
