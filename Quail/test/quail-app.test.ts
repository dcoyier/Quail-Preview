import { describe, expect, it } from "vitest";
import { currentApp } from "../src/apps/current.js";
import { getBuiltinSlashCommands } from "../src/core/slash-commands.js";
import { buildSystemPrompt } from "../src/core/system-prompt.js";

describe("quail app adapter", () => {
	it("centralizes Quail product behavior behind the active app", () => {
		expect(currentApp.id).toBe("quail");
		expect(currentApp.defaultActiveToolNames).toEqual(["quail"]);
		expect(currentApp.suppressUpstreamVersionCheck).toBe(true);
		expect(currentApp.suppressUpstreamChangelog).toBe(true);
		expect(currentApp.processingThread).toBeDefined();
	});

	it("adds Quail interactive commands through the app adapter", () => {
		const commands = getBuiltinSlashCommands().map((command) => command.name);
		expect(commands).toContain("process");
		expect(commands).toContain("exit");
		expect(commands).toContain("quit");
	});

	it("builds the Quail prompt through the app override without generic cwd/date footer", () => {
		const prompt = buildSystemPrompt({
			cwd: "/tmp/quail-test",
			selectedTools: ["quail"],
			toolSnippets: { quail: "Run Quail DSL" },
			quailActiveDatasets: [{ name: "Demo Dataset", entries: 3 }],
		});

		expect(prompt).toContain("For the quail tool");
		expect(prompt).toContain('- "Demo Dataset", 3');
		expect(prompt).not.toContain("Current working directory:");
	});
});
