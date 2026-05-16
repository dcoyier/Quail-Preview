import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createEmptyAnalysisState } from "../src/quail/analysis-state.js";
import { inspectDatasetFile, loadDatasets, processDataset, scoreEmbeddingVectorValues } from "../src/quail/dataset-store.js";
import { clearQuailDslRuntimeCaches, executeQuailCallBlocks, getQuailDslRuntimeCacheStats } from "../src/quail/dsl.js";
import { getQuailDatasetsDir } from "../src/quail/paths.js";

const DATASET_NAME = "DSL Check";
const DATASET_SLUG = "dsl-check";

describe("quail DSL", () => {
	let cwd: string;
	let previousWorkspacePath: string | undefined;
	let previousWorkspaceScope: string | undefined;

	beforeEach(async () => {
		previousWorkspacePath = process.env.QUAIL_WORKSPACE_PATH;
		previousWorkspaceScope = process.env.QUAIL_WORKSPACE_SCOPE;
		cwd = mkdtempSync(join(tmpdir(), "quail-dsl-"));
		process.env.QUAIL_WORKSPACE_PATH = join(cwd, "workspace");
		delete process.env.QUAIL_WORKSPACE_SCOPE;
		const inputPath = join(cwd, "dataset.csv");
		writeFileSync(
			inputPath,
			[
				"text,year",
				"Alpha apple first,2024",
				"Beta banana second,2025",
				"Gamma apple third,2024",
				"Delta final fourth,2026",
			].join("\n"),
			"utf8",
		);
		await processDataset({
			cwd,
			inputPath,
			name: DATASET_NAME,
			skipEmbeddings: true,
		});
	});

	afterEach(() => {
		if (previousWorkspacePath === undefined) delete process.env.QUAIL_WORKSPACE_PATH;
		else process.env.QUAIL_WORKSPACE_PATH = previousWorkspacePath;
		if (previousWorkspaceScope === undefined) delete process.env.QUAIL_WORKSPACE_SCOPE;
		else process.env.QUAIL_WORKSPACE_SCOPE = previousWorkspaceScope;
		rmSync(cwd, { recursive: true, force: true });
	});

	async function run(code: string) {
		return executeQuailCallBlocks({
			cwd,
			state: createEmptyAnalysisState(),
			blocks: [{ datasets: [DATASET_NAME], code, raw: code }],
		});
	}

	function stripStoredOrdinals() {
		const entriesPath = join(getQuailDatasetsDir(cwd), DATASET_SLUG, "entries.jsonl");
		const withoutOrdinals = readFileSync(entriesPath, "utf8")
			.split(/\r?\n/)
			.filter(Boolean)
			.map((line) => {
				const entry = JSON.parse(line) as Record<string, unknown>;
				delete entry.ordinal;
				return JSON.stringify(entry);
			})
			.join("\n");
		writeFileSync(entriesPath, `${withoutOrdinals}\n`, "utf8");
	}

	function removeFieldFromEntry(ordinal: number, field: string) {
		const entriesPath = join(getQuailDatasetsDir(cwd), DATASET_SLUG, "entries.jsonl");
		const updated = readFileSync(entriesPath, "utf8")
			.split(/\r?\n/)
			.filter(Boolean)
			.map((line, index) => {
				const entry = JSON.parse(line) as { fields?: Record<string, unknown> };
				if (index === ordinal - 1 && entry.fields) delete entry.fields[field];
				return JSON.stringify(entry);
			})
			.join("\n");
		writeFileSync(entriesPath, `${updated}\n`, "utf8");
	}

	function writeSemanticEmbeddingDataset() {
		const name = "Semantic Groups";
		const slug = "semantic-groups";
		const datasetsDir = getQuailDatasetsDir(cwd);
		const datasetDir = join(datasetsDir, slug);
		mkdirSync(datasetDir, { recursive: true });
		const entries = [
			{ id: `${slug}:000001`, dataset: name, ordinal: 1, text: "east anchor", fields: { text: "east anchor" }, tags: {}, contains: "east anchor", fieldContains: { text: "east anchor" } },
			{ id: `${slug}:000002`, dataset: name, ordinal: 2, text: "east repeat", fields: { text: "east repeat" }, tags: {}, contains: "east repeat", fieldContains: { text: "east repeat" } },
			{ id: `${slug}:000003`, dataset: name, ordinal: 3, text: "north contrast", fields: { text: "north contrast" }, tags: {}, contains: "north contrast", fieldContains: { text: "north contrast" } },
			{ id: `${slug}:000004`, dataset: name, ordinal: 4, text: "south contrast", fields: { text: "south contrast" }, tags: {}, contains: "south contrast", fieldContains: { text: "south contrast" } },
		];
		writeFileSync(join(datasetDir, "manifest.json"), JSON.stringify({
			name,
			slug,
			createdAt: "2026-05-14T00:00:00.000Z",
			updatedAt: "2026-05-14T00:00:00.000Z",
			entryCount: entries.length,
			metadataFields: ["text"],
			fieldNames: ["text"],
			textFields: ["text"],
			fieldTypes: { text: "string" },
			embeddedFields: ["text"],
			embeddingModel: "test-model",
			embeddingDimensions: 2,
			batchSize: 4,
			source: { format: "test" },
			files: { entries: "entries.jsonl", bm25: "bm25.json", embeddings: "embeddings.json" },
		}, null, 2), "utf8");
		writeFileSync(join(datasetDir, "entries.jsonl"), entries.map((entry) => JSON.stringify(entry)).join("\n") + "\n", "utf8");
		writeFileSync(join(datasetDir, "bm25.json"), JSON.stringify({
			k1: 1.5,
			b: 0.75,
			avgDocLength: 1,
			docCount: entries.length,
			docLengths: {},
			docFreq: {},
			termFreq: {},
		}), "utf8");
		writeFileSync(join(datasetDir, "embeddings.json"), JSON.stringify({
			format: "float32-binary-v1",
			model: "test-model",
			dimensions: 2,
			count: entries.length,
			ids: entries.map((entry) => `${entry.id}\u0000text`),
			vectorsFile: "embeddings.f32",
		}), "utf8");
		const buffer = Buffer.allocUnsafe(entries.length * 2 * 4);
		[
			1, 0,
			1, 0,
			0, 1,
			0, -1,
		].forEach((value, index) => buffer.writeFloatLE(value, index * 4));
		writeFileSync(join(datasetDir, "embeddings.f32"), buffer);
	}

	it("only returns normal values through print", async () => {
		const result = await run([
			"count(G0)",
			"get(tag_fields)",
			"var apple = group(contains: \"apple\")",
			"print(apple, count(apple))",
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim()).toBe("G2 2");
	});

	it("supports the new prompt's entry, field, scoped-group, and regex unit syntax", async () => {
		const result = await run([
			"print(count(entries of G0))",
			"print(count(fields of G1))",
			"print(retrieve(top 2 fields of G1))",
			`print(retrieve(top 2 entries of (scope: G0, ([text] BM25 similarity to "apple" > 0))))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output).toContain("4");
		expect(result.output).toContain("text");
		expect(result.output).toContain("year");
		expect(result.output).toContain(`${DATASET_SLUG}:000001`);
		expect(result.output).toContain(`${DATASET_SLUG}:000003`);
	});

	it("supports processed field inspection as tag units", async () => {
		const result = await run([
			`ids = retrieve(top 2 entries of G0)`,
			`print(get(ids[0])[year], get(ids[0])[text])`,
			`rows = get(ids)`,
			`print(rows[1].fields["year"], rows[1].fields["text"])`,
			`print(get(["year"]))`,
			`print(retrieve(top 1 entries[year] of G0))`,
			`print("after-field-unit")`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"2024 Alpha apple first",
			"2025 Beta banana second",
			`[`,
			`  2024,`,
			`  2025,`,
			`  2026`,
			`]`,
			`[`,
			`  2024`,
			`]`,
			"after-field-unit",
			]);
	});

	it("materializes missing processed fields in get(id).fields", async () => {
		removeFieldFromEntry(2, "text");
		removeFieldFromEntry(2, "year");
		const result = await run([
			`ids = retrieve(top 2 entries of G0)`,
			`rows = get(ids)`,
			`print(rows[0].fields["text"])`,
			`print(rows[1].fields["text"] == "")`,
			`print(rows[1].fields["year"] is None)`,
			`print(get(["text"]))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"Alpha apple first",
			"true",
			"true",
			`["Alpha apple first","Delta final fourth","Gamma apple third"]`,
		]);
	});

	it("supports g_save, create_field, group tag, typed tag values, and group untag", async () => {
		const result = await run([
			"create_field(reviewed)",
			`var apples = g_save(scope: G0, ([text] BM25 similarity to "apple" > 0))`,
			"tag(apples with reviewed set to true)",
			"print(apples, count(entries[reviewed] of apples))",
			"print(retrieve(top 2 entries[reviewed] of apples))",
			"print(retrieve(top 1 fields[reviewed] of G1))",
			"untag(reviewed from apples)",
			"print(count(entries[reviewed] of apples))",
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"Tagged 2 entries with 2 tag values across 1 field: reviewed.",
			"apples 2",
			`[`,
			`  true,`,
			`  true`,
			`]`,
			`[`,
			`  true`,
			`]`,
			"Removed 2 tag fields from 2 entries across 1 field: reviewed.",
			"0",
		]);
	});

	it("supports field-scoped boolean groups, regex remove/splice units, and arithmetic ranking", async () => {
		const result = await run([
			`tag("${DATASET_SLUG}:000001" with snippet set to "Alpha apple")`,
			`tag("${DATASET_SLUG}:000002" with snippet set to "Beta banana")`,
			`print(retrieve(top 2 fields of (scope: G1, length > 3 and not ("year"))))`,
			`print(retrieve(top 3 entries[snippet].remove(r" apple| banana").splice(0, 5) of G0))`,
			`print(retrieve(top 1 entries[snippet].find(r"Alpha|apple") of G0 sorted by BM25 similarity to "apple"))`,
			`print(retrieve(top 2 entries[snippet].find(r"\\w+") of G0))`,
			`print(retrieve(top 2 entries of G0 sorted by ([text] BM25 similarity to "banana") + (length / 100)))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output).toContain(`"text"`);
		expect(result.output).toContain(`"Alpha"`);
		expect(result.output).toContain(`"Beta"`);
		expect(result.output).toContain(`"apple"`);
		expect(result.output).toContain(`${DATASET_SLUG}:000002`);
	});

	it("preserves raw regex backslashes and ranks regex-derived unit text", async () => {
		const result = await run([
			`tag("${DATASET_SLUG}:000001" with snippet set to "Alpha apple A1")`,
			`tag("${DATASET_SLUG}:000002" with snippet set to "Beta banana B22")`,
			`print(retrieve(top 1 entries[snippet].find(r"Beta|banana") of G0 sorted by BM25 similarity to "banana"))`,
			`print(retrieve(top 2 entries[snippet].find(r"\\d+") of G0))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"Tagged 2 entries with 2 tag values across 1 field: snippet.",
			"[",
			`  "banana"`,
			"]",
			"[",
			`  "1",`,
			`  "22"`,
			"]",
		]);
	});

	it("retrieves regex-derived field snippets and reports missing call delimiters clearly", async () => {
		const ok = await run(`print(retrieve(top 2 entries[text].find(r"(?i).{0,10}apple.{0,10}") of (scope: G0, ([text].find(r"(?i)apple") length > 0))))`);

		expect(ok.errors).toEqual([]);
		expect(ok.output).toContain("apple");

		const missingPrintClose = await run(`print(retrieve(top 2 entries[text].find(r"(?i).{0,10}apple.{0,10}") of (scope: G0, ([text].find(r"(?i)apple") length > 0)))`);

		expect(missingPrintClose.output).toBe("");
		expect(missingPrintClose.errors).toMatchObject([{ code: "E_UNBALANCED_DELIMITER", line: 1 }]);
		expect(missingPrintClose.errors[0]?.message).toContain("Unclosed");
		expect(missingPrintClose.errors[0]?.message).toContain("print");
	});

	it("gives a repairable error for dot-style field units", async () => {
		const result = await run(`print(retrieve(top 1 entries.text.find(r"apple") of G0))`);

		expect(result.output).toBe("");
		expect(result.errors).toMatchObject([{ code: "E_PARSE_UNIT", line: 1 }]);
		expect(result.errors[0]?.message).toContain("entries[text].find");
	});

	it("gives repairable errors for common field and regex function mistakes", async () => {
		const bareField = await run(`print(count(entries of (scope: G0, (text.find(r"apple") length > 0))))`);
		expect(bareField.output).toBe("");
		expect(bareField.errors).toMatchObject([{ code: "E_FUNCTION", line: 1 }]);
		expect(bareField.errors[0]?.message).toContain("Field references require brackets");
		expect(bareField.errors[0]?.message).toContain("[text].find");

		const bareSimilarity = await run(`print(retrieve(top 1 entries of G0 sorted by (text BM25 similarity to "apple")))`);
		expect(bareSimilarity.output).toBe("");
		expect(bareSimilarity.errors).toMatchObject([{ code: "E_FUNCTION", line: 1 }]);
		expect(bareSimilarity.errors[0]?.message).toContain("Field references require brackets");
		expect(bareSimilarity.errors[0]?.message).toContain("[text] BM25 similarity");

		const pseudoEntry = await run(`print(count(entries of (scope: G0, (entry.find(r"apple") length > 0))))`);
		expect(pseudoEntry.output).toBe("");
		expect(pseudoEntry.errors).toMatchObject([{ code: "E_FUNCTION", line: 1 }]);
		expect(pseudoEntry.errors[0]?.message).toContain("There is no entry pseudo-field");
		expect(pseudoEntry.errors[0]?.message).toContain("Omitting [FIELD] operates on raw entry IDs");

		const missingDot = await run(`print(retrieve(top 1 entries[text] of G0 sorted by (find(r"apple") length)))`);
		expect(missingDot.output).toBe("");
		expect(missingDot.errors).toMatchObject([{ code: "E_FUNCTION", line: 1 }]);
		expect(missingDot.errors[0]?.message).toContain("Regex operations must start with a dot");
		expect(missingDot.errors[0]?.message).toContain(".find");

		const dotLength = await run(`print(retrieve(top 1 entries[text] of G0 sorted by (.find(r"apple").length)))`);
		expect(dotLength.output).toBe("");
		expect(dotLength.errors).toMatchObject([{ code: "E_FUNCTION", line: 1 }]);
		expect(dotLength.errors[0]?.message).toContain("space-based function syntax");
		expect(dotLength.errors[0]?.message).toContain("not .find");
	});

	it("explains missing group variables after failed g_save assignments", async () => {
		const result = await run([
			`bad_group = g_save((scope: G0, (text.find(r"apple") length > 0)))`,
			`print(count(entries of bad_group))`,
		].join("\n"));

		expect(result.output).toBe("");
		expect(result.errors).toMatchObject([
			{ code: "E_FUNCTION", line: 1 },
			{ code: "E_GROUP_EXPR", line: 2 },
		]);
		expect(result.errors[1]?.message).toContain("No group variable named bad_group exists");
		expect(result.errors[1]?.message).toContain("failed g_save assignments do not create groups");
	});

	it("explains standalone unit expressions and get() lookups that are not entry ids", async () => {
		const standaloneUnit = await run(`print(fields[text])`);
		expect(standaloneUnit.output).toBe("");
		expect(standaloneUnit.errors).toMatchObject([{ code: "E_PARSE_EXPR", line: 1 }]);
		expect(standaloneUnit.errors[0]?.message).toContain("fields[FIELD] are Quail units");
		expect(standaloneUnit.errors[0]?.message).toContain("retrieve(DIRECTION AMOUNT UNIT of GROUP-EXPR)");

		const unknownId = await run(`print(get("not-an-entry-id"))`);
		expect(unknownId.output).toBe("");
		expect(unknownId.errors).toMatchObject([{ code: "E_UNKNOWN_ID", line: 1 }]);
		expect(unknownId.errors[0]?.message).toContain("get(...) accepts an entry ID returned by retrieve");
		expect(unknownId.errors[0]?.message).toContain("Values returned by entries[FIELD] are field/tag values");
	});

	it("resolves Python regex pattern bindings inside Quail calls", async () => {
		const result = await run([
			`for word in ["alpha", "BETA", "delta", "missing"]:`,
			`    pat = "(?i)\\\\b" + word + "\\\\b"`,
			`    print(word, count(entries of (scope: G0, ([text].find(pat) length > 0))))`,
			`for word in ["alpha", "missing"]:`,
			`    print("concat", word, count(entries of (scope: G0, ([text].find("(?i)\\\\b" + word + "\\\\b") length > 0))))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"alpha 1",
			"BETA 1",
			"delta 1",
			"missing 0",
			"concat alpha 1",
			"concat missing 0",
		]);
	});

	it("supports Python-style inline regex flags at the start of patterns", async () => {
		const result = await run([
			`tag("${DATASET_SLUG}:000001" with snippet set to "Alpha\\nBeta")`,
			`tag("${DATASET_SLUG}:000002" with snippet set to "alpha only")`,
			`print(count(entries[snippet].find(r"(?s)Alpha.*Beta") of G0))`,
			`print(count(entries[snippet].find(r"(?m)^Beta") of G0))`,
			`print(count(entries[snippet].find(r"(?i)^alpha") of G0))`,
			`print(count(entries[snippet].find(r"(?:Alpha|Beta)") of G0))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"Tagged 2 entries with 2 tag values across 1 field: snippet.",
			"1",
			"1",
			"2",
			"2",
		]);
	});

	it("reports unsupported scoped Python inline regex flags with a repairable error", async () => {
		const result = await run(`print(retrieve(top 1 entries.find(r"(?i:alpha)") of G0))`);

		expect(result.output).toBe("");
		expect(result.errors).toMatchObject([{ code: "E_REGEX", line: 1 }]);
		expect(result.errors[0]?.message).toContain("Scoped inline regex flags");
	});

	it("treats entries[FIELD] and fields[FIELD] as unified field/tag units", async () => {
		const result = await run([
			`print(count(entries[year] of G0))`,
			`print(count(fields[year] of G1))`,
			`tag("${DATASET_SLUG}:000001" with "year" set to 2030)`,
			`tag("${DATASET_SLUG}:000002" with "year" add true)`,
			`print(retrieve(top 4 entries[year] of G0))`,
			`print(retrieve(top 4 fields[year] of G1))`,
			`print(count(tags: ["year": 2030]))`,
			`print(count(tags: ["year": true]))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"4",
			"3",
			"Tagged 2 entries with 2 tag values across 1 field: year.",
			`[`,
			`  2030,`,
			`  2025,`,
			`  true,`,
			`  2024`,
			`]`,
			`[`,
			`  2024,`,
			`  2025,`,
			`  2026,`,
			`  2030`,
			`]`,
			"1",
			"1",
		]);
	});

	it("supports get(ID)[FIELD] shorthand in direct DSL expressions", async () => {
		const result = await run([
			`print(get("${DATASET_SLUG}:000001")[year])`,
			`print(get("${DATASET_SLUG}:000001")[text])`,
			`tag("${DATASET_SLUG}:000001" with "year" set to 2030)`,
			`print(get("${DATASET_SLUG}:000001")[year])`,
			`print(get("${DATASET_SLUG}:000001").tags["year"])`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"2024",
			"Alpha apple first",
			"Tagged 1 entry with 1 tag value across 1 field: year.",
			"2030",
			"2030",
		]);
	});

	it("keeps processed and added field/tag values unified across discovery, grouping, and units", async () => {
		const result = await run([
			`create_field(review)`,
			`tag("${DATASET_SLUG}:000001" with review set to "checked")`,
			`tag("${DATASET_SLUG}:000001" with year set to 2030)`,
			`tag("${DATASET_SLUG}:000002" with year add true)`,
			`if True:`,
			`    fields = get(fields)`,
			`    tag_fields = get(tag_fields)`,
			`    print("field-check", "text" in fields, "year" in fields, "review" in fields, fields == tag_fields)`,
			`    print("year-values", get(["year"]))`,
			`    print("entry2-year", get("${DATASET_SLUG}:000002")[year])`,
			`    print("counts", count(tags: ["year": 2024]), count(entries of (scope: G0, ([year] == 2024))), count(entries[year] of G0), count(fields[year] of G1))`,
			`    print("review", count(tags: ["review": "checked"]), count(entries[review] of G0), count(fields[review] of G1))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"Tagged 2 entries with 3 tag values across 2 fields: review, year.",
			"field-check true true true true",
			`year-values [2024,2025,2026,2030,true]`,
			`entry2-year [2025,true]`,
			"counts 1 1 5 5",
			"review 1 1 1",
		]);
	});

	it("uses typed multi-value tags across units, groups, and count_by", async () => {
		const result = await run([
			`tag("${DATASET_SLUG}:000001" with codes set to ["attention", "communication"])`,
			`tag("${DATASET_SLUG}:000002" with codes set to ["fatigue", "attention"])`,
			`tag("${DATASET_SLUG}:000003" with codes set to ["fatigue"])`,
			`tag("${DATASET_SLUG}:000001" with confidence set to 0.75)`,
			`tag("${DATASET_SLUG}:000002" with confidence set to 0.25)`,
			`print(count(tags: ["codes": "attention"]))`,
			`print(count(entries of (scope: G0, [codes] == "fatigue" and not ([codes] == "attention"))))`,
			`print(count(entries of (scope: G0, [confidence] >= 0.5)))`,
			`print(retrieve(top 10 entries[codes] of G0))`,
			`print(retrieve(top 10 fields[codes] of G1))`,
			`print(count_by(["codes"] of G0))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"Tagged 3 entries with 7 tag values across 2 fields: codes, confidence.",
			"2",
			"1",
			"1",
			`[`,
			`  "attention",`,
			`  "communication",`,
			`  "fatigue",`,
			`  "attention",`,
			`  "fatigue"`,
			`]`,
			`[`,
			`  "attention",`,
			`  "communication",`,
			`  "fatigue"`,
			`]`,
			`[`,
			`  {`,
			`    "codes": "(missing)",`,
			`    "count": 1`,
			`  },`,
			`  {`,
			`    "codes": "attention",`,
			`    "count": 2`,
			`  },`,
			`  {`,
			`    "codes": "communication",`,
			`    "count": 1`,
			`  },`,
			`  {`,
			`    "codes": "fatigue",`,
			`    "count": 2`,
			`  }`,
			`]`,
		]);
	});

	it("supports dynamic tag field names in commands, units, and scoped clauses", async () => {
		const result = await run([
			`var code_field = "coded field"`,
			`create_field(code_field)`,
			`tag("${DATASET_SLUG}:000001" with code_field set to "alpha")`,
			`print(retrieve(top 1 entries[code_field] of G0))`,
			`print(retrieve(top 1 fields[code_field] of G1))`,
			`print(count(entries of (scope: G0, [code_field] == "alpha")))`,
			`print(count(tags: ["coded field": "alpha"]))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"Tagged 1 entry with 1 tag value across 1 field: coded field.",
			`[`,
			`  "alpha"`,
			`]`,
			`[`,
			`  "alpha"`,
			`]`,
			"1",
			"1",
		]);
	});

	it("supports assigning scoped group expressions directly to variables", async () => {
		const result = await run([
			`g = (scope: G0, ([year] == 2024))`,
			`print(count(entries of g))`,
			`print(retrieve(top 2 entries of g))`,
			`for threshold in [0.0]:`,
			`    h = (scope: G0, ([text] BM25 similarity to "apple" > threshold))`,
			`    print(count(entries of h))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"2",
			`["${DATASET_SLUG}:000001","${DATASET_SLUG}:000003"]`,
			"2",
		]);
	});

	it("supports scoped group literals as first-class Python values in containers and calls", async () => {
		const result = await run([
			`queries = [`,
			`    ("early", (scope: G0, ([year] == 2024))),`,
			`    ("recent", (scope: G0, ([year] >= 2025))),`,
			`]`,
			`for label, group in queries:`,
			`    print(label, count(entries of group))`,
			`groups = {"apple": (scope: G0, ([text] BM25 similarity to "apple" > 0))}`,
			`for label, group in groups.items():`,
			`    print(label, retrieve(top 1 entries of group)[0])`,
			`def show(label, group):`,
			`    print(label, count(entries of group))`,
			`show("call", (scope: G0, ([text].find(r"Gamma") length > 0)))`,
			`explicit = [("explicit", group_expr(scope: G0, ([text].find(r"Delta") length > 0)))]`,
			`for label, group in explicit:`,
			`    print(label, count(entries of group))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"early 2",
			"recent 2",
			`apple ${DATASET_SLUG}:000001`,
			"call 1",
			"explicit 1",
		]);
	});

	it("allows scoped groups to use group variables and saved groups as their base scope", async () => {
		const result = await run([
			`apple = (scope: G0, ([text] BM25 similarity to "apple" > 0))`,
			`apple_2024 = (scope: apple, ([year] == 2024))`,
			`print(count(entries of apple_2024))`,
			`mixed = (scope: G0, ([text] BM25 similarity to "banana" > 0) or ([text] BM25 similarity to "final" > 0))`,
			`late_mixed = (scope: mixed, ([year] >= 2026))`,
			`print(count(entries of late_mixed))`,
			`saved = g_save(scope: G0, ([text] BM25 similarity to "apple" > 0))`,
			`saved_2024 = (scope: saved, ([year] == 2024))`,
			`print(count(entries of saved_2024))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"2",
			"1",
			"2",
		]);
	});

	it("persists g_save variables across calls without emitting automatic G ids", async () => {
		const state = createEmptyAnalysisState();
		const first = await executeQuailCallBlocks({
			cwd,
			state,
			blocks: [{ datasets: [DATASET_NAME], code: [
				`coconut_carp = g_save((scope: G0,`,
				`    ([text].find(r"[Aa]pple") length > 0)`,
				`))`,
				`print(coconut_carp, count(entries of coconut_carp))`,
			].join("\n"), raw: "" }],
		});

		expect(first.errors).toEqual([]);
		expect(Object.keys(first.state.groups)).toEqual(["coconut_carp"]);
		expect(first.state.variables.coconut_carp).toBe("coconut_carp");
		expect(first.output.trim()).toBe("coconut_carp 2");

		const second = await executeQuailCallBlocks({
			cwd,
			state: first.state,
			blocks: [{ datasets: [DATASET_NAME], code: [
				`print(get(coconut_carp).id, count(entries of coconut_carp))`,
			].join("\n"), raw: "" }],
		});

		expect(second.errors).toEqual([]);
		expect(second.output.trim()).toBe("coconut_carp 2");
	});

	it("saves g_save variables through the Python runtime path", async () => {
		const result = await run([
			`for word in ["apple"]:`,
			`    loop_saved = g_save((scope: G0, ([text] BM25 similarity to word > 0)) )`,
			`print(loop_saved, count(entries of loop_saved))`,
			`print(get(loop_saved).id)`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(Object.keys(result.state.groups)).toEqual(["loop_saved"]);
		expect(result.state.variables.loop_saved).toBe("loop_saved");
		expect(result.output.trim().split("\n")).toEqual([
			"loop_saved 2",
			"loop_saved",
		]);
	});

	it("rejects unassigned or inline g_save calls", async () => {
		const bare = await run(`g_save(scope: G0, ([text] BM25 similarity to "apple" > 0))`);
		expect(bare.errors).toEqual([
			expect.objectContaining({ code: "E_G_SAVE_ASSIGNMENT" }),
		]);

		const inline = await run(`print(count(entries of (g_save(scope: G0, ([text] BM25 similarity to "apple" > 0)) and G0)))`);
		expect(inline.errors).toEqual([
			expect.objectContaining({ code: "E_G_SAVE_ASSIGNMENT" }),
		]);

		const reserved = await run(`G2 = g_save(scope: G0, ([text] BM25 similarity to "apple" > 0))`);
		expect(reserved.errors).toEqual([
			expect.objectContaining({ code: "E_G_SAVE_NAME" }),
		]);
	});

	it("preserves composed group assignments and boolean precedence in Python execution", async () => {
		const result = await run([
			`org = (scope: G0, ([year] == 2024))`,
			`terms = ["apple", "banana"]`,
			`all_entries = G0`,
			`print("all", count(entries of all_entries))`,
			`for q in terms:`,
			`    g = (org and (scope: G0, ([text] BM25 similarity to q > 0)))`,
			`    print(q, count(entries of g))`,
			`apple = (scope: G0, ([text] BM25 similarity to "apple" > 0))`,
			`banana = (scope: G0, ([text] BM25 similarity to "banana" > 0))`,
			`final = (scope: G0, ([text] BM25 similarity to "final" > 0))`,
			`default_precedence = apple or banana and final`,
			`explicit_parens = (apple or banana) and final`,
			`print("precedence", count(entries of default_precedence), count(entries of explicit_parens))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"all 4",
			"apple 2",
			"banana 0",
			"precedence 2 0",
		]);
	});

	it("preserves scoped group literals inside persisted Python bindings", async () => {
		const state = createEmptyAnalysisState();
		const first = await executeQuailCallBlocks({
			cwd,
			state,
			blocks: [{ datasets: [DATASET_NAME], code: [
				`def early_group():`,
				`    return (scope: G0, ([year] == 2024))`,
				`print("defined")`,
			].join("\n"), raw: "" }],
		});

		expect(first.errors).toEqual([]);
		expect(first.output.trim()).toBe("defined");

		const second = await executeQuailCallBlocks({
			cwd,
			state: first.state,
			blocks: [{ datasets: [DATASET_NAME], code: [
				`group = early_group()`,
				`print(count(entries of group))`,
			].join("\n"), raw: "" }],
		});

		expect(second.errors).toEqual([]);
		expect(second.output.trim()).toBe("2");
	});

	it("persists field-scoped saved groups and retrieves their field units", async () => {
		const result = await run([
			`var field_group = g_save(scope: G1, length >= 4)`,
			`print(get(groups))`,
			`print(get(field_group).scope, len(get(field_group).fieldNames))`,
			`print(count(fields of field_group))`,
			`print(retrieve(top 10 fields of field_group))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			`[`,
			`  "field_group"`,
			`]`,
			"fields 2",
			"2",
			`[`,
			`  "text",`,
			`  "year"`,
			`]`,
		]);
	});

	it("updates tag unit caches after Python-driven tag and untag calls", async () => {
		const result = await run([
			`print(count(entries[py_code] of G0))`,
			`ids = retrieve(top 2 entries of G0)`,
			`tag(ids with py_code set to "seen")`,
			`print(count(entries[py_code] of G0))`,
			`print(retrieve(top 3 entries[py_code] of G0))`,
			`untag(py_code from ids)`,
			`print(count(entries[py_code] of G0))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"0",
			"Tagged 2 entries with 2 tag values across 1 field: py_code.",
			"2",
			`[`,
			`  "seen",`,
			`  "seen"`,
			`]`,
			"Removed 2 tag fields from 2 entries across 1 field: py_code.",
			"0",
		]);
	});

	it("reports scope errors for mixed entry and field operations", async () => {
		const result = await run([
			`print(retrieve(top 1 fields of G0))`,
			`print(count(entries of G1))`,
			`print(count(entries of [${DATASET_SLUG}:000001, "text"]))`,
			`print(count(entries of (scope: G0, scope: G1, length > 3)))`,
			`tag(G1 with code set to "x")`,
			`untag(code from G1)`,
			`print("after-scope-errors")`,
		].join("\n"));

		expect(result.output.trim()).toBe("after-scope-errors");
		expect(result.errors).toMatchObject([
			{ code: "E_UNIT_SCOPE", line: 1 },
			{ code: "E_UNIT_SCOPE", line: 2 },
			{ code: "E_GROUP_LIST_SCOPE", line: 3 },
			{ code: "E_GROUP_SCOPE", line: 4 },
			{ code: "E_TAG_SCOPE", line: 5 },
			{ code: "E_UNTAG_SCOPE", line: 6 },
		]);
	});

	it("reports regex unit errors while continuing later DSL statements", async () => {
		const result = await run([
			`print(retrieve(top 1 entries.find("") of G0))`,
			`print(retrieve(top 1 entries.unknown(r"x") of G0))`,
			`print("after-regex-errors")`,
		].join("\n"));

		expect(result.output.trim()).toBe("after-regex-errors");
		expect(result.errors).toMatchObject([
			{ code: "E_REGEX", line: 1 },
			{ code: "E_REGEX", line: 2 },
		]);
	});

	it("supports core Python control flow, data structures, comprehensions, and Quail calls", async () => {
		const result = await run([
			"def collect_ids():",
			"    ids = []",
			"    total = count(entries of G0)",
			"    for index, entry_id in enumerate(retrieve(top 4 entries of G0)):",
			`        if index < total and (entry_id.endswith("000001") or entry_id.endswith("000003")):`,
			"            ids.append(entry_id)",
			"    return ids",
			"",
			"ids = collect_ids()",
			"class Holder:",
			"    def __init__(self, values):",
			"        self.values = values",
			"    def first(self):",
			"        return self.values[0]",
			"holder = Holder(ids)",
			"coords = (1, 2)",
			"left, right = coords",
			"unique = {entry_id for entry_id in ids}",
			"total_from_generator = sum(value for value in coords)",
			`summary = {"total": count(entries of G0), "ids": [entry_id for entry_id in ids], "ok": True, "missing": None}`,
			"print(summary)",
			"print(len(ids))",
			"print(holder.first())",
			"print(left + right, len(unique), total_from_generator)",
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			`{"total":4,"ids":["${DATASET_SLUG}:000001","${DATASET_SLUG}:000003"],"ok":true,"missing":null}`,
			"2",
			`${DATASET_SLUG}:000001`,
			"3 2 3",
		]);
	});

	it("supports get and count_by inside Python-routed code", async () => {
		const result = await run([
			`ids = retrieve(top 2 entries of G0)`,
			`rows = [(entry.fields["year"], entry.fields["text"]) for entry in get(ids)]`,
			`print(rows)`,
			`print(count_by(["year"] of G0)[0]["year"])`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			`[[2024,"Alpha apple first"],[2025,"Beta banana second"]]`,
			"2024",
		]);
	});

	it("allows command bodies to come from string variables", async () => {
		const result = await run([
			`retrieval = 'top 2 entries of G0 sorted by ([text] BM25 similarity to "banana")'`,
			`ids = retrieve(retrieval)`,
			`print(ids[0], ids[1])`,
			`count_body = 'entries of G0'`,
			`print(count(count_body))`,
			`count_by_body = '["year"] of G0'`,
			`print(count_by(count_by_body)[0]["year"])`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			`${DATASET_SLUG}:000002 ${DATASET_SLUG}:000001`,
			"4",
			"2024",
		]);
	});

	it("supports list concatenation in Python-routed get arguments", async () => {
		const result = await run([
			`a = retrieve(top 1 entries of G0)`,
			`b = retrieve(bottom 1 entries of G0)`,
			`if True:`,
			`    rows = get(a + b)`,
			`    print(rows[0].id, rows[1].id)`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim()).toBe(`${DATASET_SLUG}:000001 ${DATASET_SLUG}:000004`);
	});

	it("preserves Python methods whose names overlap Quail commands", async () => {
		const result = await run([
			`ids = retrieve(top 2 entries of G0)`,
			`rows = get(ids)`,
			`fields = rows[0].fields`,
			`print(fields.get("year"), fields.get("missing", "fallback"))`,
			`values = ["apple", "banana", "apple"]`,
			`summary = {"count": 3, "retrieve": "ok"}`,
			`print(values.count("apple"), "banana".count("a"), summary.get("count"), summary.get("retrieve"))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"2024 fallback",
			"2 3 3 ok",
		]);
	});

	it("accepts scoped group expressions in count_by and score distributions", async () => {
		const result = await run([
			`print(count_by(["year"] of (scope: G0, ([year] == 2024))))`,
			`var dist = get((BM25: ["text": "apple"]) distribution of (scope: G0, ([year] == 2024)))`,
			`if dist.max > 0:`,
			`    print("distribution scoped ok")`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			`[{"year":2024,"count":2}]`,
			"distribution scoped ok",
		]);
	});

	it("routes standalone Python tuple, set, generator, and unpacking syntax to Python", async () => {
		const result = await run([
			"coords = (1, 2)",
			"left, right = coords",
			"labels = {\"alpha\", \"beta\"}",
			"total = sum(value for value in coords)",
			"print(left + right, len(labels), total)",
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim()).toBe("3 2 3");
	});

	it("keeps Python runtime internals out of dir output", async () => {
		const result = await run([
			"values = [x for x in dir() if x.startswith('_') or x in ['REQUEST_PREFIX', 'DONE_PREFIX', '_request']]",
			"print(values)",
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim()).toBe("[]");
	});

	it("persists Python bindings but not ordinary variables across sequential calls", async () => {
		const state = createEmptyAnalysisState();
		const first = await executeQuailCallBlocks({
			cwd,
			state,
			blocks: [{ datasets: [DATASET_NAME], code: [
				"def persisted_ids():",
				"    return retrieve(top 2 entries of G0)",
				"class Offset:",
				"    def __init__(self, amount):",
				"        self.amount = amount",
				"    def add(self, value):",
				"        return self.amount + value",
				`counter = {"runs": 1, "ids": retrieve(top 2 entries of G0)}`,
				"labels = []",
				`labels.append("warm")`,
				`saved_counter = {"runs": 1, "label": "kept"}`,
				"save(saved_counter)",
				"print(counter)",
				"print(labels)",
			].join("\n"), raw: "" }],
		});
		expect(first.errors).toEqual([]);
		expect(first.output.trim().split("\n")).toEqual([
			`{"runs":1,"ids":["${DATASET_SLUG}:000001","${DATASET_SLUG}:000002"]}`,
			`["warm"]`,
		]);

			const second = await executeQuailCallBlocks({
				cwd,
				state: first.state,
				blocks: [{ datasets: [DATASET_NAME], code: [
					"try:",
					"    print(counter)",
					"except NameError:",
					`    print("counter missing")`,
					`print(saved_counter["runs"], saved_counter["label"])`,
					"print(persisted_ids()[1])",
					"offset = Offset(2)",
					"print(offset.add(3))",
				].join("\n"), raw: "" }],
			});

			expect(second.errors).toEqual([]);
			expect(second.output.trim().split("\n")).toEqual([
				"counter missing",
				"1 kept",
				`${DATASET_SLUG}:000002`,
				"5",
			]);
		});

	it("persists explicitly saved DSL variables across sequential calls", async () => {
		const state = createEmptyAnalysisState();
		const first = await executeQuailCallBlocks({
			cwd,
			state,
			blocks: [{ datasets: [DATASET_NAME], code: [
				`var saved_ids = retrieve(top 2 entries of G0)`,
				`var transient_ids = retrieve(bottom 1 entries of G0)`,
				`save(saved_ids)`,
				`print(saved_ids[0], saved_ids[1])`,
			].join("\n"), raw: "" }],
		});

		expect(first.errors).toEqual([]);
		expect(first.output.trim()).toBe(`${DATASET_SLUG}:000001 ${DATASET_SLUG}:000002`);

		const second = await executeQuailCallBlocks({
			cwd,
			state: first.state,
			blocks: [{ datasets: [DATASET_NAME], code: [
				`print(saved_ids[0], saved_ids[1])`,
				`print(type(transient_ids))`,
			].join("\n"), raw: "" }],
		});

		expect(second.errors).toEqual([]);
		expect(second.output.trim().split("\n")).toEqual([
			`${DATASET_SLUG}:000001 ${DATASET_SLUG}:000002`,
			"string",
		]);
	});

	it("rejects save() for missing or non-variable expressions", async () => {
		const result = await run([
			`save(missing_value)`,
			`save(retrieve(top 1 entries of G0))`,
			`print("after")`,
		].join("\n"));

		expect(result.output.trim()).toBe("after");
		expect(result.errors).toMatchObject([
			{ code: "E_SAVE_VARIABLE", line: 1 },
			{ code: "E_PARSE_SAVE", line: 2 },
		]);
	});

	it("rejects oversized saved variables before they poison later Python runs", async () => {
		const previous = process.env.QUAIL_MAX_SAVED_VARIABLE_BYTES;
		process.env.QUAIL_MAX_SAVED_VARIABLE_BYTES = "64";
		try {
			const state = createEmptyAnalysisState();
			const first = await executeQuailCallBlocks({
				cwd,
				state,
				blocks: [{ datasets: [DATASET_NAME], code: [
					`var too_big = ["xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"]`,
					`save(too_big)`,
					`print("after")`,
				].join("\n"), raw: "" }],
			});
			expect(first.output.trim()).toBe("after");
			expect(first.errors).toMatchObject([{ code: "E_SAVE_VALUE", line: 2 }]);
			expect(first.errors[0]?.message).toContain("save() limit");

			const second = await executeQuailCallBlocks({
				cwd,
				state: first.state,
				blocks: [{ datasets: [DATASET_NAME], code: [
					`if True:`,
					`    print("python still starts")`,
				].join("\n"), raw: "" }],
			});
			expect(second.errors).toEqual([]);
			expect(second.output.trim()).toBe("python still starts");
		} finally {
			if (previous === undefined) delete process.env.QUAIL_MAX_SAVED_VARIABLE_BYTES;
			else process.env.QUAIL_MAX_SAVED_VARIABLE_BYTES = previous;
		}
	});

	it("runs large Python programs from a temp file instead of argv", async () => {
		const payload = "x".repeat(300_000);
		const result = await run([
			`payload = "${payload}"`,
			`if True:`,
			`    print(len(payload))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim()).toBe("300000");
	});

	it("rejects Python imports with a model-repairable error", async () => {
		const result = await run([
			"import os",
			`print("should-not-run")`,
		].join("\n"));

		expect(result.output).toBe("");
		expect(result.errors).toMatchObject([{ code: "E_IMPORT_DISABLED" }]);
	});

	it("invalidates scoped group caches when Python bindings update locals", async () => {
		const result = await run([
			"def force_python_runtime():",
			"    return None",
			"for threshold in [0.0, 999.0]:",
			`    print(count(entries of (scope: G0, ([text] BM25 similarity to "apple" > threshold))))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual(["2", "0"]);
	});

	it("converts expression values to strings", async () => {
		const result = await run([
			"var total = count(G0)",
			"print(\"total=\" + str(total))",
			"print(str([\"alpha\", total]))",
			`tag("${DATASET_SLUG}:000001" with "sample_count" set to str(total))`,
			`print(get("${DATASET_SLUG}:000001").tags["sample_count"])`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"total=4",
			"[\"alpha\",4]",
			"Tagged 1 entry with 1 tag value across 1 field: sample_count.",
			"4",
		]);
	});

	it("tags group expressions", async () => {
		const result = await run([
			`tag(contains: "apple" with "assigned_primary_problem_cycle_01" set to "Aircraft")`,
			`print(count(tags: ["assigned_primary_problem_cycle_01": "Aircraft"]))`,
			`print(get("${DATASET_SLUG}:000001").tags["assigned_primary_problem_cycle_01"])`,
			`print(get("${DATASET_SLUG}:000003").tags["assigned_primary_problem_cycle_01"])`,
			`print(len(get("${DATASET_SLUG}:000002").tags))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"Tagged 2 entries with 2 tag values across 1 field: assigned_primary_problem_cycle_01.",
			"2",
			"Aircraft",
			"Aircraft",
			"2",
		]);
	});

	it("adds, removes, filters, and lists tag values on unified fields", async () => {
		const result = await run([
			`tag("${DATASET_SLUG}:000001" with "codes" set to ["attention", "communication"])`,
			`tag("${DATASET_SLUG}:000001" with "codes" add ["attention", "fatigue"])`,
			`tag("${DATASET_SLUG}:000002" with "codes" set to "fatigue")`,
			`tag("${DATASET_SLUG}:000002" with "year" set to "manual-tag")`,
			`print(get("${DATASET_SLUG}:000001").tags["codes"])`,
			`print(get(tag_fields))`,
			`print(get(["codes"]))`,
			`print(get(["year"]))`,
			`print(count(tags: ["codes": "fatigue"]))`,
			`untag("${DATASET_SLUG}:000001" with "codes" remove "attention")`,
			`print(get("${DATASET_SLUG}:000001").tags["codes"])`,
			`untag("codes" from "${DATASET_SLUG}:000001")`,
			`print(type(get("${DATASET_SLUG}:000001").tags["codes"]))`,
			`print(count(tags: ["codes": "communication"]))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"Tagged 2 entries with 6 tag values across 2 fields: codes, year.",
			`[`,
			`  "attention",`,
			`  "communication",`,
			`  "fatigue"`,
			`]`,
			`[`,
			`  "codes",`,
			`  "text",`,
			`  "year"`,
			`]`,
			`[`,
			`  "attention",`,
			`  "communication",`,
			`  "fatigue"`,
			`]`,
			`[`,
			`  2024,`,
			`  2026,`,
			`  "manual-tag"`,
			`]`,
			"2",
			"Removed 1 tag value from 1 entry across 1 field: codes.",
			`[`,
			`  "communication",`,
			`  "fatigue"`,
			`]`,
			"Removed 1 tag field from 1 entry across 1 field: codes.",
			"undefined",
			"0",
		]);
	});

	it("persists saved groups and tags but not ordinary variables across sequential Quail calls", async () => {
		const state = createEmptyAnalysisState();
		const first = await executeQuailCallBlocks({
			cwd,
			state,
			blocks: [{ datasets: [DATASET_NAME], code: [
				`var apple = group(contains: ["text": "apple"])`,
				`var running_total = count(apple)`,
				`tag(apple with "review_code" set to "fruit")`,
				`print(apple, running_total)`,
			].join("\n"), raw: "" }],
		});
		expect(first.errors).toEqual([]);
		expect(first.output.trim().split("\n")).toEqual([
			"Tagged 2 entries with 2 tag values across 1 field: review_code.",
			"G2 2",
		]);

		const second = await executeQuailCallBlocks({
			cwd,
			state: first.state,
			blocks: [{ datasets: [DATASET_NAME], code: [
				`print(get(groups))`,
				`print(get(G2).id, get(G2).entryIds[0], get(G2).entryIds[1])`,
				`print(count(G2), count(tags: ["review_code": "fruit"]))`,
				`print(type(running_total))`,
			].join("\n"), raw: "" }],
		});

		expect(second.errors).toEqual([]);
		expect(second.output.trim().split("\n")).toEqual([
			`[`,
			`  "G2"`,
			`]`,
			`G2 ${DATASET_SLUG}:000001 ${DATASET_SLUG}:000003`,
			"2 2",
			"string",
		]);
	});

	it("handles assignment operators, arithmetic, list updates, and conditional membership", async () => {
		const result = await run([
			`var n = 10`,
			`n += 2`,
			`n -= 5`,
			`var label = "score-"`,
			`label += str(n)`,
			`var ids = [${DATASET_SLUG}:000001, ${DATASET_SLUG}:000002]`,
			`ids += [${DATASET_SLUG}:000003]`,
			`ids -= ${DATASET_SLUG}:000002`,
			`print(label, n * 3, n / 2, +n, -n, len(ids))`,
			`if ${DATASET_SLUG}:000003 in ids and ${DATASET_SLUG}:000002 not in ids:`,
			`    print("membership-ok")`,
			`else:`,
			`    print("membership-bad")`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"score-7 21 3.5 7 -7 2",
			"membership-ok",
		]);
	});

	it("treats include-only group specs as explicit id sets", async () => {
		const result = await run([
			`print(count(temp(include: [${DATASET_SLUG}:000001])))`,
			`print(count(temp(include: [${DATASET_SLUG}:000001, ${DATASET_SLUG}:000002], exclude: [${DATASET_SLUG}:000002])))`,
			`print(count(temp(contains_word: "apple", include: [${DATASET_SLUG}:000002])))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"1",
			"1",
			"3",
		]);
	});

	it("accepts retrieved id lists in include and exclude specs", async () => {
		const result = await run([
			"var res = retrieve(top 3 entries of G0)",
			"print(count(temp(include: res)))",
			"print(count_by([\"year\"] of temp(include: res)))",
			"print(count(temp(include: retrieve(top 2 entries of G0))))",
			"print(count(temp(include: res, exclude: res[1:3])))",
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"3",
			`[`,
			`  {`,
			`    "year": 2024,`,
			`    "count": 2`,
			`  },`,
			`  {`,
			`    "year": 2025,`,
			`    "count": 1`,
			`  }`,
			`]`,
			"2",
			"1",
		]);
	});

	it("supports string/list slices and len", async () => {
		const result = await run([
			`var text = get("${DATASET_SLUG}:000001").text`,
			"print(text[0:5])",
			"print(text[:5])",
			"print(text[6:])",
			"print(text[-5:])",
			"print(len(text), len(text[0:5]), len([1, 2, 3]))",
			"var ids = retrieve(top 3 entries of G0)",
			"print(ids[0:2][0], ids[0:2][1], ids[-1])",
			`print(len(get("${DATASET_SLUG}:000001").tags))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
			expect(result.output.trim().split("\n")).toEqual([
				"Alpha",
				"Alpha",
				"apple first",
				"first",
				"17 5 3",
				`${DATASET_SLUG}:000001 ${DATASET_SLUG}:000002 ${DATASET_SLUG}:000003`,
				"2",
			]);
		});

	it("retrieves and indexes more than 20 entries when requested", async () => {
		const inputPath = join(cwd, "large.csv");
		writeFileSync(
			inputPath,
			[
				"text,year",
				...Array.from({ length: 45 }, (_, index) => `Row ${index + 1},2026`),
			].join("\n"),
			"utf8",
		);
		await processDataset({
			cwd,
			inputPath,
			name: "Large DSL Check",
			skipEmbeddings: true,
		});

		const result = await executeQuailCallBlocks({
			cwd,
			state: createEmptyAnalysisState(),
			blocks: [{ datasets: ["Large DSL Check"], code: [
				"var ids = retrieve(top 40 entries of G0)",
				"print(len(ids), ids[19], ids[20], ids[39])",
				"for i in [20, 21, 39]:",
				"    print(get(ids[i]).text)",
			].join("\n"), raw: "" }],
		});

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"40 large-dsl-check:000020 large-dsl-check:000021 large-dsl-check:000040",
			"Row 21",
			"Row 22",
			"Row 40",
		]);
	});

	it("handles tag-backed fields without duplicate counts or field-index misses", async () => {
		const datasetsDir = getQuailDatasetsDir(cwd);
		const datasetDir = join(datasetsDir, "tag-backed-fields");
		mkdirSync(datasetDir, { recursive: true });
		writeFileSync(join(datasetDir, "manifest.json"), JSON.stringify({
			name: "Tag Backed Fields",
			slug: "tag-backed-fields",
			createdAt: "2026-05-11T00:00:00.000Z",
			updatedAt: "2026-05-11T00:00:00.000Z",
			entryCount: 2,
			metadataFields: ["party", "year"],
			embeddingModel: "embeddinggemma:latest",
			embeddingDimensions: 0,
			batchSize: 64,
			source: { format: "tag-backed-test" },
			files: { entries: "entries.jsonl", bm25: "bm25.json", embeddings: "embeddings.json" },
		}, null, 2), "utf8");
		writeFileSync(join(datasetDir, "entries.jsonl"), [
			JSON.stringify({
					id: "tag-backed-fields:000001",
					dataset: "Tag Backed Fields",
				ordinal: 1,
				text: "Freedom for women and environment",
				tags: { party: "democrat", year: "2020" },
				contains: "freedom for women and environment",
			}),
			JSON.stringify({
					id: "tag-backed-fields:000002",
					dataset: "Tag Backed Fields",
				ordinal: 2,
				text: "Taxes defense and agriculture",
				tags: { party: "republican", year: "2024" },
				contains: "taxes defense and agriculture",
			}),
		].join("\n") + "\n", "utf8");
		writeFileSync(join(datasetDir, "bm25.json"), JSON.stringify({
			k1: 1.5,
			b: 0.75,
			avgDocLength: 4.5,
			docCount: 2,
				docLengths: { "tag-backed-fields:000001": 5, "tag-backed-fields:000002": 4 },
				docFreq: { freedom: 1, for: 1, women: 1, and: 2, environment: 1, taxes: 1, defense: 1, agriculture: 1 },
				termFreq: {
					"tag-backed-fields:000001": { freedom: 1, for: 1, women: 1, and: 1, environment: 1 },
					"tag-backed-fields:000002": { taxes: 1, defense: 1, and: 1, agriculture: 1 },
			},
		}, null, 2), "utf8");
		writeFileSync(join(datasetDir, "embeddings.json"), JSON.stringify({
			model: "embeddinggemma:latest",
			dimensions: 0,
			vectors: {},
		}, null, 2), "utf8");

		const result = await executeQuailCallBlocks({
			cwd,
			state: createEmptyAnalysisState(),
				blocks: [{ datasets: ["Tag Backed Fields"], code: [
				`print(count_by(["party"] of G0))`,
				`print(count(temp(contains_word: ["text": "freedom"])))`,
				`print(count(temp(BM25: ["text": "freedom"] > 0)))`,
				`var max_score = get((BM25: ["text": "freedom"]) distribution of (G0)).max`,
				`if max_score > 0:`,
				`    print("distribution ok")`,
				`var bare_max_score = get((BM25: ["text": "freedom"]) distribution of G0).max`,
				`if bare_max_score > 0:`,
				`    print("distribution bare ok")`,
					`var include_max_score = get((BM25: ["text": "freedom"]) distribution of temp(include: [tag-backed-fields:000001])).max`,
				`if include_max_score > 0:`,
				`    print("distribution temp ok")`,
				`var freedom = temp(contains_word: ["text": "freedom"])`,
				`print(count(freedom))`,
				`var multi = group_expr(`,
				`    temp(contains_word: ["text": "freedom"]) or`,
				`    temp(contains_word: ["text": "taxes"])`,
				`)`,
				`print(count(multi))`,
				`print(count(group_expr(temp(contains_word: ["text": "freedom"]) and G0)))`,
				`for threshold in [0.0]:`,
				`    print(count(temp(BM25: ["text": "freedom"] > threshold)))`,
					`if type(get(["year"])[0]) == "string" and get(["year"])[0] not in []:`,
					`    print("and ok")`,
				`if count(freedom) > 0 or count(temp(contains_word: ["text": "missing"])) > 0:`,
				`    print("or ok")`,
					`for entry in [tag-backed-fields:000001, tag-backed-fields:000002]:`,
				`    if get(entry).fields["party"] == "democrat":`,
				`        print("dem")`,
				`    else:`,
				`        print("other")`,
			].join("\n"), raw: "" }],
		});

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			`[{"party":"democrat","count":1},{"party":"republican","count":1}]`,
			"1",
			"1",
			"distribution ok",
			"distribution bare ok",
			"distribution temp ok",
			"1",
			"2",
			"1",
			"1",
			"and ok",
			"or ok",
			"dem",
			"other",
		]);
	});

	it("exposes processed fields and typed field values", async () => {
		const result = await run([
			`print(get("${DATASET_SLUG}:000001").fields["text"])`,
			`print(get("${DATASET_SLUG}:000001").fields["year"])`,
			`print(type(get("${DATASET_SLUG}:000001").fields["year"]))`,
			`print(get(fields)[0], get(fields)[1])`,
			`print(get(text_fields)[0])`,
			`var by_year = count_by(["year"] of G0)`,
			`print(by_year[0]["year"], by_year[0]["count"])`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"Alpha apple first",
			"2024",
			"int",
			"text year",
			"text",
			"2024 2",
		]);
	});

	it("uses field types to decide which fields are embedded and text-searchable", async () => {
		const manifestPath = join(getQuailDatasetsDir(cwd), DATASET_SLUG, "manifest.json");
		const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {
			fieldTypes: Record<string, string>;
			embeddedFields: string[];
		};
		expect(manifest.fieldTypes).toMatchObject({ text: "string", year: "int" });
		expect(manifest.embeddedFields).toEqual(["text"]);

		const result = await run([
			`print(count(temp(contains: ["year": "2024"])))`,
			`print(count(temp(BM25: ["year": "2024"] > 0)))`,
			`print(count(temp(fields_compare: ["year": ["==", 2024]])))`,
			`print(count(temp(fields_compare: ["year": [">", 2024]])))`,
			`print(count(temp(fields_compare: ["year": ["!=", 2024]])))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual(["0", "0", "2", "2", "2"]);
	});

	it("reuses score vectors, threshold masks, and field comparisons across threshold sweeps", async () => {
		clearQuailDslRuntimeCaches();
		const result = await run([
			`var q = "apple"`,
			`var pos = temp(fields_compare: ["year": ["==", 2024]])`,
			`for outer in [1, 2]:`,
			`    for t in [0.0, 0.1, 0.0]:`,
			`        var g = temp(BM25: ["text": q] > t)`,
			`        print(count(g), count(g and pos))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"2 2",
			"2 2",
			"2 2",
			"2 2",
			"2 2",
			"2 2",
		]);
		const stats = getQuailDslRuntimeCacheStats();
		expect(stats.scoreVectorMisses).toBe(1);
		expect(stats.scoreVectorHits).toBeGreaterThan(0);
		expect(stats.thresholdIdSetMisses).toBe(2);
		expect(stats.thresholdIdSetHits).toBeGreaterThan(0);
		expect(stats.fieldComparisonMisses).toBe(1);
		expect(stats.fieldComparisonHits).toBeGreaterThan(0);
	});

	it("keeps new scoped group syntax on the optimized score-vector and threshold-cache path", async () => {
		clearQuailDslRuntimeCaches();
		const result = await run([
			`for threshold in [0.0, 0.1, 0.0]:`,
			`    print(count(entries of (scope: G0, ([text] BM25 similarity to "apple" > threshold))))`,
			`    print(count(entries of (scope: G0, ([text] BM25 similarity to "apple" > threshold))))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"2",
			"2",
			"2",
			"2",
			"2",
			"2",
		]);
		const stats = getQuailDslRuntimeCacheStats();
		expect(stats.scoreVectorMisses).toBe(1);
		expect(stats.scoreVectorHits).toBeGreaterThan(0);
		expect(stats.thresholdIdSetMisses).toBe(2);
		expect(stats.thresholdIdSetHits).toBeGreaterThan(0);
	});

	it("bulk scores file-backed embedding vectors in entry order", () => {
		const previousLimit = process.env.QUAIL_EAGER_VECTOR_FILE_BYTES;
		process.env.QUAIL_EAGER_VECTOR_FILE_BYTES = "1";
		try {
			const datasetsDir = getQuailDatasetsDir(cwd);
			const datasetDir = join(datasetsDir, "binary-embeddings");
			mkdirSync(datasetDir, { recursive: true });
			writeFileSync(join(datasetDir, "manifest.json"), JSON.stringify({
				name: "Binary Embeddings",
				slug: "binary-embeddings",
				createdAt: "2026-05-13T00:00:00.000Z",
				updatedAt: "2026-05-13T00:00:00.000Z",
				entryCount: 3,
				metadataFields: ["text"],
				fieldNames: ["text"],
				textFields: ["text"],
				fieldTypes: { text: "string" },
				embeddedFields: ["text"],
				embeddingModel: "test-model",
				embeddingDimensions: 2,
				batchSize: 64,
				source: { format: "test" },
				files: { entries: "entries.jsonl", bm25: "bm25.json", embeddings: "embeddings.json" },
			}, null, 2), "utf8");
			writeFileSync(join(datasetDir, "entries.jsonl"), [
				JSON.stringify({ id: "binary-embeddings:000001", dataset: "Binary Embeddings", ordinal: 1, text: "east", fields: { text: "east" }, tags: {}, contains: "east", fieldContains: { text: "east" } }),
				JSON.stringify({ id: "binary-embeddings:000002", dataset: "Binary Embeddings", ordinal: 2, text: "north", fields: { text: "north" }, tags: {}, contains: "north", fieldContains: { text: "north" } }),
				JSON.stringify({ id: "binary-embeddings:000003", dataset: "Binary Embeddings", ordinal: 3, text: "diagonal", fields: { text: "diagonal" }, tags: {}, contains: "diagonal", fieldContains: { text: "diagonal" } }),
			].join("\n") + "\n", "utf8");
			writeFileSync(join(datasetDir, "bm25.json"), JSON.stringify({
				k1: 1.5,
				b: 0.75,
				avgDocLength: 1,
				docCount: 3,
				docLengths: {},
				docFreq: {},
				termFreq: {},
			}), "utf8");
			writeFileSync(join(datasetDir, "embeddings.json"), JSON.stringify({
				format: "float32-binary-v1",
				model: "test-model",
				dimensions: 2,
				count: 3,
				ids: ["binary-embeddings:000001", "binary-embeddings:000002", "binary-embeddings:000003"],
				vectorsFile: "embeddings.f32",
			}), "utf8");
			const buffer = Buffer.allocUnsafe(3 * 2 * 4);
			[1, 0, 0, 1, 0.6, 0.8].forEach((value, index) => buffer.writeFloatLE(value, index * 4));
			writeFileSync(join(datasetDir, "embeddings.f32"), buffer);
			const dataset = loadDatasets(cwd, ["Binary Embeddings"])[0];
			const scores = scoreEmbeddingVectorValues([
				dataset.embeddings.vectors["binary-embeddings:000001"],
				dataset.embeddings.vectors["binary-embeddings:000002"],
				dataset.embeddings.vectors["binary-embeddings:000003"],
			], [1, 0]);
			expect([...scores].map((score) => Number(score.toFixed(3)))).toEqual([1, 0, 0.6]);
		} finally {
			if (previousLimit === undefined) delete process.env.QUAIL_EAGER_VECTOR_FILE_BYTES;
			else process.env.QUAIL_EAGER_VECTOR_FILE_BYTES = previousLimit;
		}
	});

	it("ranks entries against group and entry-list embedding targets without query embedding calls", async () => {
		writeSemanticEmbeddingDataset();
		clearQuailDslRuntimeCaches();

		const result = await executeQuailCallBlocks({
			cwd,
			state: createEmptyAnalysisState(),
			blocks: [{ datasets: ["Semantic Groups"], code: [
				`ids = retrieve(top 2 entries of G0 sorted by ([text] avg per avg embed similarity to G0))`,
				`print(ids[0], ids[1])`,
				`target = retrieve(top 2 entries of G0)`,
				`print(retrieve(top 1 entries of G0 sorted by ([text] avg per avg embed similarity to target))[0])`,
				`target_group = group_expr([semantic-groups:000001, semantic-groups:000002])`,
				`print(retrieve(top 2 entries of G0 sorted by ([text] avg per avg embed similarity to target_group))[1])`,
			].join("\n"), raw: "" }],
		});

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"semantic-groups:000001 semantic-groups:000002",
			"semantic-groups:000001",
			"semantic-groups:000002",
		]);
		const stats = getQuailDslRuntimeCacheStats();
		expect(stats.queryEmbeddingMisses).toBe(0);
		expect(stats.queryEmbeddingHits).toBe(0);
		expect(stats.scoreVectorMisses).toBe(2);
		expect(stats.scoreVectorHits).toBeGreaterThan(0);
	});

	it("inspects field types and supports field type overrides before processing", async () => {
		const inputPath = join(cwd, "typed.csv");
		writeFileSync(
			inputPath,
			[
				"year,score,complete,notes",
				"2024,3.5,true,First answer",
				"2025,4.25,false,Second answer",
			].join("\n"),
			"utf8",
		);

		const inspection = inspectDatasetFile({ inputPath });
		expect(inspection.fieldTypes).toMatchObject({
			year: "int",
			score: "float",
			complete: "bool",
			notes: "string",
		});
		expect(inspection.embeddedFields).toEqual(["notes"]);

		const overridden = inspectDatasetFile({ inputPath, fieldTypes: { year: "string" } });
		expect(overridden.fieldTypes.year).toBe("string");
		expect(overridden.embeddedFields).toEqual(["notes", "year"]);

		await processDataset({
			cwd,
			inputPath,
			name: "Override Check",
			fieldTypes: { year: "string" },
			skipEmbeddings: true,
		});

		const result = await executeQuailCallBlocks({
			cwd,
			state: createEmptyAnalysisState(),
			blocks: [{ datasets: ["Override Check"], code: [
				`print(type(get("override-check:000001").fields["year"]))`,
				`print(count(temp(contains: ["year": "2024"])))`,
			].join("\n"), raw: "" }],
		});

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual(["string", "1"]);
	});

	it("filters and retrieves using explicit fields", async () => {
		const result = await run([
			`print(count(entries of (scope: G0, ([year] == 2024))))`,
			`print(count(entries of (scope: G0, ([year] == "2024"))))`,
			`print(count(temp(fields_compare: ["year": ["==", 2024]])))`,
			`print(count(temp(contains: ["text": "apple"])))`,
			`print(count(temp(contains: ["text": "apple"], contains: ["text": "Gamma"])))`,
			`print(count(temp(BM25: ["text": "apple"] > 0, BM25: ["text": "Gamma"] > 0)))`,
			`print(count(temp(contains: ["text": "banana"]) or temp(contains: ["text": "final"])))`,
			`var ranked = retrieve(top 1 entries of G0 sorted by ([text] BM25 similarity to "banana"))`,
			`print(ranked[0])`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"2",
			"2",
			"2",
			"2",
			"1",
			"1",
			"2",
			`${DATASET_SLUG}:000002`,
		]);
	});

	it("supports middle and bottom retrieval for ranked and unranked scopes", async () => {
		const result = await run([
			`print(retrieve(middle 2 entries of G0)[0], retrieve(middle 2 entries of G0)[1])`,
			`print(retrieve(bottom 2 entries of G0)[0], retrieve(bottom 2 entries of G0)[1])`,
			`var ranked_bottom = retrieve(bottom 2 entries of G0 sorted by ([text] BM25 similarity to "apple"))`,
			`print(ranked_bottom[0], ranked_bottom[1])`,
			`var ranked_middle = retrieve(middle 2 entries of G0 sorted by ([text] BM25 similarity to "apple"))`,
			`print(len(ranked_middle))`,
			`if ranked_middle[0] in retrieve(top 4 entries of G0):`,
			`    print("ranked-middle-in-scope")`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			`${DATASET_SLUG}:000002 ${DATASET_SLUG}:000003`,
			`${DATASET_SLUG}:000004 ${DATASET_SLUG}:000003`,
			`${DATASET_SLUG}:000004 ${DATASET_SLUG}:000002`,
			"2",
			"ranked-middle-in-scope",
		]);
	});

	it("runs over multiple datasets while preserving stable dataset-qualified ids", async () => {
		const inputPath = join(cwd, "second.csv");
		writeFileSync(
			inputPath,
			[
				"text,year",
				"Epsilon apple fifth,2027",
				"Zeta banana sixth,2028",
			].join("\n"),
			"utf8",
		);
		await processDataset({
			cwd,
			inputPath,
			name: "Second DSL Check",
			skipEmbeddings: true,
		});

		const result = await executeQuailCallBlocks({
			cwd,
			state: createEmptyAnalysisState(),
			blocks: [{ datasets: [DATASET_NAME, "Second DSL Check"], code: [
				`print(count(G0))`,
				`print(count(temp(contains: ["text": "apple"])))`,
				`var ids = retrieve(top 10 entries of temp(contains: ["text": "banana"]))`,
				`print(len(ids), ids[0], ids[1])`,
				`print(count_by(["year"] of temp(include: [second-dsl-check:000001, ${DATASET_SLUG}:000001])))`,
			].join("\n"), raw: "" }],
		});

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"6",
			"3",
			`2 ${DATASET_SLUG}:000002 second-dsl-check:000002`,
			`[`,
			`  {`,
			`    "year": 2024,`,
			`    "count": 1`,
			`  },`,
			`  {`,
			`    "year": 2027,`,
			`    "count": 1`,
			`  }`,
			`]`,
		]);
	});

	it("keeps quoted multiline CSV cells in one record", async () => {
		const inputPath = join(cwd, "survey.csv");
		writeFileSync(
			inputPath,
			[
				"timestamp,motivation,year",
				`2025-01-01,"First line`,
				`second career line",2025`,
				`2025-01-02,"Different answer",2026`,
			].join("\n"),
			"utf8",
		);
		await processDataset({
			cwd,
			inputPath,
			name: "Survey Check",
			skipEmbeddings: true,
			overwrite: true,
		});

		const result = await executeQuailCallBlocks({
			cwd,
			state: createEmptyAnalysisState(),
			blocks: [{ datasets: ["Survey Check"], code: [
				"print(count(G0))",
				`print(get("survey-check:000001").fields["motivation"])`,
				`print(count(temp(contains: ["motivation": "career"])))`,
			].join("\n"), raw: "" }],
		});

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"2",
			"First line",
			"second career line",
			"1",
		]);
	});

	it("allows query text to come from nested expressions", async () => {
		const result = await run([
			`var matches = retrieve(top 1 entries of G0 sorted by ([text] BM25 similarity to get("${DATASET_SLUG}:000001").text))`,
			"print(matches[0])",
			`print(count(BM25: get("${DATASET_SLUG}:000001").text))`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			`${DATASET_SLUG}:000001`,
			"2",
		]);
	});

	it("rejects semicolon-combined statements", async () => {
		const result = await run('print("one"); print("two")');

		expect(result.output).toBe("");
		expect(result.errors).toMatchObject([
			{
				code: "E_SEMICOLON",
				line: 1,
			},
		]);
	});

	it("reports model-repairable runtime errors for invalid DSL constructs", async () => {
		const result = await run([
			`print(get(tags))`,
			`print(count(temp(unknown_key: "apple")))`,
			`print(count(temp(include: "${DATASET_SLUG}:000001")))`,
			`print(get("missing-dataset:000001").text)`,
			`print(retrieve(top 1 in (contains: "apple") of G0))`,
			`print(1 / 0)`,
			`print("still-runs")`,
		].join("\n"));

		expect(result.output.trim()).toBe("still-runs");
		expect(result.errors).toMatchObject([
			{ code: "E_GET_TAGS_DISABLED", line: 1 },
			{ code: "E_GROUP_SPEC_KEY", line: 2 },
			{ code: "E_GROUP_SPEC_VALUE", line: 3 },
			{ code: "E_UNKNOWN_ID", line: 4 },
			{ code: "E_PARSE_UNIT", line: 5 },
			{ code: "E_DIVIDE_BY_ZERO", line: 6 },
		]);
	});

	it("rejects old out-of unit syntax with current v0.7 repair guidance", async () => {
		const result = await run([
			`print(retrieve(top 1 entries out of G0))`,
			`print(count(entries out of G0))`,
			`print("after")`,
		].join("\n"));

		expect(result.output.trim()).toBe("after");
		expect(result.errors).toMatchObject([
			{ code: "E_PARSE_RETRIEVE", line: 1 },
			{ code: "E_PARSE_COUNT", line: 2 },
		]);
		expect(result.errors[0]?.message).toContain("uses of, not out of");
		expect(result.errors[1]?.message).toContain("uses of, not out of");
	});
});
