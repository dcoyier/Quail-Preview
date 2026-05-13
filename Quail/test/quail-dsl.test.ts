import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createEmptyAnalysisState } from "../src/quail/analysis-state.js";
import { inspectDatasetFile, processDataset } from "../src/quail/dataset-store.js";
import { clearQuailDslRuntimeCaches, executeQuailCallBlocks, getQuailDslRuntimeCacheStats } from "../src/quail/dsl.js";
import { getQuailDatasetsDir } from "../src/quail/paths.js";

const DATASET_NAME = "DSL Check";
const DATASET_SLUG = "dsl-check";

describe("quail DSL", () => {
	let cwd: string;

	beforeEach(async () => {
		cwd = mkdtempSync(join(tmpdir(), "quail-dsl-"));
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

	it("only returns normal values through print", async () => {
		const result = await run([
			"count(all)",
			"get(tag_fields)",
			"var apple = group(contains: \"apple\")",
			"print(apple, count(apple))",
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim()).toBe("G1 2");
	});

	it("converts expression values to strings", async () => {
		const result = await run([
			"var total = count(all)",
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
			"0",
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
			"var res = retrieve(top 3 of all)",
			"print(count(temp(include: res)))",
			"print(count_by([\"year\"] of temp(include: res)))",
			"print(count(temp(include: retrieve(top 2 of all))))",
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
			"var ids = retrieve(top 3 of all)",
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
			"0",
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
				"var ids = retrieve(top 40 of all)",
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

	it("handles legacy tag-backed fields without duplicate counts or field-index misses", async () => {
		const datasetsDir = getQuailDatasetsDir(cwd);
		const datasetDir = join(datasetsDir, "legacy-fields");
		mkdirSync(datasetDir, { recursive: true });
		writeFileSync(join(datasetDir, "manifest.json"), JSON.stringify({
			name: "Legacy Fields",
			slug: "legacy-fields",
			createdAt: "2026-05-11T00:00:00.000Z",
			updatedAt: "2026-05-11T00:00:00.000Z",
			entryCount: 2,
			metadataFields: ["party", "year"],
			embeddingModel: "embeddinggemma:latest",
			embeddingDimensions: 0,
			batchSize: 64,
			source: { format: "legacy-test" },
			files: { entries: "entries.jsonl", bm25: "bm25.json", embeddings: "embeddings.json" },
		}, null, 2), "utf8");
		writeFileSync(join(datasetDir, "entries.jsonl"), [
			JSON.stringify({
				id: "legacy-fields:000001",
				dataset: "Legacy Fields",
				ordinal: 1,
				text: "Freedom for women and environment",
				tags: { party: "democrat", year: "2020" },
				contains: "freedom for women and environment",
			}),
			JSON.stringify({
				id: "legacy-fields:000002",
				dataset: "Legacy Fields",
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
			docLengths: { "legacy-fields:000001": 5, "legacy-fields:000002": 4 },
			docFreq: { freedom: 1, for: 1, women: 1, and: 2, environment: 1, taxes: 1, defense: 1, agriculture: 1 },
			termFreq: {
				"legacy-fields:000001": { freedom: 1, for: 1, women: 1, and: 1, environment: 1 },
				"legacy-fields:000002": { taxes: 1, defense: 1, and: 1, agriculture: 1 },
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
			blocks: [{ datasets: ["Legacy Fields"], code: [
				`print(count_by(["party"] of all))`,
				`print(count(temp(contains_word: ["text": "freedom"])))`,
				`print(count(temp(BM25: ["text": "freedom"] > 0)))`,
				`var max_score = get((BM25: ["text": "freedom"]) distribution of (all)).max`,
				`if max_score > 0:`,
				`    print("distribution ok")`,
				`var bare_max_score = get((BM25: ["text": "freedom"]) distribution of all).max`,
				`if bare_max_score > 0:`,
				`    print("distribution bare ok")`,
				`var include_max_score = get((BM25: ["text": "freedom"]) distribution of temp(include: [legacy-fields:000001])).max`,
				`if include_max_score > 0:`,
				`    print("distribution temp ok")`,
				`var freedom = temp(contains_word: ["text": "freedom"])`,
				`print(count(freedom))`,
				`var multi = group_expr(`,
				`    temp(contains_word: ["text": "freedom"]) or`,
				`    temp(contains_word: ["text": "taxes"])`,
				`)`,
				`print(count(multi))`,
				`print(count(group_expr(temp(contains_word: ["text": "freedom"]) and all)))`,
				`for threshold in [0.0]:`,
				`    print(count(temp(BM25: ["text": "freedom"] > threshold)))`,
				`if type(get(["year"])[0]) == "int" and get(["year"])[0] not in []:`,
				`    print("and ok")`,
				`if count(freedom) > 0 or count(temp(contains_word: ["text": "missing"])) > 0:`,
				`    print("or ok")`,
				`for entry in [legacy-fields:000001, legacy-fields:000002]:`,
				`    if get(entry).fields["party"] == "democrat":`,
				`        print("dem")`,
				`    else:`,
				`        print("other")`,
			].join("\n"), raw: "" }],
		});

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			`[`,
			`  {`,
			`    "party": "democrat",`,
			`    "count": 1`,
			`  },`,
			`  {`,
			`    "party": "republican",`,
			`    "count": 1`,
			`  }`,
			`]`,
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

	it("exposes source fields and typed field values", async () => {
		const result = await run([
			`print(get("${DATASET_SLUG}:000001").fields["text"])`,
			`print(get("${DATASET_SLUG}:000001").fields["year"])`,
			`print(type(get("${DATASET_SLUG}:000001").fields["year"]))`,
			`print(get(fields)[0], get(fields)[1])`,
			`print(get(text_fields)[0])`,
			`var by_year = count_by(["year"] of all)`,
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

	it("filters and retrieves using explicit source fields", async () => {
		const result = await run([
			`print(count(temp(fields_compare: ["year": ["==", 2024]])))`,
			`print(count(temp(contains: ["text": "apple"])))`,
			`print(count(temp(contains: ["text": "apple"], contains: ["text": "Gamma"])))`,
			`print(count(temp(BM25: ["text": "apple"] > 0, BM25: ["text": "Gamma"] > 0)))`,
			`print(count(temp(contains: ["text": "banana"]) or temp(contains: ["text": "final"])))`,
			`var ranked = retrieve(top 1 in (BM25: ["text": "banana"]) of all)`,
			`print(ranked[0])`,
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			"2",
			"2",
			"1",
			"1",
			"2",
			`${DATASET_SLUG}:000002`,
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
				"print(count(all))",
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

	it("retrieves entries by direction around an entry id", async () => {
		const result = await run([
			`var before = "shadowed-before-variable"`,
			`var after = "shadowed-after-variable"`,
			`var next_ids = retrieve(top 2 in (direction: after from "${DATASET_SLUG}:000002") of all)`,
			"print(next_ids[0], next_ids[1])",
			`var previous_ids = retrieve(top 2 in (direction: before from "${DATASET_SLUG}:000003") of all)`,
			"print(previous_ids[0], previous_ids[1])",
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim().split("\n")).toEqual([
			`${DATASET_SLUG}:000003 ${DATASET_SLUG}:000004`,
			`${DATASET_SLUG}:000002 ${DATASET_SLUG}:000001`,
		]);
	});

	it("falls back to file order for direction when stored ordinals are missing", async () => {
		stripStoredOrdinals();

		const result = await run([
			`var after = retrieve(top 2 in (direction: after from "${DATASET_SLUG}:000002") of all)`,
			"print(after[0], after[1])",
		].join("\n"));

		expect(result.errors).toEqual([]);
		expect(result.output.trim()).toBe(`${DATASET_SLUG}:000003 ${DATASET_SLUG}:000004`);
	});

	it("allows query text to come from nested expressions", async () => {
		const result = await run([
			`var matches = retrieve(top 1 in (BM25: get("${DATASET_SLUG}:000001").text) of all)`,
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
});
