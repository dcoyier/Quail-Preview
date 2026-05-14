import { existsSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import chalk from "chalk";
import { ensureQuailWorkspace, getQuailDatasetsDir, getQuailStagingDir } from "./paths.js";
import {
	defaultEmbeddingBatchSize,
	defaultEmbeddingConcurrency,
	defaultEmbeddingModel,
	datasetExists,
	inspectDatasetFile,
	listDatasets,
	processDataset,
	removeDataset,
	type FieldTypeOverride,
} from "./dataset-store.js";

interface DatasetCliArgs {
	command?: string;
	name?: string;
	input?: string;
	format?: string;
	textColumn?: string;
	model?: string;
	batchSize?: number;
	embeddingConcurrency?: number;
	tags: Record<string, string>;
	fieldTypes: Record<string, FieldTypeOverride>;
	overwrite?: boolean;
	yes?: boolean;
	skipEmbeddings?: boolean;
	dryRun?: boolean;
	text?: string;
}

function printDatasetHelp(): void {
	console.log(`Quail dataset commands

Usage:
  hatch dataset list
  hatch dataset inspect --input <file> [--field-type field=type]
  hatch dataset process --name <unique name> --input <file> [--tag field=value]
  hatch dataset process --name <unique name> --text "entry one\nentry two"
  hatch dataset remove <name> --yes

Options:
  --format <auto|txt|csv|tsv|json|jsonl>
  --text-column <column>                Legacy preview hint; embedding is type-driven
  --model <embedding model>             Default: ${defaultEmbeddingModel()}
  --batch-size <n>                      Default: ${defaultEmbeddingBatchSize()}
  --embedding-concurrency <n>           Default: ${defaultEmbeddingConcurrency()}
  --tag <field=value>                   Can be repeated; added to every entry as a source field
  --field-type <field=type>             Override inferred type; type is string, int, float, bool, list, object, mixed, or null
  --dry-run                             Inspect fields without writing a processed dataset
  --overwrite                           Replace an existing dataset of the same name
  --skip-embeddings                     Only for repair/debug workflows

Embedding defaults:
  Provider: OpenRouter; set OPENROUTER_API_KEY or QUAIL_OPENROUTER_API_KEY
  Set QUAIL_EMBEDDING_PROVIDER=ollama to use an Ollama-compatible /api/embed endpoint
`);
}

function parseKeyValue(value: string): [string, string] {
	const index = value.indexOf("=");
	if (index <= 0) throw new Error(`Expected field=value, got "${value}"`);
	const key = value.slice(0, index).trim();
	const tagValue = value.slice(index + 1).trim();
	if (!key) throw new Error(`Expected non-empty field in "${value}"`);
	return [key, tagValue];
}

function parseFieldType(value: string): [string, FieldTypeOverride] {
	const [key, fieldType] = parseKeyValue(value);
	return [key, fieldType as FieldTypeOverride];
}

function parseDatasetArgs(args: string[]): DatasetCliArgs {
	const parsed: DatasetCliArgs = { command: args[0], tags: {}, fieldTypes: {} };
	if (parsed.command === "--help" || parsed.command === "-h") {
		parsed.command = "help";
		return parsed;
	}
	if (parsed.command === "remove" && args[1] && !args[1].startsWith("--")) {
		parsed.name = args[1];
	}
	for (let i = parsed.command === "remove" && parsed.name ? 2 : 1; i < args.length; i++) {
		const arg = args[i];
		const next = args[i + 1];
		if ((arg === "--name" || arg === "-n") && next !== undefined) {
			parsed.name = next;
			i++;
		} else if ((arg === "--input" || arg === "-i") && next !== undefined) {
			parsed.input = next;
			i++;
		} else if (arg === "--text" && next !== undefined) {
			parsed.text = next;
			i++;
		} else if (arg === "--format" && next !== undefined) {
			parsed.format = next;
			i++;
		} else if (arg === "--text-column" && next !== undefined) {
			parsed.textColumn = next;
			i++;
		} else if (arg === "--model" && next !== undefined) {
			parsed.model = next;
			i++;
		} else if (arg === "--batch-size" && next !== undefined) {
			const batchSize = Number.parseInt(next, 10);
			if (!Number.isFinite(batchSize) || batchSize <= 0) throw new Error("--batch-size must be a positive integer");
			parsed.batchSize = batchSize;
			i++;
		} else if (arg === "--embedding-concurrency" && next !== undefined) {
			const embeddingConcurrency = Number.parseInt(next, 10);
			if (!Number.isFinite(embeddingConcurrency) || embeddingConcurrency <= 0) {
				throw new Error("--embedding-concurrency must be a positive integer");
			}
			parsed.embeddingConcurrency = embeddingConcurrency;
			i++;
		} else if (arg === "--tag" && next !== undefined) {
			const [key, value] = parseKeyValue(next);
			parsed.tags[key] = value;
			i++;
		} else if (arg === "--field-type" && next !== undefined) {
			const [key, value] = parseFieldType(next);
			parsed.fieldTypes[key] = value;
			i++;
		} else if (arg === "--overwrite") {
			parsed.overwrite = true;
		} else if (arg === "--dry-run") {
			parsed.dryRun = true;
		} else if (arg === "--yes" || arg === "-y") {
			parsed.yes = true;
		} else if (arg === "--skip-embeddings") {
			parsed.skipEmbeddings = true;
		} else if (arg === "--help" || arg === "-h") {
			parsed.command = "help";
		} else {
			throw new Error(`Unknown dataset argument: ${arg}`);
		}
	}
	return parsed;
}

function createTextInputFile(cwd: string, text: string): string {
	ensureQuailWorkspace(cwd);
	const filePath = join(getQuailStagingDir(cwd), `pasted-${Date.now()}.txt`);
	writeFileSync(filePath, text, "utf8");
	return filePath;
}

function getInputPath(parsed: DatasetCliArgs, cwd: string): string {
	if (!parsed.input && !parsed.text) throw new Error("--input <file> or --text <text> is required");
	const inputPath = parsed.text ? createTextInputFile(cwd, parsed.text) : resolve(cwd, parsed.input!);
	if (!existsSync(inputPath)) throw new Error(`Input file not found: ${inputPath}`);
	return inputPath;
}

function printInspection(parsed: DatasetCliArgs, inputPath: string): void {
	const inspection = inspectDatasetFile({
		inputPath,
		format: parsed.format,
		textColumn: parsed.textColumn,
		globalTags: parsed.tags,
		fieldTypes: parsed.fieldTypes,
	});
	console.log(`Detected ${inspection.entryCount} records.`);
	console.log("");
	console.log("Fields:");
	for (const field of inspection.fields) {
		const samples = field.samples.length > 0 ? `; sample: ${field.samples.join(" | ")}` : "";
		console.log(`- ${field.name}: ${field.type}; embedded: ${field.embedded ? "yes" : "no"}; non-empty: ${field.nonEmptyCount}${samples}`);
	}
	console.log("");
	console.log(`Embedded fields: ${inspection.embeddedFields.length > 0 ? inspection.embeddedFields.join(", ") : "none"}`);
}

export async function handleDatasetCommand(args: string[], cwd = process.cwd()): Promise<boolean> {
	if (args[0] !== "dataset" && args[0] !== "datasets") return false;
	let parsed: DatasetCliArgs;
	try {
		parsed = parseDatasetArgs(args.slice(1));
	} catch (error) {
		console.error(chalk.red(`Error: ${error instanceof Error ? error.message : String(error)}`));
		printDatasetHelp();
		process.exitCode = 1;
		return true;
	}

	try {
		switch (parsed.command) {
				case "list": {
					const datasets = listDatasets(cwd);
					if (datasets.length === 0) {
						console.log(`No processed datasets in ${getQuailDatasetsDir(cwd)} yet.`);
						return true;
					}
				for (const item of datasets) {
					console.log(
						`${item.name}\t${item.entryCount} entries\t${item.embeddingModel}\tfields: ${item.metadataFields.join(", ") || "none"}`,
					);
				}
				return true;
			}
			case "inspect": {
				const inputPath = getInputPath(parsed, cwd);
				printInspection(parsed, inputPath);
				return true;
			}
			case "process": {
				const inputPath = getInputPath(parsed, cwd);
				if (parsed.dryRun) {
					printInspection(parsed, inputPath);
					return true;
				}
				if (!parsed.name) throw new Error("--name is required");
				if (!parsed.overwrite && datasetExists(cwd, parsed.name)) {
					throw new Error(`Dataset "${parsed.name}" already exists. Choose a unique name or pass --overwrite.`);
				}
				const manifest = await processDataset({
					cwd,
					inputPath,
					name: parsed.name,
					format: parsed.format,
					textColumn: parsed.textColumn,
					model: parsed.model,
					batchSize: parsed.batchSize,
					embeddingConcurrency: parsed.embeddingConcurrency,
					globalTags: parsed.tags,
					fieldTypes: parsed.fieldTypes,
					overwrite: parsed.overwrite,
					skipEmbeddings: parsed.skipEmbeddings,
					onProgress: (message) => console.log(message),
				});
				console.log(
					chalk.green(
						`Processed dataset "${manifest.name}" with ${manifest.entryCount} entries (${manifest.embeddingModel}, ${manifest.embeddingDimensions} dimensions).`,
					),
				);
				console.log(`Embedded fields: ${manifest.embeddedFields?.join(", ") || "none"}`);
				return true;
			}
			case "remove": {
				if (!parsed.name) throw new Error("Dataset name is required: hatch dataset remove <name> --yes");
				if (!parsed.yes) throw new Error("Refusing to remove without --yes");
				const removed = removeDataset(cwd, parsed.name);
				console.log(removed ? `Removed dataset "${parsed.name}".` : `Dataset "${parsed.name}" was not found.`);
				return true;
			}
			case "help":
			case undefined:
				printDatasetHelp();
				return true;
			default:
				throw new Error(`Unknown dataset command: ${parsed.command}`);
		}
	} catch (error) {
		console.error(chalk.red(`Error: ${error instanceof Error ? error.message : String(error)}`));
		process.exitCode = 1;
		return true;
	}
}
