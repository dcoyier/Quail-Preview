import { existsSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import chalk from "chalk";
import { ensureQuailWorkspace, getQuailStagingDir } from "./paths.js";
import { datasetExists, listDatasets, processDataset, removeDataset } from "./dataset-store.js";
function printDatasetHelp() {
    console.log(`Quail dataset commands

Usage:
  hatch dataset list
  hatch dataset process --name <unique name> --input <file> [--tag field=value]
  hatch dataset process --name <unique name> --text "entry one\nentry two"
  hatch dataset remove <name> --yes

Options:
  --format <auto|txt|csv|tsv|json|jsonl>
  --text-column <column>
  --model <ollama embedding model>      Default: embeddinggemma:latest
  --batch-size <n>                      Default: 64
  --tag <field=value>                   Can be repeated; added to every entry
  --overwrite                           Replace an existing dataset of the same name
  --skip-embeddings                     Only for repair/debug workflows
`);
}
function parseKeyValue(value) {
    const index = value.indexOf("=");
    if (index <= 0)
        throw new Error(`Expected field=value, got "${value}"`);
    const key = value.slice(0, index).trim();
    const tagValue = value.slice(index + 1).trim();
    if (!key)
        throw new Error(`Expected non-empty metadata field in "${value}"`);
    return [key, tagValue];
}
function parseDatasetArgs(args) {
    const parsed = { command: args[0], tags: {} };
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
        }
        else if ((arg === "--input" || arg === "-i") && next !== undefined) {
            parsed.input = next;
            i++;
        }
        else if (arg === "--text" && next !== undefined) {
            parsed.text = next;
            i++;
        }
        else if (arg === "--format" && next !== undefined) {
            parsed.format = next;
            i++;
        }
        else if (arg === "--text-column" && next !== undefined) {
            parsed.textColumn = next;
            i++;
        }
        else if (arg === "--model" && next !== undefined) {
            parsed.model = next;
            i++;
        }
        else if (arg === "--batch-size" && next !== undefined) {
            const batchSize = Number.parseInt(next, 10);
            if (!Number.isFinite(batchSize) || batchSize <= 0)
                throw new Error("--batch-size must be a positive integer");
            parsed.batchSize = batchSize;
            i++;
        }
        else if (arg === "--tag" && next !== undefined) {
            const [key, value] = parseKeyValue(next);
            parsed.tags[key] = value;
            i++;
        }
        else if (arg === "--overwrite") {
            parsed.overwrite = true;
        }
        else if (arg === "--yes" || arg === "-y") {
            parsed.yes = true;
        }
        else if (arg === "--skip-embeddings") {
            parsed.skipEmbeddings = true;
        }
        else if (arg === "--help" || arg === "-h") {
            parsed.command = "help";
        }
        else {
            throw new Error(`Unknown dataset argument: ${arg}`);
        }
    }
    return parsed;
}
function createTextInputFile(cwd, text) {
    ensureQuailWorkspace(cwd);
    const filePath = join(getQuailStagingDir(cwd), `pasted-${Date.now()}.txt`);
    writeFileSync(filePath, text, "utf8");
    return filePath;
}
export async function handleDatasetCommand(args, cwd = process.cwd()) {
    if (args[0] !== "dataset" && args[0] !== "datasets")
        return false;
    let parsed;
    try {
        parsed = parseDatasetArgs(args.slice(1));
    }
    catch (error) {
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
                    console.log("No processed datasets in workspace/datasets yet.");
                    return true;
                }
                for (const item of datasets) {
                    console.log(`${item.name}\t${item.entryCount} entries\t${item.embeddingModel}\tmetadata: ${item.metadataFields.join(", ") || "none"}`);
                }
                return true;
            }
            case "process": {
                if (!parsed.name)
                    throw new Error("--name is required");
                if (!parsed.input && !parsed.text)
                    throw new Error("--input <file> or --text <text> is required");
                if (!parsed.overwrite && datasetExists(cwd, parsed.name)) {
                    throw new Error(`Dataset "${parsed.name}" already exists. Choose a unique name or pass --overwrite.`);
                }
                const inputPath = parsed.text ? createTextInputFile(cwd, parsed.text) : resolve(cwd, parsed.input);
                if (!existsSync(inputPath))
                    throw new Error(`Input file not found: ${inputPath}`);
                const manifest = await processDataset({
                    cwd,
                    inputPath,
                    name: parsed.name,
                    format: parsed.format,
                    textColumn: parsed.textColumn,
                    model: parsed.model,
                    batchSize: parsed.batchSize,
                    globalTags: parsed.tags,
                    overwrite: parsed.overwrite,
                    skipEmbeddings: parsed.skipEmbeddings,
                    onProgress: (message) => console.log(message),
                });
                console.log(chalk.green(`Processed dataset "${manifest.name}" with ${manifest.entryCount} entries (${manifest.embeddingModel}, ${manifest.embeddingDimensions} dimensions).`));
                return true;
            }
            case "remove": {
                if (!parsed.name)
                    throw new Error("Dataset name is required: hatch dataset remove <name> --yes");
                if (!parsed.yes)
                    throw new Error("Refusing to remove without --yes");
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
    }
    catch (error) {
        console.error(chalk.red(`Error: ${error instanceof Error ? error.message : String(error)}`));
        process.exitCode = 1;
        return true;
    }
}
//# sourceMappingURL=dataset-cli.js.map