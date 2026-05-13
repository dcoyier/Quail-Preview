import chalk from "chalk";
import { startQuailDslExecutorServer } from "./dsl-executor-server.js";
function printExecutorHelp() {
    console.log(`Quail DSL executor commands

Usage:
  hatch executor start [--host 127.0.0.1] [--port 0]
  hatch dsl-executor start [--host 127.0.0.1] [--port 0]

Options:
  --host <host>                         Host to bind (default: 127.0.0.1)
  --port <port>                         Port to bind; 0 chooses a free port (default: 0)
  --max-body-bytes <bytes>              Maximum JSON request body size

The executor keeps processed datasets and DSL runtime caches warm across calls.
Point a Quail session at it with QUAIL_DSL_EXECUTOR_URL=http://host:port.`);
}
function parseExecutorArgs(args) {
    const parsed = { command: args[0] };
    if (parsed.command === "--help" || parsed.command === "-h")
        parsed.command = "help";
    for (let i = 1; i < args.length; i++) {
        const arg = args[i];
        const next = args[i + 1];
        if (arg === "--help" || arg === "-h") {
            parsed.command = "help";
        }
        else if (arg === "--host" && next) {
            parsed.host = next;
            i++;
        }
        else if (arg === "--port" && next) {
            parsed.port = Number(next);
            i++;
        }
        else if (arg === "--max-body-bytes" && next) {
            parsed.maxBodyBytes = Number(next);
            i++;
        }
        else {
            throw new Error(`Unknown executor argument: ${arg}`);
        }
    }
    return parsed;
}
export async function handleExecutorCommand(args) {
    if (args[0] !== "executor" && args[0] !== "dsl-executor")
        return false;
    let parsed;
    try {
        parsed = parseExecutorArgs(args.slice(1));
    }
    catch (error) {
        console.error(chalk.red(`Error: ${error instanceof Error ? error.message : String(error)}`));
        printExecutorHelp();
        process.exitCode = 1;
        return true;
    }
    try {
        switch (parsed.command) {
            case "start":
                await startQuailDslExecutorServer({
                    host: parsed.host,
                    port: parsed.port,
                    maxBodyBytes: parsed.maxBodyBytes,
                });
                return await new Promise(() => undefined);
            case "help":
            case undefined:
                printExecutorHelp();
                return true;
            default:
                throw new Error(`Unknown executor command: ${parsed.command}`);
        }
    }
    catch (error) {
        console.error(chalk.red(`Error: ${error instanceof Error ? error.message : String(error)}`));
        process.exitCode = 1;
        return true;
    }
}
//# sourceMappingURL=executor-cli.js.map