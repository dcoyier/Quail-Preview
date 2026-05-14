import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const aiSrcIndex = fileURLToPath(new URL("../ai/src/index.ts", import.meta.url));
const aiSrcOAuth = fileURLToPath(new URL("../ai/src/oauth.ts", import.meta.url));
const agentSrcIndex = fileURLToPath(new URL("../agent/src/index.ts", import.meta.url));

const monorepoAliases = [
	existsSync(aiSrcIndex) ? { find: /^@mariozechner\/pi-ai$/, replacement: aiSrcIndex } : undefined,
	existsSync(aiSrcOAuth) ? { find: /^@mariozechner\/pi-ai\/oauth$/, replacement: aiSrcOAuth } : undefined,
	existsSync(agentSrcIndex) ? { find: /^@mariozechner\/pi-agent-core$/, replacement: agentSrcIndex } : undefined,
].filter((alias): alias is { find: RegExp; replacement: string } => Boolean(alias));

export default defineConfig({
	test: {
		globals: true,
		environment: "node",
		testTimeout: 30000,
		server: {
			deps: {
				external: [/@silvia-odwyer\/photon-node/],
			},
		},
	},
	resolve: {
		alias: monorepoAliases,
	},
});
