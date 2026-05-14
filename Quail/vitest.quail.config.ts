import { defineConfig, mergeConfig } from "vitest/config";
import baseConfig from "./vitest.config.ts";

export default mergeConfig(baseConfig, defineConfig({
	test: {
		include: ["test/quail-*.test.ts", "test/system-prompt.test.ts"],
	},
}));
