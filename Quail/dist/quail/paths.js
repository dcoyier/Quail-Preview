import { existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
export const QUAIL_WORKSPACE_DIR = "workspace";
export const QUAIL_DATASETS_DIR = "datasets";
export const QUAIL_STAGING_DIR = "staging";
export function getQuailWorkspaceDir(cwd) {
    return join(cwd, QUAIL_WORKSPACE_DIR);
}
export function getQuailDatasetsDir(cwd) {
    return join(getQuailWorkspaceDir(cwd), QUAIL_DATASETS_DIR);
}
export function getQuailStagingDir(cwd) {
    return join(getQuailWorkspaceDir(cwd), QUAIL_STAGING_DIR);
}
export function ensureQuailWorkspace(cwd) {
    for (const dir of [getQuailWorkspaceDir(cwd), getQuailDatasetsDir(cwd), getQuailStagingDir(cwd)]) {
        if (!existsSync(dir)) {
            mkdirSync(dir, { recursive: true });
        }
    }
}
//# sourceMappingURL=paths.js.map