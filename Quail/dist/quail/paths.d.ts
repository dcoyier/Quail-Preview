export declare const QUAIL_WORKSPACE_DIR = "workspace";
export declare const QUAIL_DATASETS_DIR = "datasets";
export declare const QUAIL_STAGING_DIR = "staging";
export declare function getQuailWorkspaceDir(cwd: string): string;
export declare function getQuailDatasetsDir(cwd: string): string;
export declare function getQuailStagingDir(cwd: string): string;
export declare function ensureQuailWorkspace(cwd: string): void;
