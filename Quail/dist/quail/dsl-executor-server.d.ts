import http from "node:http";
export interface QuailDslExecutorServerOptions {
    host?: string;
    port?: number;
    maxBodyBytes?: number;
    log?: (event: string, payload?: Record<string, unknown>) => void;
}
export declare function startQuailDslExecutorServer(options?: QuailDslExecutorServerOptions): Promise<http.Server>;
