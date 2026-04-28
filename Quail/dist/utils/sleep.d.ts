/**
 * Sleep helper that respects abort signal.
 */
export declare function sleep(ms: number, signal?: AbortSignal): Promise<void>;
