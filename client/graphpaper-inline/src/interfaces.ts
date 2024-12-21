// Interfaces used within the client. This is separate to messaging interfaces.

import type {GenToken} from "./stitch";

export interface MetricDef {
    name: string,
    units: string,
    description: string,
    isScalar: boolean,
    precision: number,
}

export type MetricVal = string | number | Array<number | string>;

export interface Token {
    text: string,
    prob: number,
    latency_ms: number,
    is_input: boolean,
    is_force_forwarded: boolean,
    is_generated: boolean,
    role: string,
    special: boolean,
    top_k?: Array<GenToken>
}
export declare type TokenCallback = (token: Token) => string;
