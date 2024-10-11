import {writable} from 'svelte/store';

export interface BaseToken {
    token: number,
    prob: number,
    text: string,
}

export interface GenToken extends BaseToken {
    latency_ms: number,
    is_generated: boolean,
    is_force_forwarded: boolean,
    is_input: boolean,
    top_k: Array<BaseToken>,
}

export interface NodeAttr {
    class_name: string
}

export interface TextOutput extends NodeAttr {
    class_name: 'TextOutput',
    value: string,
    is_input: boolean,
    is_generated: boolean,
    is_force_forwarded: boolean,
    token_count: number,
    prob: number,
}

export interface RoleOpenerInput extends NodeAttr {
    class_name: 'RoleOpenerInput',
    name?: string,
    text?: string,
    closer_text?: string,
}

export interface RoleCloserInput extends NodeAttr {
    class_name: 'RoleCloserInput',
    name?: string,
    text?: string,
}

export interface GuidanceMessage {
    class_name: string
}

export interface TraceMessage extends GuidanceMessage {
    class_name: 'TraceMessage',
    trace_id: number,
    parent_trace_id?: number,
    node_attr?: NodeAttr,
}

export interface ResetDisplayMessage extends GuidanceMessage {
    class_name: 'ResetDisplayMessage'
}

export interface ExecutionCompletedMessage extends GuidanceMessage {
    class_name: 'ExecutionCompletedMessage',
    last_trace_id?: number,
}

export interface ExecutionCompletedOutputMessage extends GuidanceMessage {
    class_name: 'ExecutionCompletedOutputMessage',
    trace_id: number,
    text: string,
    tokens: Array<GenToken>,
}

export interface ClientReadyMessage extends GuidanceMessage {
    class_name: 'ClientReadyMessage'
}

export interface MetricMessage extends GuidanceMessage {
    class_name: 'MetricMessage',
    name: string,
    value: number | string | Array<number> | Array<string>,
    scalar: boolean,
}

export interface StitchMessage {
    type: "resize" | "clientmsg" | "kernelmsg",
    content: any
}

export function isTraceMessage(o: NodeAttr | undefined | null): o is TraceMessage {
    if (o === undefined || o === null) return false;
    return o.class_name === "TraceMessage";
}

export function isRoleOpenerInput(o: NodeAttr | undefined | null): o is RoleOpenerInput {
    if (o === undefined || o === null) return false;
    return o.class_name === "RoleOpenerInput";
}

export function isRoleCloserInput(o: NodeAttr | undefined | null): o is RoleCloserInput {
    if (o === undefined || o === null) return false;
    return o.class_name === "RoleCloserInput";
}

export function isTextOutput(o: NodeAttr | undefined | null): o is TextOutput {
    if (o === undefined || o === null) return false;
    return o.class_name === "TextOutput";
}

export function isResetDisplayMessage(o: NodeAttr | undefined | null): o is ResetDisplayMessage {
    if (o === undefined || o === null) return false;
    return o.class_name === "ResetDisplayMessage";
}

export function isMetricMessage(o: NodeAttr | undefined | null): o is MetricMessage {
    if (o === undefined || o === null) return false;
    return o.class_name === "MetricMessage";
}

export function isExecutionCompletedMessage(o: NodeAttr | undefined | null): o is ExecutionCompletedMessage {
    if (o === undefined || o === null) return false;
    return o.class_name === "ExecutionCompletedMessage";
}

export function isExecutionCompletedOutputMessage(o: NodeAttr | undefined | null): o is ExecutionCompletedOutputMessage {
    if (o === undefined || o === null) return false;
    return o.class_name === "ExecutionCompletedOutputMessage";
}

export const kernelmsg = writable<StitchMessage | undefined>(undefined);
export const clientmsg = writable<StitchMessage | undefined>(undefined);

