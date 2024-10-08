import {writable} from 'svelte/store';

export interface NodeAttr {
    class_name: string
}

export interface TextOutput extends NodeAttr {
    class_name: 'TextOutput',
    value: string,
    is_generated: boolean,
    token_count: number,
    prob: number,
}

export interface RoleOpenerInput extends NodeAttr {
    class_name: 'RoleOpenerInput',
    name?: string,
    text?: string,
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

export interface TokenBatchMessage extends GuidanceMessage {
    class_name: 'TokenBatchMessage',
    tokens: Array<any>
}

export interface JupyterCellExecutionCompletedMessage extends GuidanceMessage {
    class_name: 'JupyterCellExecutionCompletedMessage',
    last_trace_id?: number,
}

export interface ClientReadyMessage extends GuidanceMessage {
    class_name: 'ClientReadyMessage'
}

export interface MetricMessage extends GuidanceMessage {
    class_name: 'MetricMessage',
    name: string,
    value: number | string
}

export interface StitchMessage {
    type: "resize" | "clientmsg" | "kernelmsg",
    content: any
}

export function isTraceMessage(o: NodeAttr | undefined): o is TraceMessage {
    if (o === undefined) return false;
    return o.class_name === "TraceMessage";
}

export function isRoleOpenerInput(o: NodeAttr | undefined): o is RoleOpenerInput {
    if (o === undefined) return false;
    return o.class_name === "RoleOpenerInput";
}

export function isRoleCloserInput(o: NodeAttr | undefined): o is RoleOpenerInput {
    if (o === undefined) return false;
    return o.class_name === "RoleCloserInput";
}

export function isTextOutput(o: NodeAttr | undefined): o is TextOutput {
    if (o === undefined) return false;
    return o.class_name === "TextOutput";
}

export const kernelmsg = writable<StitchMessage | undefined>(undefined);
export const clientmsg = writable<StitchMessage | undefined>(undefined);

