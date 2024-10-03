import { writable } from 'svelte/store';

export interface NodeAttr {
    class_name: string
}

export interface TextOutput extends NodeAttr {
    value: string,
    is_generated: boolean,
    token_count: number,
    prob: number,
}

export interface GuidanceMessage {
    class_name: string
}

export interface TraceMessage extends GuidanceMessage {
    trace_id: number,
    parent_trace_id?: number,
    node_attr?: NodeAttr,
}

export interface ResetDisplayMessage extends GuidanceMessage {
    class_name: 'ResetDisplayMessage'
}

export interface ClientReadyMessage extends GuidanceMessage {
    class_name: 'ClientReadyMessage'
}

export interface StitchMessage {
    type: "resize" | "clientmsg" | "kernelmsg",
    content: any
}

export const kernelmsg = writable<StitchMessage | undefined>(undefined);
export const clientmsg = writable<StitchMessage | undefined>(undefined);