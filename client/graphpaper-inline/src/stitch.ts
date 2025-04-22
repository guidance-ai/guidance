// Interfaces for working with guidance messages and stitch.

import { writable } from 'svelte/store';


export interface NodeAttr {
    class_name: string
}

export interface TextOutput extends NodeAttr {
    class_name: 'TextOutput' | 'TokenOutput',
    value: string,
    is_input: boolean,
    is_generated: boolean,
    is_force_forwarded: boolean,
    latency_ms: number,
}

export interface TokenOutput extends TextOutput {
    class_name: 'TokenOutput',
    token: Token,
    top_k: Array<Token>,
}

export interface Token {
    bytes: string
    prob: number
    masked: boolean,
}

export interface BacktrackMessage extends NodeAttr {
    class_name: 'BacktrackMessage',
    n_tokens: number,
    bytes: string,
}

export interface ImageOutput extends NodeAttr {
    class_name: 'ImageOutput',
    value: string,
    format: string,
    is_input: boolean,
}

export interface AudioOutput extends NodeAttr {
    class_name: 'AudioOutput',
    value: string,
    format: string,
    is_input: boolean,
}

export interface VideoOutput extends NodeAttr {
    class_name: 'VideoOutput',
    value: string,
    format: string,
    is_input: boolean,
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
    message_id: number,
    class_name: string,
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

export interface ExecutionStartedMessage extends GuidanceMessage {
    class_name: 'ExecutionStartedMessage',
}

export interface ExecutionCompletedMessage extends GuidanceMessage {
    class_name: 'ExecutionCompletedMessage',
    last_trace_id?: number,
}

export interface ClientReadyMessage extends GuidanceMessage {
    class_name: 'ClientReadyMessage'
}

export interface ClientReadyAckMessage extends GuidanceMessage {
    class_name: 'ClientReadyAckMessage'
}

export interface OutputRequestMessage extends GuidanceMessage {
    class_name: 'OutputRequestMessage'
}

export interface MetricMessage extends GuidanceMessage {
    class_name: 'MetricMessage',
    name: string,
    value: number | string | Array<number> | Array<string>,
    scalar: boolean,
}

export interface StitchMessage {
    type: "resize" | "clientmsg" | "kernelmsg" | "state" | "init_state",
    content: any
}

export function isGuidanceMessage(o: GuidanceMessage | undefined | null): o is GuidanceMessage {
    if (o === undefined || o === null) return false;
    return o.hasOwnProperty("class_name") && o.hasOwnProperty("message_id");
}

export function isTraceMessage(o: GuidanceMessage | undefined | null): o is TraceMessage {
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

export function isImageOutput(o: NodeAttr | undefined | null): o is ImageOutput {
    if (o === undefined || o === null) return false;
    return o.class_name === "ImageOutput";
}

export function isAudioOutput(o: NodeAttr | undefined | null): o is AudioOutput {
    if (o === undefined || o === null) return false;
    return o.class_name === "AudioOutput";
}

export function isVideoOutput(o: NodeAttr | undefined | null): o is VideoOutput {
    if (o === undefined || o === null) return false;
    return o.class_name === "VideoOutput";
}

export function isResetDisplayMessage(o: GuidanceMessage | undefined | null): o is ResetDisplayMessage {
    if (o === undefined || o === null) return false;
    return o.class_name === "ResetDisplayMessage";
}

export function isMetricMessage(o: GuidanceMessage | undefined | null): o is MetricMessage {
    if (o === undefined || o === null) return false;
    return o.class_name === "MetricMessage";
}

export function isClientReadyAckMessage(o: GuidanceMessage | undefined | null): o is MetricMessage {
    if (o === undefined || o === null) return false;
    return o.class_name === "ClientReadyAckMessage";
}

export function isExecutionCompletedMessage(o: GuidanceMessage | undefined | null): o is ExecutionCompletedMessage {
    if (o === undefined || o === null) return false;
    return o.class_name === "ExecutionCompletedMessage";
}

export function isExecutionStartedMessage(o: GuidanceMessage | undefined | null): o is ExecutionStartedMessage {
    if (o === undefined || o === null) return false;
    return o.class_name === "ExecutionStartedMessage";
}

export const kernelmsg = writable<StitchMessage | undefined>(undefined);
export const clientmsg = writable<StitchMessage | undefined>(undefined);
export const state = writable<StitchMessage | undefined>(undefined);

export enum Status {
  Running = 'Running',
  Error = 'Error',
  Done = 'Done',
}