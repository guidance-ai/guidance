import { writable } from 'svelte/store';

export interface StitchMessage {
    type: "resize" | "clientmsg" | "kernelmsg",
    content: any
}

export interface DisplayResetMessage {
    message_type: 'DisplayResetMessage'
}

export interface ModelUpdateMessage {
    message_type: 'ModelUpdateMessage',
    model_id: number,
    parent_model_id: number,
    content_type: string,
    content: string,
    prob: number,
    token_count: number,
    is_generated: number,
    is_special: number,
    role: string
}

export const kernelmsg = writable<StitchMessage>(undefined);
export const clientmsg = writable<StitchMessage>(undefined);