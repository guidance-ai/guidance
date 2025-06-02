<!-- Handles stitch client and kernel messages. -->
<script lang="ts">
    import type { Unsubscriber } from 'svelte/store';
    import { kernelmsg, clientmsg, type StitchMessage, state } from './stitch';
    import { onMount, onDestroy } from 'svelte';

    const handleMessage = (event: MessageEvent<any>) => {
        if (event.source === window.parent && 'type' in event.data) {
            if (event.data.type === 'kernelmsg') {
                let stitchMessage: StitchMessage = event.data;
                kernelmsg.set(stitchMessage);
            } else if (event.data.type === 'init_state') {
                let stitchMessage: StitchMessage = event.data;
                state.set(stitchMessage);
                const clientReadyMsg: StitchMessage = {
                    type: 'clientmsg',
                    content: JSON.stringify({ 'class_name': 'ClientReadyMessage' })
                };
                clientmsg.set(clientReadyMsg);
            }
        }
    };

    let unsubscribeClient: Unsubscriber | null = null;
    let unsubscribeState: Unsubscriber | null = null;
    onMount(() => {
        unsubscribeClient = clientmsg.subscribe((msg) => {
            if (msg !== undefined) {
                window.parent.postMessage(msg, "*");
            }
        });
        unsubscribeState = state.subscribe((msg) => {
            if (msg !== undefined) {
                window.parent.postMessage(msg, "*");
            }
        });
    });
    onDestroy(() => {
        if (unsubscribeClient) {
            unsubscribeClient();
        }
        if (unsubscribeState) {
            unsubscribeState();
        }
    });
</script>

<svelte:window on:message={handleMessage} />