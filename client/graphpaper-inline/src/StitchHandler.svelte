<!-- Handles stitch client and kernel messages. -->
<script lang="ts">
    import type { Unsubscriber } from 'svelte/store';
    import { kernelmsg, clientmsg, type StitchMessage } from './stitch';
    import { onMount, onDestroy } from 'svelte';

    const handleMessage = (event: MessageEvent<any>) => {
        if (event.source === window.parent && 'type' in event.data && event.data.type === 'kernelmsg') {
            let stitchMessage: StitchMessage = event.data;

            // Notify kernel message has been received to subscribers
            kernelmsg.set(stitchMessage);
        }
    };

    let unsubscribe: Unsubscriber | null = null;
    onMount(() => {
        unsubscribe = clientmsg.subscribe((msg) => {
            if (msg !== undefined) {
                window.parent.postMessage(msg, "*");
            }
        });
    });
    onDestroy(() => {
        if (unsubscribe) {
            unsubscribe();
        }
    });
</script>

<svelte:window on:message={handleMessage} />