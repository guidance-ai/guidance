<!-- Handles resizing of content, especially important for jupyter notebooks. -->
<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { clientmsg, type StitchMessage} from "./stitch";

    const INTERVAL_MS = 20;
    let interval: any = null;
    let htmlElem: any;

    onMount(() => {
        htmlElem = document.querySelector('html');
        window.addEventListener("load", () => {
            let prevHeight = 0;

            interval = setInterval(() => {
                const height = htmlElem.getBoundingClientRect().height;
                if (height !== prevHeight && htmlElem.checkVisibility()) {
                    const msg: StitchMessage = {
                        'type': 'resize',
                        'content': {
                            height: `${height}px`,
                            width: '100%'
                        }
                    };
                    clientmsg.set(msg);
                }
            }, INTERVAL_MS);
        });
    });
    onDestroy(() => {
        clearInterval(interval);
    });
</script>