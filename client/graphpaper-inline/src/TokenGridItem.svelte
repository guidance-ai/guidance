<script lang="ts">
    import {scaleSequential} from "d3-scale";
    import {interpolateSpectral} from "d3-scale-chromatic";

    export let token;
    const color = (x: number) => {
        return scaleSequential(interpolateSpectral)(1.0 - x)
    };
</script>

{#each token.value as ch}
    {#if ch === ' '}
        <span class={`inline-block mt-2 text-gray-300 border-b-2 hover:bg-gray-300 hover:text-gray-700`} style={`border-bottom-color: ${color(token.prob)}`}>
            &nbsp;
        </span>
    {:else if ch === '\t'}
        <span class={`inline-block mt-2 text-gray-300 border-b-2 hover:bg-gray-300 hover:text-gray-700`} style={`border-bottom-color: ${color(token.prob)}`}>
            \t&nbsp;&nbsp;
        </span>
    {:else if ch === '\n'}
        <span class={`inline-block mt-2 text-gray-300 border-b-2 hover:bg-gray-300 hover:text-gray-700`} style={`border-bottom-color: ${color(token.prob)}`}>
            \n
        </span>
        <div class="basis-full h-full"></div>
    {:else}
        <span class={`inline-block mt-2 border-b-2 hover:bg-gray-300 ${token.special ? "text-gray-300" : ""}`} style={`border-bottom-color: ${color(token.prob)}`}>
            {ch}
        </span>
    {/if}
{/each}