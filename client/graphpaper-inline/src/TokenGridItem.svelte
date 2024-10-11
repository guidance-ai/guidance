<script context="module" lang="ts">
    import type {GenToken} from "./stitch";

    export interface Token {
        text: string,
        prob: number,
        role: string,
        special: boolean,
        extra?: GenToken,
    }
</script>

<script lang="ts">
    import {scaleSequential} from "d3-scale";
    import {interpolateSpectral} from "d3-scale-chromatic";

    export let token;
    export let index;

    const color = (x: number) => {
        return scaleSequential(interpolateSpectral)(1.0 - x)
    };
</script>

{#each token.text as ch, i}
    {#if ch === ' '}
        <span data-index="{index}" role="tooltip" class={`token-grid-item inline-block mt-2 border-b-2 text-gray-300`} style={`border-bottom-color: ${color(token.prob)}`}>
            &nbsp;
        </span>
    {:else if ch === '\t'}
        <span data-index="{index}" role="tooltip" class={`token-grid-item inline-block mt-2 border-b-2 text-gray-300`} style={`border-bottom-color: ${color(token.prob)}`}>
            \t&nbsp;&nbsp;
        </span>
    {:else if ch === '\n'}
        <span data-index="{index}" role="tooltip" class={`token-grid-item inline-block mt-2 border-b-2 text-gray-300`} style={`border-bottom-color: ${color(token.prob)}`}>
            \n
        </span>
        <div class="basis-full h-full"></div>
    {:else}
        <span data-index="{index}" role="tooltip" class={`token-grid-item inline-block mt-2 border-b-2 ${token.special ? "text-gray-300" : ""}`} style={`border-bottom-color: ${color(token.prob)};`}>
            {#if i === 0}
                <span class="absolute text-xs uppercase -mt-4 text-purple-800 font-sans">
                    {token.role}
                </span>
            {/if}
            {ch}
        </span>
    {/if}
{/each}