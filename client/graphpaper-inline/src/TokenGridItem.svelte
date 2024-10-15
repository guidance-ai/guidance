<script lang="ts">
    import {type Token, type TokenCallback} from "./interfaces";
    import {scaleSequential} from "d3-scale";
    import {interpolateCool, interpolateSpectral, interpolateOrRd, interpolateYlOrRd} from "d3-scale-chromatic";
    // import {interpolateRgb} from "d3-interpolate";

    export let token: Token;
    export let index: number;
    export let underline: TokenCallback | undefined;
    export let bg: TokenCallback | undefined;

    const underlineColor = (x: number) => {
        return interpolateOrRd(x);
    };
    const bgColor = (x: number) => {
        return interpolateYlOrRd(1 - x);
    };

    let underlineStyle: string;
    $: underlineStyle = underline !== undefined ? "border-bottom-color: " + underlineColor(underline(token))  + ";": "";
    let bgStyle: string;
    $: bgStyle = bg !== undefined ? "background-color: " + bgColor(bg(token))  + ";": "";
</script>

{#each token.text as ch, i}
    {#if ch === ' '}
        <span data-index="{index}" role="tooltip" class={`token-grid-item inline-block mt-2 border-b-2 text-gray-300`} style={`${underlineStyle} ${bgStyle}`}>
            &nbsp;
        </span>
    {:else if ch === '\t'}
        <span data-index="{index}" role="tooltip" class={`token-grid-item inline-block mt-2 border-b-2 text-gray-300`} style={`${underlineStyle} ${bgStyle}`}>
            \t&nbsp;&nbsp;
        </span>
    {:else if ch === '\n'}
        <span data-index="{index}" role="tooltip" class={`token-grid-item inline-block mt-2 border-b-2 text-gray-300`} style={`${underlineStyle} ${bgStyle}`}>
            \n
        </span>
        <div class="basis-full h-full"></div>
    {:else}
        <span data-index="{index}" role="tooltip" class={`token-grid-item inline-block mt-2 border-b-2 ${token.special ? "text-gray-300" : ""}`} style={`${underlineStyle} ${bgStyle}`}>
            {#if i === 0}
                <span class="absolute text-xs uppercase -mt-4 text-purple-800 font-sans">
                    {token.role}
                </span>
            {/if}
            {ch}
        </span>
    {/if}
{/each}