<script lang="ts">
    import {type Token, type TokenCallback} from "./interfaces";

    export let token: Token;
    export let index: number;
    export let underline: TokenCallback | undefined;
    export let bg: TokenCallback | undefined;

    let underlineStyle: string;
    $: underlineStyle = underline !== undefined ? "border-bottom-color: " + underline(token)  + ";": "";
    let bgStyle: string;
    $: bgStyle = bg !== undefined ? "background-color: " + bg(token)  + ";": "";
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