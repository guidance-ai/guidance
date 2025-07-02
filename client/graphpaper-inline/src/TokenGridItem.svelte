<!-- Token(s) within token grid -->
<script lang="ts">
    import {type Token} from "./interfaces";

    export let token: Token;
    export let index: number;
    export let underlineStyle: string = "";
    export let bgStyle: string = "";
</script>

{#each token.text as ch, i}
    {#if ch === ' '}
        <span data-index="{index}" role="tooltip" class={`token-grid-item inline-block mt-2 border-b-2 text-gray-300 dark:text-gray-500`} style={`${underlineStyle} ${bgStyle}`}>
            {#if i === 0}
                <span class="absolute text-xs uppercase -mt-4 text-purple-800 dark:text-purple-300 font-sans">
                    {token.role}
                </span>
            {/if}
            &nbsp;
        </span>
    {:else if ch === '\t'}
        <span data-index="{index}" role="tooltip" class={`token-grid-item inline-block mt-2 border-b-2 text-gray-300 dark:text-gray-500`} style={`${underlineStyle} ${bgStyle}`}>
            {#if i === 0}
                <span class="absolute text-xs uppercase -mt-4 text-purple-800 dark:text-purple-300 font-sans">
                    {token.role}
                </span>
            {/if}
            \t&nbsp;&nbsp;
        </span>
    {:else if ch === '\n'}
        <span data-index="{index}" role="tooltip" class={`token-grid-item inline-block mt-2 border-b-2 text-gray-300 dark:text-gray-500`} style={`${underlineStyle} ${bgStyle}`}>
            {#if i === 0}
                <span class="absolute text-xs uppercase -mt-4 text-purple-800 dark:text-purple-300 font-sans">
                    {token.role}
                </span>
            {/if}
            \n
        </span>
        <div class="basis-full h-full"></div>
    {:else}
        <span data-index="{index}" role="tooltip" class={`token-grid-item inline-block mt-2 border-b-2 ${token.special ? "text-gray-300 dark:text-gray-500" : ""}`} style={`${underlineStyle} ${bgStyle}`}>
            {#if i === 0}
                <span class="absolute text-xs uppercase -mt-4 text-purple-800 dark:text-purple-300 font-sans">
                    {token.role}
                </span>
            {/if}
            {ch}
        </span>
    {/if}
{/each}