<script lang="ts">
    import {scaleSequential} from "d3-scale";
    import {interpolateSpectral} from "d3-scale-chromatic";

    export let token;
    const color = (x: number) => {
        return scaleSequential(interpolateSpectral)(1.0 - x)
    };

    let hovered: boolean = false;
    const handleMouseEnter = () => {
        hovered = true;
    }
    const handleMouseLeave = () => {
        hovered = false;
    }
</script>

{#each token.value as ch, i}
    {#if ch === ' '}
        <span on:mouseenter={handleMouseEnter} on:mouseleave={handleMouseLeave} role="tooltip" class={`inline-block mt-2 border-b-2 ${hovered ? "bg-gray-300 text-gray-700" : "text-gray-300"}`} style={`border-bottom-color: ${color(token.prob)}`}>
            &nbsp;
        </span>
    {:else if ch === '\t'}
        <span on:mouseenter={handleMouseEnter} on:mouseleave={handleMouseLeave} role="tooltip" class={`inline-block mt-2 border-b-2 ${hovered ? "bg-gray-300 text-gray-700" : "text-gray-300"}`} style={`border-bottom-color: ${color(token.prob)}`}>
            \t&nbsp;&nbsp;
        </span>
    {:else if ch === '\n'}
        <span on:mouseenter={handleMouseEnter} on:mouseleave={handleMouseLeave} role="tooltip" class={`inline-block mt-2 border-b-2 ${hovered ? "bg-gray-300 text-gray-700" : "text-gray-300"}`} style={`border-bottom-color: ${color(token.prob)}`}>
            \n
        </span>
        <div class="basis-full h-full"></div>
    {:else}
        <span on:mouseenter={handleMouseEnter} on:mouseleave={handleMouseLeave} role="tooltip" class={`inline-block mt-2 border-b-2 ${hovered ? "bg-gray-300 text-gray-700" : token.special ? "text-gray-300" : ""}`} style={`border-bottom-color: ${color(token.prob)}`}>
            {#if i === 0}
                <span class="absolute text-xs uppercase -mt-4 text-purple-800 font-sans">
                    {token.role}
                </span>
            {/if}
            {ch}
        </span>
    {/if}
{/each}