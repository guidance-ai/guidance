<script lang="ts">
    import { scaleSequential } from 'd3-scale';
    import { interpolateSpectral } from 'd3-scale-chromatic';
    export let tokens: ArrayLike<any> = [];
    
    const color = (x: number) => {
        scaleSequential(interpolateSpectral)(1.0 - x)
    };
    const specialTokens = [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
        "<|endoftext|>",
    ]
    const roles = [
        "system",
        "user",
        "assistant"
    ]
    const specialTokenPattern = /<\|(.*)\|>/;
    const imagePattern = /<-img:(.*)->/;
</script>

<div class="pt-6 pb-6 flex text-gray-800 font-token">
    <!-- Gutter -->
    <div class="border-r-2 border-purple-700 min-w-4"></div>

    <!-- Tokens view -->
    <div class="px-4">
        <span class="flex flex-wrap text-sm">
            {#each tokens as token, i}
            {#if token.content.match(imagePattern)}

            <span class="pb-1 inline-block mt-2 border-b-2 hover:bg-gray-300 hover:brightness-75" style={`border-bottom-color: ${color(token.prob)}`}>
                <img src={`data:image/jpeg;base64,${token.match(imagePattern)?.[1] || ""}`} alt="inlined img"/>
            </span>
            {:else if token.is_special == 1.0}
            {#if token.role !== ""}
            {#if i == 0}
            <div class="basis-full h-2"></div>
            {:else}
            <!-- Gap between messages -->
            {#each {length: 2} as _, i}
            <div class="basis-full h-0"></div>
            <span class="inline-block">&nbsp;</span>
            {/each}
            <div class="basis-full h-0"></div>
            {/if}
            <span class="inline-block relative">
                <span class="absolute bottom-7 text-xs mt-2 uppercase -mb-1 text-purple-800 font-sans">
                    {token.role}
                </span>

                <span class={`inline-block text-gray-300 mt-2 border-b-2 hover:bg-gray-300 hover:text-gray-700`} style={`border-bottom-color: ${color(token.prob)}`}>
                    {token.content}
                </span>
            </span>
            {:else}
            <div class="basis-full h-0"></div>
            <span class="inline-block relative">
                <span class={`inline-block text-gray-300 mt-2 border-b-2 hover:bg-gray-300 hover:text-gray-700`} style={`border-bottom-color: ${color(token.prob)}`}>
                    {token.content}
                </span>
            </span>
            {/if}
            {:else}
            {#each token.content as ch}
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
            <span class={`inline-block mt-2 border-b-2 hover:bg-gray-300`} style={`border-bottom-color: ${color(token.prob)}`}>
                {ch}
            </span>
            {/if}
            {/each}
            {/if}
            {/each}
            <span class="inline-block mt-2 border-b-2 border-white bg-gray-700 animate-cpulse">
                &nbsp;
            </span>
        </span>
    </div>
</div>