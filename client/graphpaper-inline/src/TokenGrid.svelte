<script lang="ts">
    import { scaleSequential } from 'd3-scale';
    import { interpolateSpectral } from 'd3-scale-chromatic';
    import {type NodeAttr, type TextOutput} from './stitch';

    export let nodeAttrs: Array<NodeAttr> = [];
    // let specialTokens: Array<string> = [];
    // $: {
    //     for (let nodeAttr of nodeAttrs) {
    //     }
    // }

    const color = (x: number) => {
        return scaleSequential(interpolateSpectral)(1.0 - x)
    };
    const imagePattern = /<-img:(.*)->/;
</script>

<div class="pt-6 pb-6 flex text-gray-800 font-token">
    <!-- Tokens view -->
    <div class="px-4">
        <span class="flex flex-wrap text-sm">
            {#each nodeAttrs as nodeAttr, i}
                {#if nodeAttr.class_name === 'RoleOpenerInput'}
                    <!--{#if i === 0}-->
                    <!--    <div class="basis-full h-2"></div>-->
                    <!--{:else}-->
                    <!--    &lt;!&ndash; Gap between messages &ndash;&gt;-->
                    <!--    {#each {length: 2} as _, i}-->
                    <!--        <div class="basis-full h-0"></div>-->
                    <!--        <span class="inline-block">&nbsp;</span>-->
                    <!--    {/each}-->
                    <!--    <div class="basis-full h-0"></div>-->
                    <!--{/if}-->
                    <!--<span class="inline-block relative">-->
                    <!--    <span class="absolute bottom-7 text-xs mt-2 uppercase -mb-1 text-purple-800 font-sans">-->
                    <!--        {nodeAttr.name}-->
                    <!--    </span>-->
                    <!--    <span class={`inline-block text-gray-300 mt-2 border-b-2 hover:bg-gray-300 hover:text-gray-700`}>-->
                    <!--        &nbsp;-->
                    <!--    </span>-->
                    <!--</span>-->
                    <!-- Do nothing -->
                {:else if nodeAttr.class_name === 'RoleCloserInput'}
                    <!-- Do nothing -->
                {:else if nodeAttr.class_name === 'TextOutput'}
                    {#if nodeAttr.value.match(imagePattern)}
                        <span class="pb-1 inline-block mt-2 border-b-2 hover:bg-gray-300 hover:brightness-75" style={`border-bottom-color: ${color(nodeAttr.prob)}`}>
                            <img src={`data:image/jpeg;base64,${nodeAttr.value.match(imagePattern)?.[1] || ""}`} alt="inlined img"/>
                        </span>
                    <!--{:else if token.is_special == 1.0}-->
                    <!--    {#if token.role !== ""}-->
                    <!--        {#if i == 0}-->
                    <!--            <div class="basis-full h-2"></div>-->
                    <!--        {:else}-->
                    <!--            &lt;!&ndash; Gap between messages &ndash;&gt;-->
                    <!--            {#each {length: 2} as _, i}-->
                    <!--                <div class="basis-full h-0"></div>-->
                    <!--                <span class="inline-block">&nbsp;</span>-->
                    <!--            {/each}-->
                    <!--            <div class="basis-full h-0"></div>-->
                    <!--        {/if}-->
                    <!--        <span class="inline-block relative">-->
                    <!--            <span class="absolute bottom-7 text-xs mt-2 uppercase -mb-1 text-purple-800 font-sans">-->
                    <!--                {token.role}-->
                    <!--            </span>-->

                    <!--            <span class={`inline-block text-gray-300 mt-2 border-b-2 hover:bg-gray-300 hover:text-gray-700`} style={`border-bottom-color: ${color(token.prob)}`}>-->
                    <!--                {token.value}-->
                    <!--            </span>-->
                    <!--        </span>-->
                    <!--    {:else}-->
                    <!--        <div class="basis-full h-0"></div>-->
                    <!--        <span class="inline-block relative">-->
                    <!--            <span class={`inline-block text-gray-300 mt-2 border-b-2 hover:bg-gray-300 hover:text-gray-700`} style={`border-bottom-color: ${color(token.prob)}`}>-->
                    <!--                {token.value}-->
                    <!--            </span>-->
                    <!--        </span>-->
                    <!--    {/if}-->
                    {:else}
                        <!-- Regular tokens -->
                        {#each nodeAttr.value as ch}
                            {#if ch === ' '}
                                <span class={`inline-block mt-2 text-gray-300 border-b-2 hover:bg-gray-300 hover:text-gray-700`} style={`border-bottom-color: ${color(nodeAttr.prob)}`}>
                                    &nbsp;
                                </span>
                            {:else if ch === '\t'}
                                <span class={`inline-block mt-2 text-gray-300 border-b-2 hover:bg-gray-300 hover:text-gray-700`} style={`border-bottom-color: ${color(nodeAttr.prob)}`}>
                                    \t&nbsp;&nbsp;
                                </span>
                            {:else if ch === '\n'}
                                <span class={`inline-block mt-2 text-gray-300 border-b-2 hover:bg-gray-300 hover:text-gray-700`} style={`border-bottom-color: ${color(nodeAttr.prob)}`}>
                                    \n
                                </span>
                                <div class="basis-full h-full"></div>
                            {:else}
                                <span class={`inline-block mt-2 border-b-2 hover:bg-gray-300`} style={`border-bottom-color: ${color(nodeAttr.prob)}`}>
                                    {ch}
                                </span>
                            {/if}
                        {/each}
                    {/if}
                {/if}
            {/each}

            <span class="inline-block mt-2 border-b-2 border-white bg-gray-700 animate-cpulse">
            </span>
        </span>
    </div>
</div>