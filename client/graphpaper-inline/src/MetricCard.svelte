<!-- Each metric is displayed as a card. -->
<script lang="ts">
    import Minibar from "./Minibar.svelte";
    import { type MetricDef, type MetricVal} from "./interfaces";

    export let metricDef: MetricDef;
    export let value: MetricVal;

    const minibarPadding = {
        'left': 0,
        'right': 0,
        'top': 4,
        'bottom': 0,
    };
</script>

<div class={`flex flex-col items-center py-1 px-4 hover:bg-gray-700 group`}>
    <div class={`uppercase tracking-wider text-xs text-gray-500 group-hover:text-gray-100 whitespace-nowrap`}>{metricDef.name}</div>
    {#if value.constructor === Array}
        <Minibar values={value} svgClass={"w-12 h-6"} rectClass={`fill-gray-700 group-hover:fill-gray-100`} padding={minibarPadding}/>
    {:else}
        {#if typeof value === "number"}
            <div class={`min-w-12 text-center text-lg text-gray-700 group-hover:text-gray-100`}>{value.toFixed(metricDef.precision)}<span class="text-xs pl-1">{metricDef.units}</span></div>
        {:else}
            <div class={`min-w-12 text-center text-lg text-gray-700 group-hover:text-gray-100`}>{value}<span class="text-xs pl-1">{metricDef.units}</span></div>
        {/if}
    {/if}
</div>