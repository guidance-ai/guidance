<script lang="ts" context="module">
    export interface MetricDef {
        name: string,
        units: string,
        description: string,
        isScalar: boolean,
        precision: number,
    }
    export type MetricVal = string | number | ArrayLike<number> | ArrayLike<string>;
</script>

<script lang="ts">
    import Minibar from "./Minibar.svelte";
    export let metricDef: MetricDef;
    export let value: MetricVal;
    export let i: number;

    const minibarPadding = {
        'left': 0,
        'right': 0,
        'top': 4,
        'bottom': 0,
    }
    const selectedIndex = 5;
</script>

<div class={`${i === selectedIndex ? "bg-gray-600" : ""} flex flex-col items-center py-1 px-4 hover:bg-gray-600 hover:cursor-pointer group`}>
    <div class={`uppercase tracking-wider text-xs ${i === selectedIndex ? "text-gray-100" : "text-gray-500"} group-hover:text-gray-100 whitespace-nowrap`}>{metricDef.name}</div>
    {#if value.constructor === Array}
        <Minibar values={value} svgClass={"w-12 h-6"} rectClass={`${i === selectedIndex ? "fill-gray-100" : "fill-gray-700"} group-hover:fill-gray-100`} padding={minibarPadding}/>
    {:else}
        {#if typeof value === "number"}
            <div class={`text-lg ${i === selectedIndex ? "text-gray-100" : "text-gray-700"} group-hover:text-gray-100`}>{value.toFixed(metricDef.precision)}<span class="text-xs pl-1">{metricDef.units}</span></div>
        {:else}
            <div class={`text-lg ${i === selectedIndex ? "text-gray-100" : "text-gray-700"} group-hover:text-gray-100`}>{value}<span class="text-xs pl-1">{metricDef.units}</span></div>
        {/if}
    {/if}
</div>