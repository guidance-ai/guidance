<script lang="ts" context="module">
    </script>

<script lang="ts">
    import Minibar from "./Minibar.svelte";
    import { createEventDispatcher } from 'svelte';
    import { type MetricDef, type MetricVal} from "./interfaces";

    export let metricDef: MetricDef;
    export let value: MetricVal;
    export let selected: boolean;
    export let enabled: boolean;

    const dispatch = createEventDispatcher();

    const minibarPadding = {
        'left': 0,
        'right': 0,
        'top': 4,
        'bottom': 0,
    };

    const onClick = (_: MouseEvent | KeyboardEvent) => {
        dispatch('forwardclick', metricDef.name);
    };
</script>

<div class={`${selected ? "bg-gray-600" : ""} flex flex-col items-center py-1 px-4 hover:bg-gray-800 group ${enabled ? "" : "cursor-not-allowed pointer-events-none"}`} role="button" tabindex="-1" on:click={onClick} on:keydown={onClick}>
    <div class={`uppercase tracking-wider text-xs ${selected ? "text-gray-100" : "text-gray-500"} group-hover:text-gray-100 whitespace-nowrap`}>{metricDef.name}</div>
    {#if value.constructor === Array}
        <Minibar values={value} svgClass={"w-12 h-6"} rectClass={`${selected ? "fill-gray-100" : "fill-gray-700"} group-hover:fill-gray-100`} padding={minibarPadding}/>
    {:else}
        {#if typeof value === "number"}
            <div class={`min-w-12 text-center text-lg ${selected ? "text-gray-100" : "text-gray-700"} group-hover:text-gray-100`}>{value.toFixed(metricDef.precision)}<span class="text-xs pl-1">{metricDef.units}</span></div>
        {:else}
            <div class={`min-w-12 text-center text-lg ${selected ? "text-gray-100" : "text-gray-700"} group-hover:text-gray-100`}>{value}<span class="text-xs pl-1">{metricDef.units}</span></div>
        {/if}
    {/if}
</div>