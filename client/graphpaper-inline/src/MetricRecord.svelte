<!-- Each metric is displayed as a card. -->
<script lang="ts">
  import { type MetricDef, type MetricVal } from './interfaces';
  import Sparkline from './Sparkline.svelte';

  export let metricDef: MetricDef;
  export let value: MetricVal;

  const minibarPadding = {
    'left': 0,
    'right': 0,
    'top': 5,
    'bottom': 3
  };
</script>

<style>
    .dot-divider:not(:last-child)::after {
        content: "•"; /* Dot separator */
        color: #d1d5db; /* Dot color light mode */ 
    }
    .dark .dot-divider:not(:last-child)::after {
        color: #6b7280; /* Dot color dark mode */
        margin-left: 0.5rem;
    }
</style>

<span class={`dot-divider flex items-center text-xs whitespace-nowrap px-1`} title="{metricDef.description}">
    <span>
        {#if value.constructor === Array}
            <span class={`text-gray-600 dark:text-gray-400 whitespace-nowrap pr-[0.125rem]`}>{metricDef.name}</span>
            <Sparkline values={value} svgClass={"w-8 h-4 inline"} padding={minibarPadding} />
        {:else}
            <span class={`text-gray-600 dark:text-gray-400 whitespace-nowrap pr-[0.125rem]`}>{metricDef.name}</span>
            {#if typeof value === "number"}
                <span class={`font-medium text-gray-700 dark:text-gray-300 `}>{value.toFixed(metricDef.precision)}
                  {#if metricDef.units !== ''}
                    <span class="">{metricDef.units}</span>
                  {/if}
                </span>
            {:else}
                <span class={`font-medium text-center text-gray-700 dark:text-gray-300 `}>{value}
                  {#if metricDef.units !== ''}
                    <span class="">{metricDef.units}</span>
                  {/if}
                </span>
          {/if}
        {/if}
    </span>
</span>