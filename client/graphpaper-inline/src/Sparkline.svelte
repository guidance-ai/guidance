<!-- Sparkline for tracking distributions or values over time scaled from 0 to 1. -->

<script lang="ts">
  import { scaleLinear } from 'd3-scale';

  export let values;
  export let svgClass: string;
  export let padding = {
    'left': 0,
    'right': 0,
    'top': 0,
    'bottom': 0
  };
  $: typedValues = values as Array<number>;

  let height = 0;
  let width = 0;
  $: xScale = scaleLinear()
    .domain([0, typedValues.length-1])
    .range([padding.left, padding.left + width - padding.right]);

  $: yScale = scaleLinear()
    .domain([0, 1])
    .range([height - padding.bottom, padding.top]);

  $: pathData = typedValues.map((v, i) => ({
    x: xScale(i),
    y: yScale(v),
  }))
</script>

<div class="inline-block font-medium text-gray-700 dark:text-gray-300" bind:clientHeight={height} bind:clientWidth={width}>
  <svg class={svgClass}>
    <g>
      <path d="{pathData.map((v, i) => `${i === 0 ? 'M' : 'L'} ${v.x} ${v.y}`).join(' ')}" fill="none" stroke-width="1.25" stroke="#374151" class="stroke-gray-700 dark:stroke-gray-300"/>
    </g>
  </svg>
</div>