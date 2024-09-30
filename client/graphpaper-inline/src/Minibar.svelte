<!-- Mini bar graph for tracking distributions or values over time scaled from 0 to 1. -->

<script lang="ts">
	import {scaleLinear} from 'd3-scale';

    export let values: ArrayLike<Number>;
	export let svgClass: string;
	export let rectClass: string;
	export let padding = {
		'left': 0,
		'right': 0,
		'top': 0,
		'bottom': 0,
	};

	let height = 0;
	let width = 0;
	$: xScale = scaleLinear()
		.domain([0, values.length])
		.range([0, width - padding.right]);

	$: yScale = scaleLinear()
		.domain([0, 1])
		.range([height - padding.bottom, padding.top]);

	$: innerWidth = width - (padding.left + padding.right);
	$: barWidth = innerWidth / values.length;
</script>

<div class="flex items-center" bind:clientHeight={height} bind:clientWidth={width}>
	<svg class={svgClass}>
		<g>
		{#each values as val, i}
			<rect x={xScale(i)} y={yScale(val)} width={barWidth-1} height={yScale(0) - yScale(val)} class={rectClass}></rect>
			{/each}
		</g>
	</svg>
</div>