<script lang="ts">
    import './main.css';
    import TokenGrid from './TokenGrid.svelte';
    import ResizeListener from './ResizeListener.svelte';
	import {
		kernelmsg,
		clientmsg,
		type StitchMessage,
		type NodeAttr,
		type GenToken,
		isTraceMessage,
		isTextOutput,
		isRoleOpenerInput,
		isRoleCloserInput,
		isResetDisplayMessage,
		isMetricMessage,
		isExecutionCompletedMessage,
		isExecutionCompletedOutputMessage,
	} from './stitch';
    import StitchHandler from './StitchHandler.svelte';
	import {onMount} from "svelte";
	import {type MetricDef, type MetricVal} from "./interfaces";
	import MetricCard from "./MetricCard.svelte";
	import {mockGenTokens, mockNodeAttrs} from "./mocks";

    let msg: any;
    let textComponents: Array<NodeAttr> = [];
	let tokenDetails: Array<GenToken> = [];
	let completedExecution: boolean = false;
	let showMetrics: boolean = true;

	textComponents = mockNodeAttrs;
	tokenDetails = mockGenTokens;

	$: if ($kernelmsg !== undefined) {
		if ($kernelmsg.content !== '') {
			msg = JSON.parse($kernelmsg.content);
			if (isTraceMessage(msg)) {
				if (isTextOutput(msg.node_attr)) {
					textComponents.push(msg.node_attr);
				} else if (isRoleOpenerInput(msg.node_attr)) {
					textComponents.push(msg.node_attr)
				} else if (isRoleCloserInput(msg.node_attr)) {
					textComponents.push(msg.node_attr)
				}
			} else if (isResetDisplayMessage(msg)) {
				textComponents = [];
				completedExecution = false;
			} else if (isMetricMessage(msg)) {
				const name = msg.name;
				const value = msg.value;

				if (name in metrics && name in metricDefs) {
					let currVal = metrics[name];
					const metricDef = metricDefs[name];
					if (metricDef.isScalar === false) {
						if (value.constructor === Array) {
							metrics[name] = value;
						} else {
							currVal = currVal as Array<any>;
							metrics[name] = [...currVal.slice(1), value as string | number];
						}
					} else if (metricDef.isScalar === true) {
						metrics[name] = value;
					} else {
						console.log(`Cannot handle metric: ${name}: ${value}.`)
					}
				}
			} else if (isExecutionCompletedMessage(msg)) {
				completedExecution = true;
			} else if (isExecutionCompletedOutputMessage(msg)) {
				tokenDetails = msg.tokens;
			}
			textComponents = textComponents;
		}
	}

	const metricDefs: Record<string, MetricDef> = {
		'status': {
			name: 'status',
			units: '',
			description: 'Determines whether engine is running, completed or in error.',
			isScalar: true,
			precision: 0,
		},
		'cpu': {
			name: 'cpu',
			units: '%',
			description: 'Average utilization across CPU cores.',
			isScalar: false,
			precision: 1,
		},
		'gpu': {
			name: 'gpu',
			units: '%',
			description: 'Average utilization across GPUs.',
			isScalar: false,
			precision: 1,
		},
		'ram': {
			name: 'ram',
			units: 'G',
			description: 'Utilization of RAM.',
			isScalar: true,
			precision: 1,
		},
		'vram': {
			name: 'vram',
			units: 'G',
			description: 'Utilization of video RAM.',
			isScalar: true,
			precision: 1,
		},
		'wall time': {
			name: 'wall time',
			units: 's',
			description: 'Time taken from initial display to engine completion.',
			isScalar: true,
			precision: 1,
		},
		'avg latency': {
			name: 'avg latency',
			units: 'ms',
			description: 'Average roundtrip latency per token',
			isScalar: true,
			precision: 0,
		},
		'consumed': {
			name: 'consumed',
			units: 'tkn',
			description: 'Total tokens consumed by language model.',
			isScalar: true,
			precision: 0,
		},
		'token reduction': {
			name: 'token reduction',
			units: '%',
			description: 'Total tokens consumed by language model divided by total tokens.',
			isScalar: true,
			precision: 0,
		}
	};
	const metrics: Record<string, MetricVal> = {
		'status': '✓',
		'gpu': [0.0, 0.0, 0.0, 0.0, 0.0],
		'vram': 0,
		// 'cpu': [0.0, 0.0, 0.0, 0.0, 0.0],
		// 'ram': 0,
		'wall time': 0,
		'avg latency': 0,
		'consumed': 0,
		'token reduction': 0,
	};

	onMount(() => {
		const msg: StitchMessage = {
			type: "clientmsg",
			content: JSON.stringify({ 'class_name': 'ClientReadyMessage' })
		}
		clientmsg.set(msg);
	});
</script>

<svelte:head>
    <title>graphpaper</title>
    <meta name="description" content="graphpaper"/>
</svelte:head>

<StitchHandler/>
<ResizeListener/>
<div class="w-full min-h-96">
	<nav class="sticky top-0 z-50 opacity-90">
		<!-- Metric bar -->
		{#if showMetrics}
			<section class="w-full flex bg-gray-100 border-b border-gray-200 text-gray-500 justify-between">
				<div class="pl-2 flex">
					{#each Object.entries(metrics) as [name, value]}
						<MetricCard value={value} metricDef={metricDefs[name]}/>
					{/each}
				</div>
			</section>
		{/if}
		<!-- Controls -->
		<section>
			<div class="text-sm pt-2 pb-2 flex border-b border-gray-200">
				<div class="ml-4 mr-4 inline-flex justify-between items-center">
					<span class="text-xs inline-flex flex-col justify-center uppercase tracking-wider">
						Metrics
					</span>
					<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" class="size-3">
						<path d="M8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z" />
						<path fill-rule="evenodd" d="M1.38 8.28a.87.87 0 0 1 0-.566 7.003 7.003 0 0 1 13.238.006.87.87 0 0 1 0 .566A7.003 7.003 0 0 1 1.379 8.28ZM11 8a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" clip-rule="evenodd" />
					</svg>
				</div>
				<div class="mr-4 inline-flex justify-between items-center bg-gray-200">
					<span class="">
						Type
					</span>
					<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" class="size-4">
						<path fill-rule="evenodd" d="M4.22 6.22a.75.75 0 0 1 1.06 0L8 8.94l2.72-2.72a.75.75 0 1 1 1.06 1.06l-3.25 3.25a.75.75 0 0 1-1.06 0L4.22 7.28a.75.75 0 0 1 0-1.06Z" clip-rule="evenodd" />
					</svg>
				</div>
				<div class="mr-4 inline-flex justify-between items-center border-b-2 border-gray-400">
					<span class="">
						Probability
					</span>
					<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" class="size-4">
						<path fill-rule="evenodd" d="M4.22 6.22a.75.75 0 0 1 1.06 0L8 8.94l2.72-2.72a.75.75 0 1 1 1.06 1.06l-3.25 3.25a.75.75 0 0 1-1.06 0L4.22 7.28a.75.75 0 0 1 0-1.06Z" clip-rule="evenodd" />
					</svg>
				</div>
			</div>
		</section>
	</nav>
	<!-- Content pane -->
	<section class="w-full">
		<TokenGrid textComponents={textComponents} tokenDetails={tokenDetails} isCompleted={completedExecution}/>
	</section>
</div>