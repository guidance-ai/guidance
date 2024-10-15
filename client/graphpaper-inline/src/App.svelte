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
	let mode: string;

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
			} else if (isMetricMessage(msg)) {
				// TODO(nopdive): Move aggregation to server-side.
				const name = msg.name;
				const value = msg.value;

				if (name in metrics && name in metricDefs) {
					let currVal = metrics[name];
					const metricDef = metricDefs[name];
					if (metricDef.isScalar === false) {
						currVal = currVal as Array<any>;
						metrics[name] = [...currVal.slice(1), value as string | number];
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
				console.log(textComponents);
				console.log(msg.tokens);
			}
			textComponents = textComponents;
		}
	}

	// TODO(nopdive): Restrict metrics while updating style.
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
			units: 'GB',
			description: 'Utilization of RAM.',
			isScalar: true,
			precision: 0,
		},
		'vram': {
			name: 'vram',
			units: 'GB',
			description: 'Utilization of video RAM.',
			isScalar: true,
			precision: 0,
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
	let selectedMetricDef: MetricDef = metricDefs['consumed'];

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

	let metricModes = new Set<string>();
	metricModes.add('avg latency');
	metricModes.add('consumed');

	onMount(() => {
		const msg: StitchMessage = {
			type: "clientmsg",
			content: JSON.stringify({ 'class_name': 'ClientReadyMessage' })
		}
		clientmsg.set(msg);
	});

	const onNavClick = (event: CustomEvent<string>) => {
		mode = event.detail;
		selectedMetricDef = metricDefs[mode];
	};
</script>

<svelte:head>
    <title>graphpaper</title>
    <meta name="description" content="graphpaper"/>
</svelte:head>

<StitchHandler/>
<ResizeListener/>
<div class="w-full min-h-48">
	<!-- Navigation bar -->
	<nav class="sticky top-0 z-30 opacity-90 w-full flex bg-gray-100 text-gray-500 justify-between">
		<div class="pl-2 flex">
			{#each Object.entries(metrics) as [name, value], i}
				<MetricCard value={value} selected={name === selectedMetricDef.name} metricDef={metricDefs[name]} on:forwardclick={onNavClick} enabled={metricModes.has(name)}/>
			{/each}
		</div>
	</nav>
	<!-- Content pane -->
	<section class="w-full">
		<TokenGrid textComponents={textComponents} tokenDetails={tokenDetails} isCompleted={completedExecution} metricDef={selectedMetricDef}/>
	</section>
</div>