<script lang="ts">
    import './main.css';
    import TokenGrid from './TokenGrid.svelte';
    import ResizeListener from './ResizeListener.svelte';
	import {
		kernelmsg,
		clientmsg,
		type StitchMessage,
		type NodeAttr,
		isTraceMessage,
		isTextOutput, isRoleOpenerInput, isRoleCloserInput
	} from './stitch';
    import StitchHandler from './StitchHandler.svelte';
	import {onMount} from "svelte";
	import MetricCard, {type MetricDef, type MetricVal} from "./MetricCard.svelte";
	import {mockTokens} from "./mocks";

    let msg: any;
    let nodeAttrs: Array<NodeAttr> = [];
	let completedExecution: boolean = false;

	$: if ($kernelmsg !== undefined) {
		if ($kernelmsg.content !== '') {
			msg = JSON.parse($kernelmsg.content);
			if (isTraceMessage(msg)) {
				if (isTextOutput(msg.node_attr)) {
					nodeAttrs.push(msg.node_attr);
				} else if (isRoleOpenerInput(msg.node_attr)) {
					nodeAttrs.push(msg.node_attr)
				} else if (isRoleCloserInput(msg.node_attr)) {
					nodeAttrs.push(msg.node_attr)
				}
			} else if (msg.class_name === "TokenBatchMessage") {
				console.log(msg.tokens);
			} else if (msg.class_name === "ResetDisplayMessage") {
				nodeAttrs = [];
			} else if (msg.class_name === "MetricMessage") {
				const name = msg.name;
				const value = msg.value;

				if (name in metrics && name in metricDefs) {
					let currVal = metrics[name];
					const metricDef = metricDefs[name];
					if (metricDef.isScalar === false && currVal instanceof Array) {
						metrics[name] = [...currVal.slice(1), value];
					} else if (metricDef.isScalar === true) {
						metrics[name] = value;
					} else {
						console.log(`Cannot handle metric: ${name}: ${value}.`)
					}
				}
			} else if (msg.class_name === 'JupyterCellExecutionCompletedMessage') {
				completedExecution = true;
			}
			nodeAttrs = nodeAttrs;
		}
	}

	nodeAttrs = mockTokens;

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
	}

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
	}

	onMount(() => {
		const msg: StitchMessage = {
			type: "clientmsg",
			content: JSON.stringify({ 'class_name': 'ClientReadyMessage' })
		}
		clientmsg.set(msg);
	})
</script>

<svelte:head>
    <title>graphpaper</title>
    <meta name="description" content="graphpaper"/>
</svelte:head>

<StitchHandler/>
<ResizeListener/>
<div class="w-full">
	<!-- Navigation bar -->
	<nav class="sticky top-0 z-50 opacity-90 w-full flex bg-gray-100 text-gray-500 justify-between">
		<div class="pl-2 flex">
			{#each Object.entries(metrics) as [name, value], i}
				<MetricCard value={value} i={i} metricDef={metricDefs[name]} />
			{/each}
		</div>
	</nav>
	<!-- Content pane -->
	<section class="w-full">
		<TokenGrid nodeAttrs={nodeAttrs} isCompleted={completedExecution}/>
	</section>
</div>