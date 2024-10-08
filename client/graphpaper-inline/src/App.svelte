<script lang="ts">
    import './main.css';
    import TokenGrid from './TokenGrid.svelte';
    import ResizeListener from './ResizeListener.svelte';
	import {kernelmsg, clientmsg, type TextOutput, type StitchMessage} from './stitch';
    import StitchHandler from './StitchHandler.svelte';
	import {onMount} from "svelte";
	import MetricCard, {type MetricDef, type MetricVal} from "./MetricCard.svelte";

    let msg: any;
    let tokens: Array<TextOutput> = [];
	let completedExecution: boolean = false;

	$: if ($kernelmsg !== undefined) {
		if ($kernelmsg.content !== '') {
			msg = JSON.parse($kernelmsg.content);
			if (msg.class_name === "TraceMessage") {
				if (msg.node_attr?.class_name === "TextOutput") {
					tokens.push(msg.node_attr);
				} else if (msg.node_attr?.class_name === "RoleOpenerInput") {
					// console.log(msg.node_attr);
				} else if (msg.node_attr?.class_name === "RoleCloserInput") {
					// console.log(msg.node_attr);
				}
			} else if (msg.class_name === "TokenBatchMessage") {
				console.log(msg.tokens);
			} else if (msg.class_name === "ResetDisplayMessage") {
				tokens = [];
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
			tokens = tokens;
		}
	}

	// const textToTextOutput = (text: string) => {
	// 	const textOutput: TextOutput = {
	// 		class_name: "TextOutput",
	// 		value: text,
	// 		is_generated: false,
	// 		token_count: 0,
	// 		prob: 0.
	// 	}
	// 	return textOutput;
	// }
	// let text = "Pack my box with five dozen liquor jugs valued at $12.95, according to sources.";
	// text += "\n<-img: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->";
	// text = "<|system|>\n" + text + " <|end|>\n<|assistant|>\n";
	// text += "{\n    'glossary': {\n        'title': 'example glossary',\n		'GlossDiv': {\n            'title': 'S',\n			'GlossList': {\n                'GlossEntry': {\n                    'ID': 'SGML',\n					'SortAs': 'SGML',\n					'GlossTerm': 'Standard Generalized Markup Language',\n					'Acronym': 'SGML',\n					'Abbrev': 'ISO 8879:1986',\n					'GlossDef': {\n                        'para': 'A meta-markup language, used to create markup languages such as DocBook.',\n						'GlossSeeAlso': ['GML', 'XML']\n                    },\n					'GlossSee': 'markup'\n                }\n            }\n        }\n    }\n}";
	// text += " <|end|>"
	// text += "\n<|user|>\n"
	// text += "A language model is a probabilistic model of a natural language.[1]\nIn 1980, the first significant statistical language model was proposed, and during the decade IBM performed Shannon-style experiments, in which potential sources for language modeling improvement were identified by observing and analyzing the performance of human subjects in predicting or correcting text.[2]  Language models are useful for a variety of tasks, including speech recognition[3] (helping prevent predictions of low-probability (e.g. nonsense) sequences), machine translation,[4] natural language generation (generating more human-like text), optical character recognition, handwriting recognition,[5] grammar induction,[6] and information retrieval.[7][8]  Large language models, currently their most advanced form, are a combination of larger datasets (frequently using words scraped from the public internet), feedforward neural networks, and transformers. They have superseded recurrent neural network-based models, which had previously superseded the pure statistical models, such as word n-gram language model.";
	// text += " <|end|> <|endoftext|>"
	// tokens = (text.match(/(\s+|\S+)/g) || []).map(textToTextOutput);

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
		<TokenGrid tokens={tokens}/>
	</section>
</div>