<!-- Main app that handles token rendering and metrics.

This has bidirectional communication between the guidance server (usually Jupyter kernel) and client.
For upcoming features, we won't be able to send all details over the wire, and will need to operate on client request.
-->
<script lang="ts">
  import './main.css';
  import TokenGrid from './TokenGrid.svelte';
  import ResizeListener from './ResizeListener.svelte';
  import {
    clientmsg,
    type GuidanceMessage,
    isAudioOutput,
    isBacktrackMessage,
    isClientReadyAckMessage,
    isExecutionCompletedMessage,
    isExecutionStartedMessage,
    isImageOutput,
    isMetricMessage,
    isResetDisplayMessage,
    isRoleCloserInput,
    isRoleOpenerInput,
    isTokenOutput,
    isTraceMessage,
    isVideoOutput,
    kernelmsg,
    type NodeAttr,
    state,
    Status,
    type StitchMessage
  } from './stitch';
  import StitchHandler from './StitchHandler.svelte';
  import { onMount } from 'svelte';
  import MetricRecord from './MetricRecord.svelte';
  import Select from './Select.svelte';
  import { metricDefs } from './metrics';
  import type { MetricVal } from './interfaces';
  // import { mockNodeAttrs } from './mocks';

  interface AppState {
    components: Array<NodeAttr>,
    status: Status,
    metrics: Record<string, MetricVal>,
    shownMetrics: Array<string>,
    requireFullReplay: boolean,
    currentMessageId: number,
  }
  let appState: AppState = {
    components: [],
    status: Status.Running,
    shownMetrics: [],
    metrics: {
      'status': Status.Running,
      'wall time': 0,
      'consumed': 0,
      'token reduction': 0,
      'avg latency': 0,
      'cpu': [0., 0., 0., 0., 0.],
      'gpu': [0., 0., 0., 0., 0.],
      'ram': 0,
      'vram': 0,
    },
    requireFullReplay: true,
    currentMessageId: -1,
  };
  // appState.components = mockNodeAttrs;

  let bgField: string = 'Type';
  let underlineField: string = 'Probability';

  const handleMessage = (msg: GuidanceMessage): void => {
    console.log("Received GuidanceMessage:", msg);

    // Duplicates can randomly occur from ipywidget layer.
    if (appState.currentMessageId === msg.message_id) {
      console.log(`Duplicate message detected: ${msg.message_id}`)
      return;
    } else {
      appState.currentMessageId = msg.message_id;
    }

    if (isTraceMessage(msg)) {
      if (isTokenOutput(msg.node_attr)) {
        // console.log(msg.node_attr);
        appState.components.push(msg.node_attr);
      } else if (isRoleOpenerInput(msg.node_attr)) {
        appState.components.push(msg.node_attr);
      } else if (isRoleCloserInput(msg.node_attr)) {
        appState.components.push(msg.node_attr);
      } else if (isAudioOutput(msg.node_attr)) {
        appState.components.push(msg.node_attr);
      } else if (isImageOutput(msg.node_attr)) {
        appState.components.push(msg.node_attr);
      } else if (isVideoOutput(msg.node_attr)) {
        appState.components.push(msg.node_attr);
      } else if (isBacktrackMessage(msg.node_attr)) {
        let numBacktrack = msg.node_attr.n_tokens;
        console.log(`Backtracking ${numBacktrack} tokens.`);
        for (let i = 0; i < numBacktrack; i++) {
          appState.components.pop();
        }
      } else {
        console.log("Unknown trace msg node_attr: ", msg)
      }
    } else if (isExecutionStartedMessage(msg)) {
      appState.requireFullReplay = false;
    } else if (isClientReadyAckMessage(msg)) {
      if (appState.requireFullReplay) {
        console.log('Require full replay and went past completion output message.');
        const msg: StitchMessage = {
          type: 'clientmsg',
          content: JSON.stringify({ 'class_name': 'OutputRequestMessage' })
        };
        clientmsg.set(msg);
      }
    } else if (isResetDisplayMessage(msg)) {
      appState.components = [];
      appState.status = appState.status !== Status.Error ? Status.Running : appState.status;
    } else if (isMetricMessage(msg)) {
      const name = msg.name;
      const value = msg.value;

      if (name in appState.metrics && name in metricDefs) {
        let currVal = appState.metrics[name];
        const metricDef = metricDefs[name];
        if (metricDef.isScalar === false) {
          if (value.constructor === Array) {
            appState.metrics[name] = value;
          } else {
            currVal = currVal as Array<any>;
            appState.metrics[name] = [...currVal.slice(1), value as string | number];
          }
        } else if (metricDef.isScalar === true) {
          appState.metrics[name] = value;
        } else {
          console.error(`Cannot handle metric: ${name}: ${value}.`);
        }

        // NOTE(nopdive): Need to update status too.
        if (name === 'status') {
          appState.status = value as Status;
        }
      }
    } else if (isExecutionCompletedMessage(msg)) {
      appState.status = Status.Done;

      // Good time to save state.
      const savedState = JSON.stringify(appState);
      const stateMessage: StitchMessage = {
        type: 'state',
        content: savedState,
      };
      state.set(stateMessage);

      // console.log(appState.components);
    }

    appState = appState;
  };

  $: if ($state !== undefined && $state.content !== '') {
    // console.log("Client state received.")
    appState = JSON.parse($state.content);
  }
  $: if ($kernelmsg !== undefined && $kernelmsg.content !== '') {
    const msg = JSON.parse($kernelmsg.content);
    handleMessage(msg);
  }
  $: {
    if (appState.status === Status.Running) {
      appState.shownMetrics = [
        'status',
        // 'wall time',
        'cpu',
        'ram',
        'gpu',
        'vram',
      ];
    } else {
      appState.shownMetrics = [
        'status',
        'consumed',
        'token reduction',
        'avg latency',
        // 'wall time',
      ];
    }
  }

  onMount(() => {
    const msg: StitchMessage = {
      type: 'init_stitch',
      content: ''
    };
    clientmsg.set(msg);
  });
</script>

<svelte:head>
  <title>graphpaper</title>
  <meta name="description" content="graphpaper" />
</svelte:head>

<StitchHandler />
<ResizeListener />
<div class="w-full">
  <nav class="sticky top-0 z-50 opacity-90">
    <section class="">
      <div class="text-sm pt-2 pb-2 flex justify-between border-b border-gray-200">
        <!-- Controls -->
        <span class="flex mr-2">
          <Select values={["None", "Type", "Probability", "Latency (ms)"]} classes="ml-4 pl-1 bg-gray-200"
                  defaultValue={"Type"}
                  on:select={(selected) => bgField = selected.detail} />
          <Select values={["None", "Probability", "Latency (ms)"]} classes="border-b-2 pl-1 border-gray-400"
                  defaultValue={"Probability"} on:select={(selected) => underlineField = selected.detail} />
        </span>
        <!-- Metrics -->
        <span class="flex mr-4 text-gray-300 overflow-x-scroll scrollbar-thin scrollbar-track-gray-100 scrollbar-thumb-gray-200">
          {#each appState.shownMetrics as name}
            <MetricRecord value={appState.metrics[name]} metricDef={metricDefs[name]} />
          {/each}
        </span>
      </div>
    </section>
  </nav>

  {#if appState.components.length > 0}
  <!-- Content pane -->
  <section class="w-full min-h-40">
    <TokenGrid components={appState.components}
               isCompleted={['Done', 'Error'].includes(appState.status)}
               isError={appState.status === Status.Error}
               bgField={bgField} underlineField={underlineField} requireFullReplay="{appState.requireFullReplay}" />
  </section>
  {:else}
  <div class="flex items-center justify-center py-6">
    <span class="text-gray-500 text-lg">No tokens to display.</span>
  </div>
  {/if}
</div>
