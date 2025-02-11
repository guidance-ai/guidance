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
    state,
    type GenTokenExtra,
    type GuidanceMessage,
    isClientReadyAckMessage,
    isExecutionCompletedMessage,
    isExecutionStartedMessage,
    isMetricMessage,
    isResetDisplayMessage,
    isRoleCloserInput,
    isRoleOpenerInput,
    isTextOutput,
    isTokensMessage,
    isTraceMessage,
    kernelmsg,
    type NodeAttr,
    type StitchMessage
  } from './stitch';
  import StitchHandler from './StitchHandler.svelte';
  import { onMount } from 'svelte';
  import MetricCard from './MetricCard.svelte';
  import Select from './Select.svelte';
  import { metricDefs } from './metrics';
  import type { MetricVal } from './interfaces';

  interface AppState {
    textComponents: Array<NodeAttr>,
    tokenDetails: Array<GenTokenExtra>,
    metrics: Record<string, MetricVal>,
    isCompleted: boolean,
    isError: boolean,
    requireFullReplay: boolean,
  }
  let appState: AppState = {
    textComponents: [],
    tokenDetails: [],
    metrics: {
      'status': '⟳',
      'cpu': [0.0, 0.0, 0.0, 0.0, 0.0],
      'ram': 0,
      'wall time': 0,
      'avg latency': 0,
      'consumed': 0,
      'token reduction': 0
    },
    isCompleted: false,
    isError: false,
    requireFullReplay: false,
  };
  // appState.textComponents = mockNodeAttrs;
  // appState.tokenDetails = mockGenTokens;

  let showMetrics: boolean = true;
  let bgField: string = 'Type';
  let underlineField: string = 'Probability';

  const handleMessage = (msg: GuidanceMessage): void => {
    if (isTraceMessage(msg)) {
      if (isTextOutput(msg.node_attr)) {
        appState.textComponents.push(msg.node_attr);
      } else if (isRoleOpenerInput(msg.node_attr)) {
        appState.textComponents.push(msg.node_attr);
      } else if (isRoleCloserInput(msg.node_attr)) {
        appState.textComponents.push(msg.node_attr);
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
      appState.textComponents = [];
      appState.isCompleted = false;
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

        if (name === 'status') {
          appState.isError = value === '⚠';
        }
      }
    } else if (isExecutionCompletedMessage(msg)) {
      appState.isCompleted = true;
    } else if (isTokensMessage(msg)) {
      appState.requireFullReplay = false;
      appState.isCompleted = true;
      appState.tokenDetails = msg.tokens;

      // Good time to save state.
      const savedState = JSON.stringify(appState);
      const stateMessage: StitchMessage = {
        type: 'state',
        content: savedState,
      };
      state.set(stateMessage);

      // console.log(appState.textComponents);
      // console.log(appState.tokenDetails);
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

  onMount(() => {
    const msg: StitchMessage = {
      type: 'clientmsg',
      content: JSON.stringify({ 'class_name': 'ClientReadyMessage' })
    };
    clientmsg.set(msg);
  });

  const onMetricClick = (_: any) => {
    showMetrics = !showMetrics;
  };
  const doNothing = (_: any) => {
  };
</script>

<svelte:head>
  <title>graphpaper</title>
  <meta name="description" content="graphpaper" />
</svelte:head>

<StitchHandler />
<ResizeListener />
<div class="w-full min-h-72">
  <nav class="sticky top-0 z-50 opacity-90">
    <!-- Metric bar -->
    {#if showMetrics}
      <section class="w-full flex bg-gray-100 border-b border-gray-200 text-gray-500 justify-between">
        <div class="pl-2 flex">
          {#each Object.entries(appState.metrics) as [name, value]}
            <MetricCard value={value} metricDef={metricDefs[name]} />
          {/each}
        </div>
      </section>
    {/if}

    <!-- Controls -->
    <section>
      <div class="text-sm pt-2 pb-2 flex border-b border-gray-200">
        <div class="ml-4 mr-4 inline-flex justify-between items-center hover:bg-gray-700 hover:text-white"
             tabindex="-1"
             on:click={onMetricClick} on:keydown={doNothing} role="button">
					<span class="text-xs inline-flex flex-col justify-center uppercase tracking-wider">
						Metrics
					</span>
          {#if showMetrics}
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" class="size-3">
              <path d="M8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z" />
              <path fill-rule="evenodd"
                    d="M1.38 8.28a.87.87 0 0 1 0-.566 7.003 7.003 0 0 1 13.238.006.87.87 0 0 1 0 .566A7.003 7.003 0 0 1 1.379 8.28ZM11 8a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"
                    clip-rule="evenodd" />
            </svg>
          {:else}
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" class="size-3">
              <path fill-rule="evenodd"
                    d="M3.28 2.22a.75.75 0 0 0-1.06 1.06l10.5 10.5a.75.75 0 1 0 1.06-1.06l-1.322-1.323a7.012 7.012 0 0 0 2.16-3.11.87.87 0 0 0 0-.567A7.003 7.003 0 0 0 4.82 3.76l-1.54-1.54Zm3.196 3.195 1.135 1.136A1.502 1.502 0 0 1 9.45 8.389l1.136 1.135a3 3 0 0 0-4.109-4.109Z"
                    clip-rule="evenodd" />
              <path
                d="m7.812 10.994 1.816 1.816A7.003 7.003 0 0 1 1.38 8.28a.87.87 0 0 1 0-.566 6.985 6.985 0 0 1 1.113-2.039l2.513 2.513a3 3 0 0 0 2.806 2.806Z" />
            </svg>
          {/if}
        </div>
        <Select values={["None", "Type", "Probability", "Latency (ms)"]} classes="bg-gray-200"
                defaultValue={"Type"}
                on:select={(selected) => bgField = selected.detail} />
        <Select values={["None", "Probability", "Latency (ms)"]} classes="border-b-2 border-gray-400"
                defaultValue={"Probability"} on:select={(selected) => underlineField = selected.detail} />
      </div>
    </section>
  </nav>

  <!-- Content pane -->
  <section class="w-full">
    <TokenGrid textComponents={appState.textComponents} tokenDetails={appState.tokenDetails} isCompleted={appState.isCompleted}
               isError={appState.isError}
               bgField={bgField} underlineField={underlineField} requireFullReplay="{appState.requireFullReplay}" />
  </section>
</div>