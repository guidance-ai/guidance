<!-- Token grid that exposes each token and hover info. -->
<script lang="ts">
  import {
    isRoleOpenerInput,
    isRoleCloserInput,
    isTokenOutput,
    isTextOutput,
    isAudioOutput,
    type NodeAttr,
    type RoleOpenerInput,
    isImageOutput,
    isVideoOutput,
    type AudioOutput,
    type VideoOutput,
    type ImageOutput,
  } from "./stitch";
  import CustomAudio from "./CustomAudio.svelte";
  import CustomVideo from "./CustomVideo.svelte";
  import TokenGridItem from "./TokenGridItem.svelte";
  import {
    type FlatToken,
    type TokenCallback,
    type MultimodalNode,
    type MediaType,
  } from "./interfaces";
  import { longhover } from "./longhover";
  import DOMPurify from "dompurify";

  import {interpolateGreens, interpolateBlues} from "d3-scale-chromatic";

  export let components: Array<NodeAttr>;
  export let isCompleted: boolean;
  export let isError: boolean;
  export let requireFullReplay: boolean = false;
  export let bgField: string = "Token";
  export let underlineField: string = "Probability";
  export let backtrackCount: number = 0;
  export let resetCount: number = 0;

  let underline: TokenCallback = (_: FlatToken) => "";
  let bg: TokenCallback = (_: FlatToken) => "";

  const tokenDisplayValue = (x: FlatToken, s: string) => {
    if (s === "Probability") {
      return x.prob?.toFixed(3);
    } else if (s === "Latency (ms)") {
      return x.latency_ms?.toFixed(0);
    } else if (s === "Type") {
      if (x.is_input) {
        return "Input";
      } else if (x.is_force_forwarded) {
        return "Forwarded";
      } else if (x.is_generated) {
        return "Generated";
      }
    } else if (s === "None") {
      return "";
    }
  };

  const getBrightness = (rgba: string) => {
    const rgbMatch = rgba.match(/rgba?\(\s*(\d+),\s*(\d+),\s*(\d+)/);
    if (!rgbMatch) {
      console.error("Invalid RGBA format.");
      return 0;
    }

    const r = parseInt(rgbMatch[1], 10);
    const g = parseInt(rgbMatch[2], 10);
    const b = parseInt(rgbMatch[3], 10);
    return r * 0.299 + g * 0.587 + b * 0.114;
  };

  const getTextColor = (backgroundColor: string) => {
    const brightness = getBrightness(backgroundColor);
    return brightness > 186 ? "rgba(0, 0, 0, 1)" : "rgba(255, 255, 255, 1)"; // Black for light bg, white for dark bg
  };

  const bgStyle = (
    x: number | undefined,
    color?: ((x: number) => string) | undefined
  ) => {
    if (x === undefined) {
      return "";
    }

    // let colorVal = interpolateYlOrRd(x * 0.85);
    let colorVal = interpolateBlues(x);
    if (color !== undefined) {
      colorVal = color(x);
    }
    let textColor = getTextColor(colorVal);
    return `background-color: ${colorVal}; color: ${textColor};`;
  };

  const underlineStyle = (
    x: number | undefined,
    color?: ((x: number) => string) | undefined
  ) => {
    if (x === undefined) {
      return "";
    }

    let colorVal = interpolateGreens(x * 0.7);
    if (color !== undefined) {
      colorVal = color(x);
    }
    return `border-bottom-color: ${colorVal};`;
  };

  const bgTokenStyle = (x: FlatToken) => {
    let color = "";
    if (x.is_input) {
      color = "rgba(255, 255, 255, 0)";
    } else if (x.is_force_forwarded) {
      color = "rgba(243, 244, 246, 1)";
    } else if (x.is_generated) {
      color = "rgba(229, 231, 235, 1)";
    } else {
      // console.log(`ERROR: token ${x.text} does not have emit flags.`);
      // Make slightly off white for error detection without console spam
      color = "rgba(255, 255, 254, 0)";
    }
    return `background-color: ${color};`;
  };

  function findTargetWords(
    text: string,
    targetWords: string[]
  ): [number, number, string][] {
    // NOTE(nopdive): Not the most efficient approach, but there aren't many special words anyway.

    const results: [number, number, string][] = [];
    for (const targetWord of targetWords) {
      let start = 0;
      while ((start = text.indexOf(targetWord, start)) !== -1) {
        results.push([start, start + targetWord.length, targetWord]);
        start += targetWord.length;
      }
    }

    results.sort((a, b) => (a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]));
    return results;
  }

  const checkOverlapped = (
    tokenStart: number,
    tokenEnd: number,
    matchStart: number,
    matchEnd: number
  ) => {
    let overlapped = false;
    let noSpecialOverride = false;

    if (tokenStart <= matchStart && tokenEnd - 1 >= matchStart) {
      // Match with token leading
      overlapped = true;
    } else if (tokenStart <= matchEnd - 1 && tokenEnd >= matchEnd) {
      // Match with token trailing
      overlapped = true;

      // Visually looks bad when next word is also greyed out.
      noSpecialOverride = true;
    } else if (tokenStart >= matchStart && tokenEnd <= matchEnd) {
      // Match with token equal or within
      overlapped = true;
    }

    return [overlapped, noSpecialOverride];
  };

  let multimodalNodes: MultimodalNode[] = [];
  let tokens: Array<FlatToken> = [];
  let activeOpenerRoles: Array<RoleOpenerInput> = [];
  let activeCloserRoleText: Array<string> = [];
  let specialSet: Set<string> = new Set<string>();
  let namedRoleSet: Record<string, string> = {};
  let displayedRoles: Set<string> = new Set<string>();
  let roleCounters: Record<string, number> = {};
  let currentTokenIndex: number = 0;
  let statCounter: Record<string, number> = {};
  let lastBacktrackCount: number = 0;
  let lastResetCount: number = 0;
  
  // Reset currentTokenIndex and clear tokens when backtracking occurs
  $: if (backtrackCount !== lastBacktrackCount) {
    currentTokenIndex = 0;
    tokens = [];  // Clear existing tokens
    multimodalNodes = [];  // Clear existing media
    activeOpenerRoles = [];  // Reset role tracking
    activeCloserRoleText = [];  // Reset role tracking
    specialSet.clear();  // Clear special token set
    namedRoleSet = {};  // Clear named role mapping
    displayedRoles.clear();  // Clear displayed roles tracking
    roleCounters = {};  // Clear role counters
    lastBacktrackCount = backtrackCount;
  }
  
  // Reset state when ResetDisplayMessage is received
  $: if (resetCount !== lastResetCount) {
    currentTokenIndex = 0;
    tokens = [];
    multimodalNodes = [];
    activeOpenerRoles = [];
    activeCloserRoleText = [];
    specialSet.clear();
    namedRoleSet = {};
    displayedRoles.clear();
    roleCounters = {};
    lastResetCount = resetCount;
  }
  
  $: {
    if (components.length === 0) {
      // Reset
      tokens = []
      multimodalNodes = [];
      activeOpenerRoles = [];
      activeCloserRoleText = [];
      specialSet.clear();
      namedRoleSet = {};
      displayedRoles.clear();
      roleCounters = {};
      currentTokenIndex = 0;
    } else if (currentTokenIndex > components.length) {
      // Handle case where components were removed (e.g., after reset + partial replay)
      currentTokenIndex = components.length;
    }

    for (; currentTokenIndex < components.length; currentTokenIndex += 1) {
      const nodeAttr = components[currentTokenIndex];
      const createMediaNode = (
        mediaType: MediaType,
        node: AudioOutput | VideoOutput | ImageOutput
      ): MultimodalNode => {
        return {
          type: "media",
          data: {
            type: mediaType,
            value: node.value,
            format: node.format,
            context: {
              roleStack: [...activeOpenerRoles], // Clone current role stack
              index: currentTokenIndex,
            },
          },
        };
      };

      if (isRoleOpenerInput(nodeAttr)) {
        activeOpenerRoles.push(nodeAttr);
        activeCloserRoleText.push(nodeAttr.closer_text || "");
        
        // Increment counter for this role type
        const roleName = nodeAttr.name || "";
        roleCounters[roleName] = (roleCounters[roleName] || 0) + 1;
      } else if (isRoleCloserInput(nodeAttr)) {
        // Close the most recent role
        if (activeOpenerRoles.length > 0) {
          activeOpenerRoles.pop();
          activeCloserRoleText.pop();
        }
      } else if (isAudioOutput(nodeAttr)) {
        multimodalNodes.push(createMediaNode("audio", nodeAttr));
      } else if (isImageOutput(nodeAttr)) {
        multimodalNodes.push(createMediaNode("image", nodeAttr));
      } else if (isVideoOutput(nodeAttr)) {
        multimodalNodes.push(createMediaNode("video", nodeAttr));
      } else if (isTokenOutput(nodeAttr) || (isTextOutput(nodeAttr) && !nodeAttr.value.includes("<|im_start|>") && !nodeAttr.value.includes("<|im_end|>"))) {
        // console.log("Processing token:", nodeAttr.value, "Active roles:", activeOpenerRoles.map(r => r.name));
        if (activeOpenerRoles.length === 0) {
          if (
            activeCloserRoleText.length !== 0 &&
            activeCloserRoleText[activeCloserRoleText.length - 1] ===
              nodeAttr.value
          ) {
            const token: FlatToken = {
              text: nodeAttr.value,
              prob: isTokenOutput(nodeAttr) ? nodeAttr.token.prob : 0,
              latency_ms: nodeAttr.latency_ms,
              role: "",
              special: true,
              is_input: nodeAttr.is_input,
              is_force_forwarded: nodeAttr.is_force_forwarded,
              is_generated: nodeAttr.is_generated,
              is_masked: isTokenOutput(nodeAttr) ? nodeAttr.token.masked : false,
              top_k: (isTokenOutput(nodeAttr) && nodeAttr.top_k !== null) ? nodeAttr.top_k.map(t => ({
                text: t.bytes,
                prob: t.prob,
                is_masked: t.masked,
                latency_ms: 0,
              })) : undefined,
            };
            specialSet.add(token.text);
            // TODO: handle interleaving tokens with multimodal data
            // multimodalNodes.push({ type: "token", data: token });
            statCounter["latency.max"] = Math.max(token.latency_ms, statCounter["latency.max"] || 0);
            tokens.push(token)
            activeCloserRoleText.pop();
          } else {
            const token: FlatToken = {
              text: nodeAttr.value,
              prob: isTokenOutput(nodeAttr) ? nodeAttr.token.prob : 0,
              latency_ms: nodeAttr.latency_ms,
              role: "",
              special: false,
              is_input: nodeAttr.is_input,
              is_force_forwarded: nodeAttr.is_force_forwarded,
              is_generated: nodeAttr.is_generated,
              is_masked: isTokenOutput(nodeAttr) ? nodeAttr.token.masked : false,
              top_k: (isTokenOutput(nodeAttr) && nodeAttr.top_k !== null) ? nodeAttr.top_k.map(t => ({
                text: t.bytes,
                prob: t.prob,
                is_masked: t.masked,
                latency_ms: 0,
              })) : undefined,
            };
            // multimodalNodes.push({ type: "token", data: token });
            statCounter["latency.max"] = Math.max(token.latency_ms, statCounter["latency.max"] || 0);
            tokens.push(token);
          }
        } else {
          const activeOpenerRole =
            activeOpenerRoles[activeOpenerRoles.length - 1];
          
          // Check if this is a role marker token (like "<|im_start|>assistant\n")
          const isRoleMarker = nodeAttr.value.includes("<|im_start|>") || 
                              nodeAttr.value.includes("<|im_end|>");
          
          if (isRoleMarker) {
            // Hide role marker tokens - don't add to display
            specialSet.add(nodeAttr.value);
          } else {
            // Check if we've already displayed this role section
            const roleName = activeOpenerRole.name || "";
            const roleCount = roleCounters[roleName] || 0;
            const roleKey = `${roleName}-${roleCount}`;
            const shouldShowRole = !displayedRoles.has(roleKey);
            
            if (shouldShowRole) {
              displayedRoles.add(roleKey);
            }
            
            const token: FlatToken = {
              text: nodeAttr.value, 
              prob: isTokenOutput(nodeAttr) ? nodeAttr.token.prob : 0, 
              latency_ms: nodeAttr.latency_ms,
              role: shouldShowRole ? (activeOpenerRole.name || "") : "",
              special: false,
              is_input: nodeAttr.is_input, 
              is_force_forwarded: nodeAttr.is_force_forwarded,
              is_generated: nodeAttr.is_generated,
              is_masked: isTokenOutput(nodeAttr) ? nodeAttr.token.masked : false,
              top_k: (isTokenOutput(nodeAttr) && nodeAttr.top_k !== null) ? nodeAttr.top_k.map(t => ({
                text: t.bytes,
                prob: t.prob,
                is_masked: t.masked,
                latency_ms: 0,
              })) : undefined,
            };
            statCounter["latency.max"] = Math.max(token.latency_ms, statCounter["latency.max"] || 0);
            tokens.push(token);
          }
        }
            }
        }
        // NOTE(nopdive): Often the closer text is missing at the end of output.
        if (activeOpenerRoles.length !== 0 || activeCloserRoleText.length !== 0) {
            // console.log("Opener and closer role texts did not balance.")
        }

    // Visual updates
    if (!isCompleted || isError) {
      underline = (_: FlatToken) => "border: none;";
    } else if (underlineField === "Probability") {
      underline = (x: FlatToken) => underlineStyle(x.prob);
    } else if (underlineField === "Latency (ms)") {
      underline = (x: FlatToken) =>
        underlineStyle(
          Math.log(x.latency_ms) / Math.log(statCounter["latency.max"])
        );
    } else {
      underline = (_: FlatToken) => "border: none;";
    }

    if (!isCompleted || isError) {
      // bg = (_: Token) => "";
      bg = (x: FlatToken) => bgTokenStyle(x);
    } else if (bgField === "Type") {
      bg = (x: FlatToken) => bgTokenStyle(x);
    } else if (bgField === "Probability") {
      bg = (x: FlatToken) => bgStyle(x.prob);
    } else if (bgField === "Latency (ms)") {
      bg = (x: FlatToken) =>
        bgStyle(Math.log(x.latency_ms) / Math.log(statCounter["latency.max"]));
      console.log(statCounter["latency.max"]);
    } else {
      bg = (_: FlatToken) => "";
    }

    // End bookkeeping (svelte)
    isCompleted = isCompleted;
    isError = isError;
    components = components;
    multimodalNodes = multimodalNodes;
    tokens = tokens;
  }

  let tooltip: HTMLElement;
  let tooltipX = 0;
  let tooltipY = 0;
  let tooltipToken: FlatToken;
  const mouseLongHoverDuration = 200;

  const handleLongMouseOver = (event: CustomEvent<MouseEvent>) => {
    const target = event.detail.target as HTMLElement;
    if (target.matches(".token-grid-item")) {
      const index = target.dataset.index;
      const positionXOffset = 15;
      const positionYOffset = 10;

      // Add tooltip
      const rect = target.getBoundingClientRect();
      tooltipX = rect.left + window.scrollX + rect.width / 2 + positionXOffset;
      tooltipY = rect.bottom + window.scrollY + positionYOffset;
      
      const indexNum = Number(index);
    //   const node = multimodalNodes[indexNum];
    //   if (node.type === "token") {
      tooltipToken = tokens[indexNum];
      
      // Show tooltip first to get accurate dimensions
      tooltip.style.display = "block";
      
      // Use requestAnimationFrame to ensure dimensions are calculated after render
      requestAnimationFrame(() => {
        // Adjust if near edge of viewport
        const tooltipRect = tooltip.getBoundingClientRect();
        
        // Check right edge
        if (tooltipX + tooltipRect.width > window.innerWidth) {
          tooltipX = window.innerWidth - tooltipRect.width - 10;
        }
        
        // Check bottom edge - this is the key fix for first hover
        if (tooltipY + tooltipRect.height > window.innerHeight) {
          // Position above the element instead of below
          tooltipY = rect.top + window.scrollY - tooltipRect.height - positionYOffset;
        }
        
        // Ensure tooltip stays within left edge
        if (tooltipX < 10) {
          tooltipX = 10;
        }
        
        // Ensure tooltip stays within top edge
        if (tooltipY < 10) {
          tooltipY = 10;
        }
      });
    }
  };

  let highlightPrevColor = "";
  let highlightPrevBackgroundColor = "";
  const handleMouseOver = (event: MouseEvent) => {
    const target = event.target as HTMLElement;
    if (target.matches(".token-grid-item")) {
      const index = target.dataset.index;
      const siblingsIncludingSelf = target.parentElement?.querySelectorAll(
        `.token-grid-item[data-index="${index}"]`
      );

      // Add highlight
      if (siblingsIncludingSelf) {
        for (const sibling of siblingsIncludingSelf) {
          const htmlSibling = sibling as HTMLElement;
          highlightPrevColor = htmlSibling.style.color;
          highlightPrevBackgroundColor = htmlSibling.style.backgroundColor;
          htmlSibling.style.color = "rgb(249, 250, 251)";
          htmlSibling.style.backgroundColor = "rgb(75, 85, 99)";
        }
      }
    }
  };

  const handleLongMouseOut = (event: CustomEvent<MouseEvent>) => {
    const target = event.detail.target as HTMLElement;
    if (target.matches(".token-grid-item")) {
      // Remove tooltip
      tooltip.style.display = "none";
    }
  };

  const handleMouseOut = (event: MouseEvent) => {
    const target = event.target as HTMLElement;
    if (target.matches(".token-grid-item")) {
      const index = target.dataset.index;
      const siblingsIncludingSelf = target.parentElement?.querySelectorAll(
        `.token-grid-item[data-index="${index}"]`
      );

      // Remove highlight
      if (siblingsIncludingSelf) {
        for (const sibling of siblingsIncludingSelf) {
          const htmlSibling = sibling as HTMLElement;
          htmlSibling.style.color = highlightPrevColor;
          htmlSibling.style.backgroundColor = highlightPrevBackgroundColor;
        }
      }
    }
  };
  const doNothing = (_: any) => {};
  const renderText = (text: string) => {
    return DOMPurify.sanitize(
      text
        .replaceAll(" ", "&nbsp;")
        .replaceAll("\t", "\\t")
        .replaceAll("\n", "\\n")
    );
  };
  const continuationToken = {
    text: "...",
    prob: 1,
    latency_ms: 0,
    role: "",
    special: false,
    is_input: true,
    is_force_forwarded: false,
    is_generated: true,
  };
</script>

<!-- Tooltip -->
<div
  bind:this={tooltip}
  class="px-1 pt-1 pb-3 absolute opacity-95 bg-white shadow border border-gray-300 pointer-events-none z-50"
  style="top: {tooltipY}px; left: {tooltipX}px; display: none;"
>
  <div>
    {#if tooltipToken}
      <div class={`col-1 flex flex-col items-center`}>
        <div class="text-2xl px-1 pb-1 text-left w-full bg-white">
          <div class="mb-5 mt-1">
            <TokenGridItem
              token={tooltipToken}
              index={-1}
              underlineStyle={underline(tooltipToken)}
              bgStyle={bg(tooltipToken)}
            />
          </div>
          <table class="w-full">
            <tbody class="text-xs tracking-wider">
              {#if bgField !== "None"}
                <tr>
                  <td>
                    <span style={bg(tooltipToken)}>
                      {bgField}
                    </span>
                  </td>
                  <td class="text-right">
                    <span class="pl-1">
                      {tokenDisplayValue(tooltipToken, bgField) ?? "None"}
                    </span>
                  </td>
                </tr>
              {/if}
              {#if underlineField !== "None"}
                <tr>
                  <td>
                    <span class="border-b-2" style={underline(tooltipToken)}>
                      {underlineField}
                    </span>
                  </td>
                  <td class="text-right">
                    <span>
                      {tokenDisplayValue(tooltipToken, underlineField) ?? "None"}
                    </span>
                  </td>
                </tr>
              {/if}
            </tbody>
          </table>
        </div>
        {#if tooltipToken.top_k !== undefined}
          <hr class="bg-gray-400 w-full my-2" />
          <table class="w-full">
            <thead>
              <tr>
                <th
                  class={`px-1 pb-1 font-normal text-xs text-left text-gray-700 tracking-wide`}
                >
                  Candidate
                </th>
                <th
                  class={`px-1 pb-1 font-normal text-xs text-right text-gray-700 tracking-wide`}
                >
                  Prob
                </th>
              </tr>
            </thead>
            <tbody>
              {#each tooltipToken.top_k as candidate, i}
                <tr
                  class={`${i === 5 ? "border-t border-dashed border-gray-300" : ""}`}
                >
                  <td
                    class={`px-1 text-left font-mono text-sm decoration-2 ${candidate.is_masked ? "line-through" : ""}`}
                  >
                    <span class="bg-gray-200">
                      {@html renderText(candidate.text)}
                    </span>
                  </td>
                  <td
                    class={`px-1 text-right font-mono text-sm decoration-2 ${candidate.is_masked ? "line-through" : ""}`}
                  >
                    {candidate.prob?.toFixed(3)}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        {/if}
      </div>
    {:else}
      <div class="text-sm border-b text-red-700">
        Missing tokens will show on completion.
      </div>
    {/if}
  </div>
</div>

<!-- Tokens view -->
<div class="pt-6 pb-6 flex text-gray-800 font-token">
  <div class="px-4">
    <span
      class="flex flex-wrap text-sm"
      role="main"
      use:longhover={mouseLongHoverDuration}
      on:longmouseover={handleLongMouseOver}
      on:longmouseout={handleLongMouseOut}
      on:mouseover={handleMouseOver}
      on:mouseout={handleMouseOut}
      on:focus={doNothing}
      on:blur={doNothing}
    >
      {#if requireFullReplay}
        <TokenGridItem token={continuationToken} index={-1} />
        <div class="basis-full h-2"></div>
      {/if}

      <!-- Render tokens first -->
      {#each tokens as token, i}
          {#if token.role !== ""}
            <!-- Add spacing before role (except for first) -->
            {#if i > 0}
              <div class="basis-full py-3"></div>
            {/if}
            <!-- Force line break for role header -->
            <div class="basis-full h-0"></div>
          {/if}
          <TokenGridItem
            token={token}
            index={i}
            underlineStyle={underline(token)}
            bgStyle={bg(token)}
          />
      {/each}

      {#if isCompleted === false}
        <span
          class="inline-block mt-2 border-b-2 border-white bg-gray-700 animate-cpulse"
        >
          &nbsp;
        </span>
      {/if}
    </span>

    <!-- Render media nodes afterward -->
    {#each multimodalNodes as node}
      {#if node.type === "media"}
        {#if node.data.type == "audio"}
          <div class="my-3">
            <CustomAudio audioData={node.data} />
          </div>
        {/if}
        {#if node.data.type == "video"}
          <div class="my-3">
            <CustomVideo videoData={node.data} />
          </div>
        {/if}
        {#if node.data.type == "image"}
          <div class="my-3">
            <img
              src={`data:${node.data.format};base64,${node.data.value}`}
              alt="Image output"
            />
          </div>
        {/if}
      {/if}
    {/each}
  </div>
</div>
