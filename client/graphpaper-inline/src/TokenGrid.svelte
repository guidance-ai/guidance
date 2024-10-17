<script lang="ts">
    import {isRoleOpenerInput, isTextOutput, type NodeAttr, type RoleOpenerInput, type GenToken} from './stitch';
    import TokenGridItem from "./TokenGridItem.svelte";
    import {type Token, type TokenCallback} from "./interfaces";
    import {type MetricDef} from "./interfaces";
    import {longhover} from "./longhover";
    import DOMPurify from "dompurify";

    import {scaleSequential} from "d3-scale";
    import {interpolateCool, interpolateSpectral, interpolateOrRd, interpolateYlOrRd} from "d3-scale-chromatic";

    const underlineColor = (x: number) => {
        return interpolateOrRd(x);
    };
    const bgColor = (x: number) => {
        return interpolateYlOrRd(1 - x);
    };

    export let textComponents: Array<NodeAttr>;
    export let tokenDetails: Array<GenToken>;
    export let isCompleted: boolean = false;
    let metricDef: MetricDef = {
        name: 'consumed'
    };

    let underline: TokenCallback | undefined;
    let bg: TokenCallback | undefined;

    const colorTokenEmit = (x: Token) => {
        if (x.is_input) {
            return "rgba(255, 255, 255, 0)";
        } else if (x.is_force_forwarded) {
            return "rgba(243, 244, 246, 1)";
        } else if (x.is_generated) {
            return "rgba(229, 231, 235, 1)";
        } else {
            console.log(`ERROR: token ${x.text} does not have emit flags.`)
            return "rgba(255, 255, 255, 0)";
        }
    }

    function findTargetWords(text: string, targetWords: string[]): [number, number, string][] {
        // NOTE(nopdive): Not the most efficient approach, but there aren't many special words anyway.

        const results: [number, number, string][] = [];
        for (const targetWord of targetWords) {
            let start = 0;
            while ((start = text.indexOf(targetWord, start)) !== -1) {
                results.push([start, start + targetWord.length, targetWord]);
                start += targetWord.length;
            }
        }

        results.sort((a, b) => a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]);
        return results;
    }

    let tokens: Array<Token> = [];
    let activeOpenerRoles: Array<RoleOpenerInput> = [];
    let activeCloserRoleText: Array<string> = [];
    let specialSet: Set<string> = new Set<string>();
    let namedRoleSet: Record<string, string> = {};
    let currentTokenIndex: number = 0;
    $: {
        if (textComponents.length === 0) {
            // Reset
            tokens = [];
            activeOpenerRoles = [];
            activeCloserRoleText = [];
            specialSet.clear();
            namedRoleSet = {};
            currentTokenIndex = 0;
        }

        for (; currentTokenIndex < textComponents.length; currentTokenIndex += 1) {
            const nodeAttr = textComponents[currentTokenIndex];

            if (isRoleOpenerInput(nodeAttr)) {
                activeOpenerRoles.push(nodeAttr);
                activeCloserRoleText.push(nodeAttr.closer_text || "");
            } else if (isTextOutput(nodeAttr)) {
                if (activeOpenerRoles.length === 0) {
                    if (activeCloserRoleText.length !== 0 && activeCloserRoleText[activeCloserRoleText.length - 1] === nodeAttr.value) {
                        const token = {
                            text: nodeAttr.value, prob: 1, role: "", special: true,
                            is_input: nodeAttr.is_input, is_force_forwarded: nodeAttr.is_force_forwarded,
                            is_generated: nodeAttr.is_generated,
                        };
                        specialSet.add(token.text);
                        tokens.push(token);
                        activeCloserRoleText.pop();
                    } else {
                        const token = {
                            text: nodeAttr.value, prob: 1, role: "", special: false,
                            is_input: nodeAttr.is_input, is_force_forwarded: nodeAttr.is_force_forwarded,
                            is_generated: nodeAttr.is_generated,
                        };
                        tokens.push(token);
                    }
                } else {
                    const activeOpenerRole = activeOpenerRoles[activeOpenerRoles.length - 1];
                    if (activeOpenerRole.text && activeOpenerRole.text !== nodeAttr.value) {
                        console.log(`Active role text does not match next text output: ${activeOpenerRole.text} - ${nodeAttr.value}`)
                    }

                    const token = {
                        text: nodeAttr.value, prob: 1, role: activeOpenerRole.name || "", special: true,
                        is_input: nodeAttr.is_input, is_force_forwarded: nodeAttr.is_force_forwarded,
                        is_generated: nodeAttr.is_generated,
                    };
                    if (token.role !== "") {
                        namedRoleSet[nodeAttr.value] = token.role;
                    }
                    specialSet.add(token.text);
                    tokens.push(token);
                    activeOpenerRoles.pop();
                }
            }
        }
        // NOTE(nopdive): Often the closer text is missing at the end of output.
        if (activeOpenerRoles.length !== 0 || activeCloserRoleText.length !== 0) {
            // console.log("Opener and closer role texts did not balance.")
        }

        // Process tokens to have detail if we have it (should only happen once at the end).
        const isDetailed = (tokenDetails.length > 0);
        if (isDetailed) {
            // Preprocess for special words
            const fullText = tokenDetails.map((x) => {return x.text}).join("");
            const specialMatchStack = findTargetWords(fullText, Array.from(specialSet));

            tokens = [];

            let tokenStart = 0;
            let tokenEnd = 0;
            let withinRoleMatch = false;
            for (const tokenDetail of tokenDetails) {
                tokenStart = tokenEnd;
                tokenEnd = tokenStart + tokenDetail.text.length;
                let special = false;
                let role = "";

                if (specialMatchStack.length > 0) {
                    // Drop special matches that token has passed
                    let [matchStart, matchEnd, match] = specialMatchStack[0];
                    while (tokenStart >= matchEnd) {
                        let value = specialMatchStack.shift();
                        if (value !== undefined) {
                            [matchStart, matchEnd, match] = value;
                        } else {
                            break;
                        }
                    }

                    // TODO(nopdive): Review, might be off by one.
                    let overlapped = false;
                    if (tokenStart <= matchStart && (tokenEnd-1) >= matchStart) {
                        // Match with token leading
                        overlapped = true;
                    } else if (tokenStart <= (matchEnd-1) && tokenEnd >= matchEnd) {
                        // Match with token trailing
                        overlapped = true;
                    } else if (tokenStart >= matchStart && tokenEnd <= matchEnd) {
                        // Match with token equal or within
                        overlapped = true;
                    }

                    if (overlapped) {
                        if (Object.keys(namedRoleSet).includes(match)) {
                            if (!withinRoleMatch) {
                                role = namedRoleSet[match];
                                withinRoleMatch = true;
                            }
                        }
                        special = true;
                    } else {
                        withinRoleMatch = false;
                    }
                }

                const token = {
                    text: tokenDetail.text,
                    prob: tokenDetail.prob,
                    role: role,
                    special: special,
                    is_input: tokenDetail.is_input,
                    is_force_forwarded: tokenDetail.is_force_forwarded,
                    is_generated: tokenDetail.is_generated,
                    extra: tokenDetail,
                };
                tokens.push(token);
            }
        }

        // Visual updates
        if (metricDef.name === 'avg latency') {
            bg = (x: Token) => bgColor(x.extra?.latency_ms || 0);
            underline = undefined;
        } else {
            underline = (x: Token) => underlineColor(x.prob);
            bg = (x: Token) => colorTokenEmit(x);
        }

        // End bookkeeping (svelte)
        tokenDetails = tokenDetails;
        textComponents = textComponents;
        tokens = tokens;
    }

    let tooltip: HTMLElement;
    let tooltipX = 0;
    let tooltipY = 0;
    let tooltipToken: Token;
    const mouseLongHoverDuration = 200;

    const handleLongMouseOver = (event: CustomEvent<MouseEvent>) => {
        const target = event.detail.target as HTMLElement;
        if (target.matches('.token-grid-item')) {
            const index = target.dataset.index;
            const positionXOffset = 15;
            const positionYOffset = 10;

            // Add tooltip
            const rect = target.getBoundingClientRect();
            tooltipX = (rect.left + window.scrollX + rect.width / 2) + positionXOffset;
            tooltipY = (rect.bottom + window.scrollY) + positionYOffset;
            tooltip.style.display = 'block';
            const indexNum = Number(index);
            tooltipToken = tokens[indexNum];

            // Adjust if near edge of viewport
            if (tooltipX + tooltip.offsetWidth > window.innerWidth) {
                tooltipX = window.innerWidth - tooltip.offsetWidth;
            }
            if (tooltipY + tooltip.offsetHeight > window.innerHeight) {
                tooltipY = window.innerHeight - tooltip.offsetHeight;
            }
        }
    }

    let highlightPrevColor = '';
    let highlightPrevBackgroundColor = '';
    const handleMouseOver = (event: MouseEvent) => {
        const target = event.target as HTMLElement;
        if (target.matches('.token-grid-item')) {
            const index = target.dataset.index;
            const siblingsIncludingSelf = target.parentElement?.querySelectorAll(`.token-grid-item[data-index="${index}"]`);

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
    }

    const handleLongMouseOut = (event: CustomEvent<MouseEvent>) => {
        const target = event.detail.target as HTMLElement;
        if (target.matches('.token-grid-item')) {
            // Remove tooltip
            tooltip.style.display = 'none';
        }
    }

    const handleMouseOut = (event: MouseEvent) => {
        const target = event.target as HTMLElement;
        if (target.matches('.token-grid-item')) {
            const index = target.dataset.index;
            const siblingsIncludingSelf = target.parentElement?.querySelectorAll(`.token-grid-item[data-index="${index}"]`);

            // Remove highlight
            if (siblingsIncludingSelf) {
                for (const sibling of siblingsIncludingSelf) {
                    const htmlSibling = sibling as HTMLElement;
                    htmlSibling.style.color = highlightPrevColor;
                    htmlSibling.style.backgroundColor = highlightPrevBackgroundColor;
                }
            }
        }
    }
    const doNothing = (_: any) => {}
    const renderText = (text: string) => {
        return DOMPurify.sanitize(
            text.replaceAll(' ', '&nbsp;').replaceAll('\t', '\\t').replaceAll('\n', '\\n')
        );
    }
</script>

<!-- Tooltip -->
<div bind:this={tooltip} class="px-1 pt-2 pb-3 absolute opacity-95 bg-white shadow border border-gray-300 pointer-events-none z-50" style="top: {tooltipY}px; left: {tooltipX}px; display: none;">
    <div>
        {#if tooltipToken}
            <div class={`col-1 flex flex-col items-center`}>
                <div class="text-2xl px-1 pt-1 pb-1 text-left w-full bg-white">
                    <div class="mb-4">
                        <span class="border-b-2 border-red-700">
                        {@html renderText(tooltipToken.text)}
                        </span>
                    </div>
                    <table class="w-full">
                        <tbody class="text-xs tracking-wider">
                            <tr>
                                <td>
                                    <span class="bg-gray-200">
                                        Type
                                    </span>
                                </td>
                                <td class="text-right">
                                    <span class="pl-1">
                                        {#if tooltipToken.is_generated}
                                            Generated
                                        {:else if tooltipToken.is_input}
                                            Input
                                        {:else if tooltipToken.is_force_forwarded}
                                            Forced
                                        {:else}
                                            Token
                                        {/if}
                                    </span>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <span class="border-b-2 border-red-700">
                                        Probability
                                    </span>
                                </td>
                                <td class="text-right">
                                    <span>
                                        {tooltipToken.prob.toFixed(3)}
                                    </span>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                {#if tooltipToken.extra !== undefined}
                <hr class="bg-gray-400 w-full my-2"/>
                <table class="w-full">
                    <thead>
                        <tr>
                            <th class={`px-1 pb-1 font-normal text-xs text-left text-gray-700 tracking-wide`}>
                                Candidate
                            </th>
                            <th class={`px-1 pb-1 font-normal text-xs text-right text-gray-700 tracking-wide`}>
                                Prob
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                    {#each tooltipToken.extra.top_k as candidate}
                        <tr>
                            <td class={`px-1 text-left font-mono text-sm decoration-2 ${candidate.is_masked ? "line-through" : ""}`}>
                                <span class="bg-gray-200">
                                    {@html renderText(candidate.text)}
                                </span>
                            </td>
                            <td class={`px-1 text-right font-mono text-sm decoration-2 ${candidate.is_masked ? "line-through" : ""}`}>
                                {candidate.prob.toFixed(3)}
                            </td>
                        </tr>
                    {/each}
                    </tbody>
                </table>
                {/if}
            </div>
        {/if}
    </div>
</div>

<!-- Tokens view -->
<div class="pt-6 pb-6 flex text-gray-800 font-token">
    <div class="px-4">
        <span class="flex flex-wrap text-sm" role="main" use:longhover={mouseLongHoverDuration} on:longmouseover={handleLongMouseOver} on:longmouseout={handleLongMouseOut} on:mouseover={handleMouseOver} on:mouseout={handleMouseOut} on:focus={doNothing} on:blur={doNothing}>
            {#each tokens as token, i}
                {#if token.special === true && token.role !== ""}
                    <!-- Vertical spacing for role -->
                    {#if i === 0}
                        <div class="basis-full h-2"></div>
                    {:else}
                        {#each {length: 2} as _}
                            <div class="basis-full h-0"></div>
                            <span class="inline-block">&nbsp;</span>
                        {/each}
                        <div class="basis-full h-0"></div>
                    {/if}
                {/if}

                <TokenGridItem token={token} index={i} underline={underline} bg={bg}/>
            {/each}

            {#if isCompleted === false}
                <span class="inline-block mt-2 border-b-2 border-white bg-gray-700 animate-cpulse">
                    &nbsp;
                </span>
            {/if}
        </span>
    </div>
</div>