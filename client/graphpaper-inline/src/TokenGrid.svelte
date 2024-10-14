<!-- TODO(nopdive): Process tokens incrementally. -->

<script lang="ts">
    import {isRoleOpenerInput, isTextOutput, type NodeAttr, type RoleOpenerInput, type GenToken} from './stitch';
    import TokenGridItem, {type Token} from "./TokenGridItem.svelte";
    import {longhover} from "./longhover";
    import DOMPurify from "dompurify";

    export let textComponents: Array<NodeAttr>;
    export let tokenDetails: Array<GenToken>;
    export let isCompleted: boolean = false;

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
    $: {
        let activeOpenerRoles: Array<RoleOpenerInput> = [];
        let activeCloserRoleText: Array<string> = [];

        let specialSet: Set<string> = new Set<string>();
        let namedRoleSet: Record<string, string> = {};

        tokens = [];
        for (let nodeAttr of textComponents) {
            if (isRoleOpenerInput(nodeAttr)) {
                activeOpenerRoles.push(nodeAttr);
                activeCloserRoleText.push(nodeAttr.closer_text || "");
            } else if (isTextOutput(nodeAttr)) {
                if (activeOpenerRoles.length === 0) {
                    if (activeCloserRoleText.length !== 0 && activeCloserRoleText[activeCloserRoleText.length - 1] === nodeAttr.value) {
                        const token = {text: nodeAttr.value, prob: 1, role: "", special: true};
                        specialSet.add(token.text);
                        tokens.push(token);
                        activeCloserRoleText.pop();
                    } else {
                        const token = {text: nodeAttr.value, prob: 1, role: "", special: false};
                        tokens.push(token);
                    }
                } else {
                    const activeOpenerRole = activeOpenerRoles[activeOpenerRoles.length - 1];
                    if (activeOpenerRole.text && activeOpenerRole.text !== nodeAttr.value) {
                        console.log(`Active role text does not match next text output: ${activeOpenerRole.text} - ${nodeAttr.value}`)
                    }

                    const token = {text: nodeAttr.value, prob: 1, role: activeOpenerRole.name || "", special: true};
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

        // Process tokens to have detail if we have it
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

                // const role = Object.keys(namedRoleSet).includes(tokenDetail.text) ? namedRoleSet[tokenDetail.text] : "";
                // const special = specialSet.has(tokenDetail.text);

                const token = {
                    text: tokenDetail.text,
                    prob: tokenDetail.prob,
                    role: role,
                    special: special,
                    extra: tokenDetail
                }
                tokens.push(token);
            }
        }

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

    const handleMouseOver = (event: MouseEvent) => {
        const target = event.target as HTMLElement;
        if (target.matches('.token-grid-item')) {
            const index = target.dataset.index;
            const siblingsIncludingSelf = target.parentElement?.querySelectorAll(`.token-grid-item[data-index="${index}"]`);

            // Add highlight
            if (siblingsIncludingSelf) {
                for (const sibling of siblingsIncludingSelf) {
                    sibling.classList.add('text-gray-50');
                    sibling.classList.add('bg-gray-400');
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
                    sibling.classList.remove('text-gray-50');
                    sibling.classList.remove('bg-gray-400');
                }
            }
        }
    }
    const doNothing = (_: any) => {}
    const escapeWhitespaceCharacters = (text: string) => {
        return text.replaceAll(' ', '&nbsp;').replaceAll('\t', '\\t').replaceAll('\n', '\\n');
    }
</script>

<!-- Tooltip -->
<div bind:this={tooltip} class="px-1 pt-2 pb-3 absolute opacity-95 bg-gray-100 border-l-4 border-l-red-500 border-b-2 border-b-gray-300 text-gray-700 pointer-events-none z-50" style="top: {tooltipY}px; left: {tooltipX}px; display: none;">
    <div>
        {#if tooltipToken}
            <div class={`col-1 flex flex-col items-center`}>
                <div class="text-lg px-1 pb-2 text-left w-full">
                    <div class="uppercase text-xs text-gray-500 tracking-wide">
                        Token
                    </div>
                    <div class="bg-gray-200">
                        {@html DOMPurify.sanitize(escapeWhitespaceCharacters(tooltipToken.text))}
                    </div>
                </div>
                {#if tooltipToken.extra !== undefined}
                <table class="divide-gray-200">
                    <thead>
                        <tr>
                            <th class={`px-1 pb-1 uppercase font-normal text-xs text-left text-gray-500 tracking-wide`}>
                            </th>
                            <th class={`px-1 pb-1 uppercase font-normal text-xs text-right text-gray-500 tracking-wide`}>
                                Prob
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                    {#each tooltipToken.extra.top_k as candidate}
                        <tr>
                            <!-- TODO(nopdive): Text strike through is ugly, replace with line drawn through whole row via css pseudo-element -->
                            <td class={`px-1 font-mono text-sm decoration-2 ${candidate.is_masked ? "line-through" : ""}`}><span class="bg-gray-200">{@html DOMPurify.sanitize(escapeWhitespaceCharacters(candidate.text))}</span></td>
                            <td class={`px-1 font-mono text-sm decoration-2 ${candidate.is_masked ? "line-through" : ""}`}>{candidate.prob.toFixed(3)}</td>
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

                <TokenGridItem token={token} index={i} />

            {/each}

            {#if isCompleted === false}
                <span class="inline-block mt-2 border-b-2 border-white bg-gray-700 animate-cpulse">
                    &nbsp;
                </span>
            {/if}
        </span>
    </div>
</div>