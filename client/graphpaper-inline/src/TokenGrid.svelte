<!-- TODO(nopdive): Tooltip fed by token info -->

<script lang="ts">
    import {isRoleOpenerInput, isTextOutput, type NodeAttr, type RoleOpenerInput} from './stitch';
    import TokenGridItem from "./TokenGridItem.svelte";
    import {longhover} from "./longhover";
    import DOMPurify from "dompurify";

    interface Token {
        value: string,
        prob: number,
        role: string,
        special: boolean,
    }

    export let nodeAttrs: Array<NodeAttr>;
    export let isCompleted: boolean = false;
    let tokens: Array<Token> = [];
    $: {
        let activeOpenerRoles: Array<RoleOpenerInput> = [];
        let activeCloserRoleText: Array<string> = [];

        tokens = [];
        for (let nodeAttr of nodeAttrs) {
            if (isRoleOpenerInput(nodeAttr)) {
                activeOpenerRoles.push(nodeAttr);
                activeCloserRoleText.push(nodeAttr.closer_text || "");
            } else if (isTextOutput(nodeAttr)) {
                if (activeOpenerRoles.length === 0) {
                    if (activeCloserRoleText.length !== 0 && activeCloserRoleText[activeCloserRoleText.length - 1] === nodeAttr.value) {
                        const token = {value: nodeAttr.value, prob: 1, role: "", special: true};
                        tokens.push(token);
                        activeCloserRoleText.pop();
                    } else {
                        const token = {value: nodeAttr.value, prob: 1, role: "", special: false};
                        tokens.push(token);
                    }
                } else {
                    const activeOpenerRole = activeOpenerRoles[activeOpenerRoles.length - 1];
                    if (activeOpenerRole.text && activeOpenerRole.text !== nodeAttr.value) {
                        console.log(`Active role text does not match next text output: ${activeOpenerRole.text} - ${nodeAttr.value}`)
                    }
                    const token = {value: nodeAttr.value, prob: 1, role: activeOpenerRole.name || "", special: true};
                    tokens.push(token);
                    activeOpenerRoles.pop();
                }
            }

            nodeAttrs = nodeAttrs;
            tokens = tokens;
        }

        // NOTE(nopdive): Often the closer text is missing at the end of output.
        if (activeOpenerRoles.length !== 0 || activeCloserRoleText.length !== 0) {
            // console.log("Opener and closer role texts did not balance.")
        }
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
                <div class="text-lg px-1 pb-3 text-left w-full">
                    <div class="uppercase text-xs text-gray-500 tracking-wide">
                        Token
                    </div>
                    <div class="bg-gray-200">
                        {@html DOMPurify.sanitize(escapeWhitespaceCharacters(tooltipToken.value))}
                    </div>
                </div>
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
                        <tr class="line-through">
                            <td class="px-1 font-mono text-sm"><span class="bg-gray-200">degenerates</span></td>
                            <td class="px-1 font-mono text-sm">0.983</td>
                        </tr>
                        <tr>
                            <td class="px-1 font-mono text-sm"><span class="bg-gray-200">jugs</span></td>
                            <td class="px-1 font-mono text-sm">0.201</td>
                        </tr>
                        <tr>
                            <td class="px-1 font-mono text-sm"><span class="bg-gray-200">madness</span></td>
                            <td class="px-1 font-mono text-sm">0.005</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        {/if}
    </div>
</div>

<div class="pt-6 pb-6 flex text-gray-800 font-token">
    <!-- Tokens view -->
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