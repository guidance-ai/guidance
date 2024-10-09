<!-- TODO(nopdive): Tooltip fed by token info -->

<script lang="ts">
    import {isRoleOpenerInput, isTextOutput, type NodeAttr, type RoleOpenerInput} from './stitch';
    import TokenGridItem from "./TokenGridItem.svelte";
    import {longhover} from "./longhover";

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
    let tooltipText = '';
    const mouseLongHoverDuration = 200;

    const handleLongMouseOver = (event: CustomEvent<MouseEvent>) => {
        const target = event.detail.target as HTMLElement;
        if (target.matches('.token-grid-item')) {
            const index = target.dataset.index;
            const positionXOffset = 15;
            const positionYOffset = 10;

            // Add tooltip
            tooltipText = index || "";
            const rect = target.getBoundingClientRect();
            tooltipX = (rect.left + window.scrollX + rect.width / 2) + positionXOffset;
            tooltipY = (rect.bottom + window.scrollY) + positionYOffset;
            tooltip.style.display = 'block';

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

</script>

<!-- Tooltip -->
<div bind:this={tooltip} class="absolute opacity-70 bg-black text-gray-50 pointer-events-none z-10" style="top: {tooltipY}px; left: {tooltipX}px">
    {tooltipText}
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