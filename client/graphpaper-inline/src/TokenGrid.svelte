<script lang="ts">
    import {isRoleOpenerInput, isTextOutput, type NodeAttr, type RoleOpenerInput} from './stitch';
    import TokenGridItem from "./TokenGridItem.svelte";

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
        }
        if (activeOpenerRoles.length !== 0 || activeCloserRoleText.length !== 0) {
            console.log("Opener and closer role texts did not balance.")
        }
        tokens = tokens;
    }

</script>

<div class="pt-6 pb-6 flex text-gray-800 font-token">
    <!-- Tokens view -->
    <div class="px-4">
        <span class="flex flex-wrap text-sm">
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
                <TokenGridItem token={token} />
            {/each}

            {#if isCompleted === false}
                <span class="inline-block mt-2 border-b-2 border-white bg-gray-700 animate-cpulse">
                    &nbsp;
                </span>
            {/if}
        </span>
    </div>
</div>