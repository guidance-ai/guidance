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
        for (let nodeAttr of nodeAttrs) {
            console.log(nodeAttr);
            if (isRoleOpenerInput(nodeAttr)) {
                activeOpenerRoles.push(nodeAttr);
            } else if (isTextOutput(nodeAttr)) {
                if (activeOpenerRoles.length === 0) {
                    const token = {value: nodeAttr.value, prob: 1, role: "", special: false};
                    tokens.push(token);
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
    }
</script>

<div class="pt-6 pb-6 flex text-gray-800 font-token">
    <!-- Tokens view -->
    <div class="px-4">
        <span class="flex flex-wrap text-sm">
            {#each tokens as token, i}
                {#if token.special === true}
                    {#if token.role !== ""}
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

                        <!-- Token with role annotation -->
                        <span class="inline-block relative">
                            <span class="absolute bottom-7 text-xs mt-2 uppercase -mb-1 text-purple-800 font-sans">
                                {token.role}
                            </span>
                            <TokenGridItem token={token} />
                        </span>
                    {:else}
                        <!-- Token without role annotation -->
                        <div class="basis-full h-0"></div>
                        <span class="inline-block relative">
                            <TokenGridItem token={token} />
                        </span>
                    {/if}
                {:else if token.special === false}
                    <!-- Regular token -->
                    <TokenGridItem token={token} />
                {/if}
            {/each}
            {#if isCompleted === false}
                <span class="inline-block mt-2 border-b-2 border-white bg-gray-700 animate-cpulse">
                    &nbsp;
                </span>
            {/if}
        </span>
    </div>
</div>