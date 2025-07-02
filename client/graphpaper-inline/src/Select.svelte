<!-- Custom select dropdown -->
<script lang="ts">
    import { clickOutside } from "./clickoutside";
    import { createEventDispatcher } from 'svelte';

    export let classes: string = "";
    export let values: Array<string> = [];
    export let defaultValue: string = "";

    let selected = defaultValue;
    let showList = false;

    const dispatch = createEventDispatcher();

    const onDropdownClick = (_: MouseEvent) => {
        showList = !showList;
    };
    const onOutClick = (_: MouseEvent) => {
        showList = false;
    };
    const selectOption = (option: string) => {
        selected = option;
        showList = false;
        dispatch('select', selected);
    }
</script>

<div class="relative" use:clickOutside on:outclick={onOutClick}>
    <button use:clickOutside on:click={onDropdownClick}>
        <span class={`mr-4 inline-flex justify-between items-center hover:bg-gray-700 hover:text-white dark:hover:bg-gray-300 dark:hover:text-gray-900 ${classes}`}>
            <span class="">
                {selected}
            </span>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" class="size-4">
                <path fill-rule="evenodd" d="M4.22 6.22a.75.75 0 0 1 1.06 0L8 8.94l2.72-2.72a.75.75 0 1 1 1.06 1.06l-3.25 3.25a.75.75 0 0 1-1.06 0L4.22 7.28a.75.75 0 0 1 0-1.06Z" clip-rule="evenodd" />
            </svg>
        </span>
    </button>
    {#if showList}
        <ul role="listbox" class="absolute pt-2 pb-3 z-10 mt-1 bg-white dark:bg-gray-800 shadow border border-gray-300 dark:border-gray-600">
            {#each values as value, i}
                <li class={`w-full px-4 py-1 ${i === 0 ? "mt-1" : ""} hover:bg-gray-700 hover:text-white dark:hover:bg-gray-300 dark:hover:text-gray-900 text-nowrap`} role="option" aria-selected="false" on:click={(_) => selectOption(value)} on:keypress={(_) => {}}>{value}</li>
            {/each}
        </ul>
    {/if}
</div>
