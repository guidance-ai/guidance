// Action for clicking outside an element.

export function clickOutside(node: HTMLElement) {
	const handleClick = (event: MouseEvent) => {
		let target = event.target as HTMLElement;
		if (!node.contains(target)) {
			node.dispatchEvent(new CustomEvent('outclick'));
		}
	};

	document.addEventListener('click', handleClick, true);

	return {
		destroy() {
			document.removeEventListener('click', handleClick, true);
		}
	};
}