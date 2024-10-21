// Action for long mouse hovers.

export function longhover(node: HTMLElement, duration: number) {
    let timer: any;

    const handleMouseOver = (event: MouseEvent) => {
        timer = setTimeout(() => {
            node.dispatchEvent(new CustomEvent('longmouseover', {detail: event}));
        }, duration);
    };
    const handleMouseOut = (event: MouseEvent) => {
        clearTimeout(timer);
        node.dispatchEvent(new CustomEvent('longmouseout', {detail: event}));
    }

    node.addEventListener('mouseover', handleMouseOver);
    node.addEventListener('mouseout', handleMouseOut);

    return {
        update(newDuration: number) {
            duration = newDuration
        },
        destroy() {
            node.removeEventListener('mouseover', handleMouseOver);
            node.removeEventListener('mouseout', handleMouseOut);
        }
    };
}
