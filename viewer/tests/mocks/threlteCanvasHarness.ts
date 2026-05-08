let lastRenderer: unknown = null;

export function setLastRenderer(renderer: unknown) {
	lastRenderer = renderer;
}

export function getLastRenderer() {
	return lastRenderer;
}

export function resetLastRenderer() {
	lastRenderer = null;
}
