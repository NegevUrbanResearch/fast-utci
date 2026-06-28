<script lang="ts">
	import { setLastRenderer } from './threlteCanvasHarness';

	export let createRenderer:
		| ((canvas: HTMLCanvasElement) => unknown)
		| undefined = undefined;

	let canvas: HTMLCanvasElement;

	$: if (canvas && createRenderer) {
		setLastRenderer(createRenderer(canvas));
	}
</script>

<canvas bind:this={canvas} data-testid="mock-canvas">
	<slot />
</canvas>
