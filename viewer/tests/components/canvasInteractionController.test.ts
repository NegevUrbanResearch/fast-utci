import { describe, expect, it, vi } from 'vitest';
import { createCanvasInteractionController } from '$lib/components/viewer/canvasInteractionController';

describe('createCanvasInteractionController', () => {
	it('attaches and detaches canvas and window interaction listeners', () => {
		const canvas = document.createElement('canvas');
		const createPointerEvent = (type: string) =>
			typeof PointerEvent === 'undefined' ? new Event(type) : new PointerEvent(type);
		const onPointerMove = vi.fn();
		const onPointerLeave = vi.fn();
		const onWheel = vi.fn();
		const onPointerDown = vi.fn();
		const onWindowPointerUp = vi.fn();

		const controller = createCanvasInteractionController({
			canvas,
			windowTarget: window,
			onPointerMove,
			onPointerLeave,
			onWheel,
			onPointerDown,
			onWindowPointerUp,
			onWindowPointerCancel: onWindowPointerUp
		});

		canvas.dispatchEvent(new MouseEvent('mousemove'));
		canvas.dispatchEvent(new MouseEvent('mouseleave'));
		canvas.dispatchEvent(new WheelEvent('wheel'));
		canvas.dispatchEvent(createPointerEvent('pointerdown'));
		window.dispatchEvent(createPointerEvent('pointerup'));
		window.dispatchEvent(createPointerEvent('pointercancel'));

		expect(onPointerMove).toHaveBeenCalledTimes(1);
		expect(onPointerLeave).toHaveBeenCalledTimes(1);
		expect(onWheel).toHaveBeenCalledTimes(1);
		expect(onPointerDown).toHaveBeenCalledTimes(1);
		expect(onWindowPointerUp).toHaveBeenCalledTimes(2);

		controller.dispose();
		canvas.dispatchEvent(new MouseEvent('mousemove'));
		window.dispatchEvent(createPointerEvent('pointerup'));

		expect(onPointerMove).toHaveBeenCalledTimes(1);
		expect(onWindowPointerUp).toHaveBeenCalledTimes(2);
	});
});
