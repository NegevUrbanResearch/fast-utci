export interface CanvasInteractionController {
	dispose(): void;
}

export interface CanvasInteractionControllerParams {
	canvas: HTMLCanvasElement;
	windowTarget?: Window;
	onPointerMove?: (event: MouseEvent) => void;
	onPointerLeave?: (event: MouseEvent) => void;
	onWheel?: (event: WheelEvent) => void;
	onPointerDown?: (event: PointerEvent) => void;
	onClick?: (event: MouseEvent) => void;
	onWindowPointerUp?: (event: PointerEvent) => void;
	onWindowPointerCancel?: (event: PointerEvent) => void;
}

export function createCanvasInteractionController(
	params: CanvasInteractionControllerParams
): CanvasInteractionController {
	const windowTarget = params.windowTarget;
	if (params.onPointerMove) {
		params.canvas.addEventListener('mousemove', params.onPointerMove, { passive: true });
	}
	if (params.onPointerLeave) {
		params.canvas.addEventListener('mouseleave', params.onPointerLeave, { passive: true });
	}
	if (params.onWheel) params.canvas.addEventListener('wheel', params.onWheel, { passive: true });
	if (params.onPointerDown) {
		params.canvas.addEventListener('pointerdown', params.onPointerDown, { passive: true });
	}
	if (params.onClick) params.canvas.addEventListener('click', params.onClick);
	if (windowTarget && params.onWindowPointerUp) {
		windowTarget.addEventListener('pointerup', params.onWindowPointerUp, { passive: true });
	}
	if (windowTarget && params.onWindowPointerCancel) {
		windowTarget.addEventListener('pointercancel', params.onWindowPointerCancel, { passive: true });
	}

	return {
		dispose() {
			if (params.onPointerMove) {
				params.canvas.removeEventListener('mousemove', params.onPointerMove);
			}
			if (params.onPointerLeave) {
				params.canvas.removeEventListener('mouseleave', params.onPointerLeave);
			}
			if (params.onWheel) params.canvas.removeEventListener('wheel', params.onWheel);
			if (params.onPointerDown) {
				params.canvas.removeEventListener('pointerdown', params.onPointerDown);
			}
			if (params.onClick) params.canvas.removeEventListener('click', params.onClick);
			if (windowTarget && params.onWindowPointerUp) {
				windowTarget.removeEventListener('pointerup', params.onWindowPointerUp);
			}
			if (windowTarget && params.onWindowPointerCancel) {
				windowTarget.removeEventListener('pointercancel', params.onWindowPointerCancel);
			}
		}
	};
}
