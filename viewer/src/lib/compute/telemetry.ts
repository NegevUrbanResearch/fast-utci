export interface ComputeTelemetryEvent {
	stage: string;
	ts: number;
	ms?: number;
	data?: Record<string, number | string | boolean | null | undefined>;
}

type Listener = (event: ComputeTelemetryEvent) => void;

const listeners = new Set<Listener>();
const history: ComputeTelemetryEvent[] = [];
const MAX_HISTORY = 300;

export function emitComputeTelemetry(
	stage: string,
	params: { ms?: number; data?: ComputeTelemetryEvent['data'] } = {}
): void {
	const event: ComputeTelemetryEvent = {
		stage,
		ts: Date.now(),
		...(params.ms !== undefined ? { ms: params.ms } : {}),
		...(params.data ? { data: params.data } : {})
	};

	history.push(event);
	if (history.length > MAX_HISTORY) history.shift();

	for (const listener of listeners) listener(event);
}

export function subscribeComputeTelemetry(listener: Listener): () => void {
	listeners.add(listener);
	return () => listeners.delete(listener);
}

export function getComputeTelemetryHistory(): ComputeTelemetryEvent[] {
	return [...history];
}

export function clearComputeTelemetryHistory(): void {
	history.length = 0;
}

