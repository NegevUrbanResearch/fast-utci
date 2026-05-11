export interface DebugLegacyAcceptedOutput<TPayload = unknown> {
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	output: {
		gpuBuffer?: { destroy?: () => void };
		gpuOutputHandle?: { dispose?: () => void };
	};
	payload: TPayload;
}

export interface DebugLegacyDeferredFallback<TPayload = unknown> {
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	payload: TPayload;
}

export interface DebugSelectedHourLegacyCounters {
	legacySelectedHourDispatchCount: number;
	legacyScrubScheduleCount: number;
}

export interface DeferredFallbackKey {
	requestId: number;
	monthIndex: number;
	timeIndex: number;
}

export interface DebugSelectedHourLegacyHost<
	TAcceptedPayload = unknown,
	TFallbackPayload = unknown
> {
	getAcceptedOutput(): DebugLegacyAcceptedOutput<TAcceptedPayload> | null;
	setAcceptedOutput(next: DebugLegacyAcceptedOutput<TAcceptedPayload> | null): void;
	clearAcceptedOutput(): void;
	releaseAcceptedOutput(key: DeferredFallbackKey): void;
	setDeferredCpuFallback(next: DebugLegacyDeferredFallback<TFallbackPayload> | null): void;
	takeDeferredCpuFallback(
		key: DeferredFallbackKey
	): DebugLegacyDeferredFallback<TFallbackPayload> | null;
	recordDispatch(): number;
	recordScrubSchedule(): number;
	invalidateScrubSchedule(): number;
	getScrubScheduleRunId(): number;
	getCounters(): DebugSelectedHourLegacyCounters;
	resetCounters(): void;
	dispose(): void;
}

type ManagedAcceptedOutput<TPayload> = {
	value: DebugLegacyAcceptedOutput<TPayload>;
	releasable: boolean;
};

function getAcceptedOutputKey(
	output: Pick<DebugLegacyAcceptedOutput<unknown>, 'requestId' | 'monthIndex' | 'timeIndex'>
): string {
	return `${output.requestId}:${output.monthIndex}:${output.timeIndex}`;
}

function destroyAcceptedOutput(output: DebugLegacyAcceptedOutput<unknown> | null | undefined): void {
	if (!output) return;
	const gpuOutputHandle = output.output.gpuOutputHandle;
	if (gpuOutputHandle) {
		gpuOutputHandle.dispose?.();
	} else {
		output.output.gpuBuffer?.destroy?.();
	}
	output.output.gpuBuffer = undefined;
	output.output.gpuOutputHandle = undefined;
}

function destroyManagedAcceptedOutput<TPayload>(entry: ManagedAcceptedOutput<TPayload> | null): void {
	destroyAcceptedOutput(entry?.value as DebugLegacyAcceptedOutput<unknown> | null | undefined);
}

function maybeDestroyManagedAcceptedOutput<TPayload>(
	entry: ManagedAcceptedOutput<TPayload>
): boolean {
	if (!entry.releasable) return false;
	destroyManagedAcceptedOutput(entry);
	return true;
}

export function createDebugSelectedHourLegacyHost<
	TAcceptedPayload = unknown,
	TFallbackPayload = unknown
>(): DebugSelectedHourLegacyHost<TAcceptedPayload, TFallbackPayload> {
	let acceptedOutput: ManagedAcceptedOutput<TAcceptedPayload> | null = null;
	const retiredAcceptedOutputs = new Map<string, ManagedAcceptedOutput<TAcceptedPayload>>();
	let deferredCpuFallback: DebugLegacyDeferredFallback<TFallbackPayload> | null = null;
	let scrubScheduleRunId = 0;
	let counters: DebugSelectedHourLegacyCounters = {
		legacySelectedHourDispatchCount: 0,
		legacyScrubScheduleCount: 0
	};

	function retireAcceptedOutput(entry: ManagedAcceptedOutput<TAcceptedPayload> | null): void {
		if (!entry) return;
		if (maybeDestroyManagedAcceptedOutput(entry)) {
			return;
		}
		retiredAcceptedOutputs.set(getAcceptedOutputKey(entry.value), entry);
	}

	return {
		getAcceptedOutput() {
			return acceptedOutput?.value ?? null;
		},
		setAcceptedOutput(next) {
			if (acceptedOutput && acceptedOutput.value.output !== next?.output) {
				retireAcceptedOutput(acceptedOutput);
			}
			if (!next) {
				acceptedOutput = null;
				return;
			}
			if (acceptedOutput?.value.output === next.output) {
				acceptedOutput = {
					value: next,
					releasable: acceptedOutput.releasable
				};
				return;
			}
			acceptedOutput = {
				value: next,
				releasable: false
			};
		},
		clearAcceptedOutput() {
			if (!acceptedOutput) return;
			retireAcceptedOutput(acceptedOutput);
			acceptedOutput = null;
		},
		releaseAcceptedOutput(key) {
			const currentKey = acceptedOutput ? getAcceptedOutputKey(acceptedOutput.value) : null;
			const releasedKey = getAcceptedOutputKey(key);
			if (currentKey === releasedKey && acceptedOutput) {
				acceptedOutput = {
					...acceptedOutput,
					releasable: true
				};
				return;
			}
			const retired = retiredAcceptedOutputs.get(releasedKey);
			if (!retired) return;
			retiredAcceptedOutputs.delete(releasedKey);
			destroyManagedAcceptedOutput(retired);
		},
		setDeferredCpuFallback(next) {
			deferredCpuFallback = next;
		},
		takeDeferredCpuFallback(key) {
			if (
				!deferredCpuFallback ||
				deferredCpuFallback.requestId !== key.requestId ||
				deferredCpuFallback.monthIndex !== key.monthIndex ||
				deferredCpuFallback.timeIndex !== key.timeIndex
			) {
				return null;
			}
			const matchedFallback = deferredCpuFallback;
			deferredCpuFallback = null;
			return matchedFallback;
		},
		recordDispatch() {
			counters.legacySelectedHourDispatchCount += 1;
			return counters.legacySelectedHourDispatchCount;
		},
		recordScrubSchedule() {
			counters.legacyScrubScheduleCount += 1;
			scrubScheduleRunId += 1;
			return scrubScheduleRunId;
		},
		invalidateScrubSchedule() {
			scrubScheduleRunId += 1;
			return scrubScheduleRunId;
		},
		getScrubScheduleRunId() {
			return scrubScheduleRunId;
		},
		getCounters() {
			return { ...counters };
		},
		resetCounters() {
			// Keep the scrub invalidation run id monotonic; resetCounters only clears diagnostics counters.
			counters = {
				legacySelectedHourDispatchCount: 0,
				legacyScrubScheduleCount: 0
			};
		},
		dispose() {
			deferredCpuFallback = null;
			for (const retired of retiredAcceptedOutputs.values()) {
				destroyManagedAcceptedOutput(retired);
			}
			retiredAcceptedOutputs.clear();
			destroyManagedAcceptedOutput(acceptedOutput);
			acceptedOutput = null;
		}
	};
}
