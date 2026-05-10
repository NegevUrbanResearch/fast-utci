import type { Analysis } from '$lib/types/analysis';
import {
	disposeSelectedHourGpuResidentOutput,
	prepareSelectedHourLiveSession,
	type SelectedHourCpuFallbackOutput,
	type SelectedHourGpuResidentOutput,
	type SelectedHourLiveSession
} from '$lib/compute/liveUtciSelectedHourSession';
import type { SelectedHourRenderTimingSubsteps } from '$lib/compute/onDemandDiagnostics';
import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/liveSelectedHourSurfaceIdentity';

export type LiveSelectedHourRenderTransport =
	| 'idle'
	| 'cpu-uploaded-selected-hour'
	| 'compute-buffer-selected-hour';

export type LiveSelectedHourControllerSurfaceDiagnostics = {
	utciSurfaceSource?: string;
	selectedHourTransferCount?: number;
	dataTextureBuildCount?: number;
	cpuPublishRequestId?: number;
	cpuPublishMonthIndex?: number;
	cpuPublishHourIndex?: number;
	cpuPublishTimeIndex?: number;
	cpuPublishSelectionKey?: string;
	gpuResidentCopyStatus?: 'idle' | 'pending' | 'complete' | 'failed';
	gpuResidentCopyError?: string;
	gpuResidentCopyRequestId?: number;
} & SelectedHourRenderTimingSubsteps;

export type LiveSelectedHourControllerState = {
	analysis: Analysis | null;
	acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null;
	surfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
	loading: boolean;
	error: string | null;
	renderTransport: LiveSelectedHourRenderTransport;
	sameDeviceForComputeAndRender: boolean | null;
	pendingRenderUpdateStartedAt: number | undefined;
	renderSurfaceDiagnostics: LiveSelectedHourControllerSurfaceDiagnostics;
	ready: boolean;
	renderReady: boolean;
	awaitingGpuSurface: boolean;
};

type LiveSelectedHourControllerMutableState = Omit<
	LiveSelectedHourControllerState,
	'ready' | 'renderReady' | 'awaitingGpuSurface'
>;

export type LiveSelectedHourSessionConfig = Omit<
	Parameters<typeof prepareSelectedHourLiveSession>[0],
	'signal'
>;

export type LiveSelectedHourControllerRequest = {
	sessionKey: string;
	sessionConfig: LiveSelectedHourSessionConfig;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey?: string;
	colorMode: 'normalized' | 'discrete';
	preferGpuResident: boolean;
	rendererDevice?: GPUDevice;
};

export type LiveSelectedHourControllerRequestResult = {
	accepted: boolean;
	reason?: 'stale' | 'disposed';
	state: LiveSelectedHourControllerState;
};

export type LiveSelectedHourController = {
	getState(): LiveSelectedHourControllerState;
	subscribe(
		listener: (state: LiveSelectedHourControllerState) => void
	): () => void;
	requestSelection(
		request: LiveSelectedHourControllerRequest
	): Promise<LiveSelectedHourControllerRequestResult>;
	handleRenderSurfaceDiagnostics(
		diagnostics: LiveSelectedHourControllerSurfaceDiagnostics
	): Promise<void>;
	dispose(): void;
};

type DeferredCpuFallbackState = {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	analysis: Analysis | null;
	loadCpuFallback?: () => Promise<SelectedHourCpuFallbackOutput>;
};

type AcceptedCpuPublication = {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey: string;
};

type CreateLiveSelectedHourControllerOptions = {
	prepareSession?: (
		params: Parameters<typeof prepareSelectedHourLiveSession>[0]
	) => Promise<SelectedHourLiveSession>;
};

const EMPTY_SURFACE_DIAGNOSTICS: LiveSelectedHourControllerSurfaceDiagnostics = {};

function cloneState(state: LiveSelectedHourControllerState): LiveSelectedHourControllerState {
	return {
		...state,
		surfaceIdentity: state.surfaceIdentity ? { ...state.surfaceIdentity } : null,
		renderSurfaceDiagnostics: { ...state.renderSurfaceDiagnostics }
	};
}

function createSelectionKey(params: {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
}): string {
	return `${params.requestId}:${params.monthIndex}:${params.hourIndex}:${params.timeIndex}`;
}

function createSurfaceIdentity(params: {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey: string;
	pendingRenderUpdateStartedAt: number | undefined;
	acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null;
}): LiveSelectedHourSurfaceIdentity {
	return {
		requestId: params.requestId,
		monthIndex: params.monthIndex,
		hourIndex: params.hourIndex,
		timeIndex: params.timeIndex,
		selectionKey: params.selectionKey,
		pendingRenderUpdateStartedAt: params.pendingRenderUpdateStartedAt,
		acceptedGpuResidentOutput: params.acceptedGpuResidentOutput
	};
}

function hasAcceptedCpuRenderSurface(params: {
	renderTransport: LiveSelectedHourRenderTransport;
	renderSurfaceDiagnostics: LiveSelectedHourControllerSurfaceDiagnostics;
	acceptedCpuPublication: AcceptedCpuPublication | null;
}): boolean {
	const { renderTransport, renderSurfaceDiagnostics, acceptedCpuPublication } = params;
	if (renderTransport !== 'cpu-uploaded-selected-hour' || acceptedCpuPublication == null) {
		return false;
	}

	return (
		renderSurfaceDiagnostics.utciSurfaceSource === 'cpu-uploaded-selected-hour' &&
		renderSurfaceDiagnostics.cpuPublishRequestId === acceptedCpuPublication.requestId &&
		renderSurfaceDiagnostics.cpuPublishMonthIndex === acceptedCpuPublication.monthIndex &&
		renderSurfaceDiagnostics.cpuPublishHourIndex === acceptedCpuPublication.hourIndex &&
		renderSurfaceDiagnostics.cpuPublishTimeIndex === acceptedCpuPublication.timeIndex &&
		renderSurfaceDiagnostics.cpuPublishSelectionKey === acceptedCpuPublication.selectionKey
	);
}

function hasAcceptedGpuRenderSurface(
	state: LiveSelectedHourControllerMutableState
): boolean {
	const acceptedRequestId = state.acceptedGpuResidentOutput?.requestId;
	return (
		acceptedRequestId !== undefined &&
		state.sameDeviceForComputeAndRender === true &&
		state.renderSurfaceDiagnostics.gpuResidentCopyRequestId === acceptedRequestId &&
		state.renderSurfaceDiagnostics.gpuResidentCopyStatus === 'complete' &&
		state.renderSurfaceDiagnostics.utciSurfaceSource === 'compute-buffer-selected-hour'
	);
}

function deriveState(
	state: LiveSelectedHourControllerMutableState,
	acceptedCpuPublication: AcceptedCpuPublication | null
): LiveSelectedHourControllerState {
	const awaitingGpuSurface =
		state.acceptedGpuResidentOutput != null &&
		state.renderTransport === 'compute-buffer-selected-hour' &&
		!hasAcceptedGpuRenderSurface(state);
	const ready = state.analysis != null || state.acceptedGpuResidentOutput != null;
	const cpuRenderReady =
		state.renderTransport !== 'cpu-uploaded-selected-hour' ||
		acceptedCpuPublication == null ||
		hasAcceptedCpuRenderSurface({
			renderTransport: state.renderTransport,
			renderSurfaceDiagnostics: state.renderSurfaceDiagnostics,
			acceptedCpuPublication
		});
	return {
		...state,
		awaitingGpuSurface,
		ready,
		renderReady: ready && !awaitingGpuSurface && cpuRenderReady
	};
}

function createInitialState(): LiveSelectedHourControllerState {
	return deriveState({
		analysis: null,
		acceptedGpuResidentOutput: null,
		surfaceIdentity: null,
		loading: false,
		error: null,
		renderTransport: 'idle',
		sameDeviceForComputeAndRender: null,
		pendingRenderUpdateStartedAt: undefined,
		renderSurfaceDiagnostics: EMPTY_SURFACE_DIAGNOSTICS
	}, null);
}

function isAbortError(error: unknown): boolean {
	return error instanceof DOMException && error.name === 'AbortError';
}

function withControllerRequestId(
	gpuResidentOutput: SelectedHourGpuResidentOutput | null,
	requestId: number
): SelectedHourGpuResidentOutput | null {
	return gpuResidentOutput ? { ...gpuResidentOutput, requestId } : null;
}

function mergeRenderSurfaceDiagnostics(
	current: LiveSelectedHourControllerSurfaceDiagnostics,
	diagnostics: LiveSelectedHourControllerSurfaceDiagnostics,
	trackedGpuRequestId: number | undefined,
	acceptedCpuPublication: AcceptedCpuPublication | null
): LiveSelectedHourControllerSurfaceDiagnostics {
	if (Object.keys(diagnostics).length === 0) {
		return current;
	}

	const hasGpuRequestScopedUpdate =
		diagnostics.utciSurfaceSource === 'compute-buffer-selected-hour' ||
		(diagnostics.gpuResidentCopyStatus !== undefined &&
			diagnostics.gpuResidentCopyStatus !== 'idle') ||
		diagnostics.gpuResidentCopyError !== undefined ||
		diagnostics.gpuResidentCopyRequestId !== undefined;
	if (
		hasGpuRequestScopedUpdate &&
		(trackedGpuRequestId === undefined ||
			diagnostics.gpuResidentCopyRequestId !== trackedGpuRequestId)
	) {
		return {
			...current,
			selectedHourTransferCount:
				diagnostics.selectedHourTransferCount ?? current.selectedHourTransferCount,
			dataTextureBuildCount:
				diagnostics.dataTextureBuildCount ?? current.dataTextureBuildCount
		};
	}

	const hasCpuRequestScopedUpdate =
		diagnostics.utciSurfaceSource === 'cpu-uploaded-selected-hour' ||
		diagnostics.cpuPublishRequestId !== undefined ||
		diagnostics.cpuPublishMonthIndex !== undefined ||
		diagnostics.cpuPublishHourIndex !== undefined ||
		diagnostics.cpuPublishTimeIndex !== undefined ||
		diagnostics.cpuPublishSelectionKey !== undefined;
	if (hasCpuRequestScopedUpdate && acceptedCpuPublication != null) {
		const matchesAcceptedCpuPublication =
			diagnostics.cpuPublishRequestId === acceptedCpuPublication.requestId &&
			diagnostics.cpuPublishMonthIndex === acceptedCpuPublication.monthIndex &&
			diagnostics.cpuPublishHourIndex === acceptedCpuPublication.hourIndex &&
			diagnostics.cpuPublishTimeIndex === acceptedCpuPublication.timeIndex &&
			diagnostics.cpuPublishSelectionKey === acceptedCpuPublication.selectionKey;
		if (!matchesAcceptedCpuPublication) {
			return current;
		}
	}
	if (hasCpuRequestScopedUpdate && acceptedCpuPublication == null) {
		return current;
	}

	const next = {
		...current,
		...diagnostics
	};
	const isAcceptedIdleCpuPublication =
		hasCpuRequestScopedUpdate &&
		diagnostics.gpuResidentCopyStatus === 'idle' &&
		diagnostics.utciSurfaceSource === 'cpu-uploaded-selected-hour';
	if (isAcceptedIdleCpuPublication) {
		if (!Object.prototype.hasOwnProperty.call(diagnostics, 'gpuResidentCopyError')) {
			delete next.gpuResidentCopyError;
		}
		if (!Object.prototype.hasOwnProperty.call(diagnostics, 'gpuResidentCopyRequestId')) {
			delete next.gpuResidentCopyRequestId;
		}
	}

	return next;
}

function areDiagnosticsEqual(
	left: LiveSelectedHourControllerSurfaceDiagnostics,
	right: LiveSelectedHourControllerSurfaceDiagnostics
): boolean {
	const leftKeys = Object.keys(left);
	const rightKeys = Object.keys(right);
	if (leftKeys.length !== rightKeys.length) {
		return false;
	}
	for (const key of leftKeys) {
		if (left[key as keyof LiveSelectedHourControllerSurfaceDiagnostics] !== right[key as keyof LiveSelectedHourControllerSurfaceDiagnostics]) {
			return false;
		}
	}
	return true;
}

export function createLiveSelectedHourController(
	options: CreateLiveSelectedHourControllerOptions = {}
): LiveSelectedHourController {
	const prepareSession = options.prepareSession ?? prepareSelectedHourLiveSession;
	const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
	let disposed = false;
	let state = createInitialState();
	let activeRequestToken = 0;
	let sessionEpoch = 0;
	let currentSessionKey: string | null = null;
	let currentSession: SelectedHourLiveSession | null = null;
	let currentSessionPromise: Promise<SelectedHourLiveSession> | null = null;
	let currentSessionAbortController: AbortController | null = null;
	let deferredCpuFallback: DeferredCpuFallbackState | null = null;
	let acceptedCpuPublication: AcceptedCpuPublication | null = null;

	function emit(): void {
		const snapshot = cloneState(state);
		for (const listener of listeners) {
			listener(snapshot);
		}
	}

	function setState(
		updater:
			| Partial<LiveSelectedHourControllerMutableState>
			| ((
					current: LiveSelectedHourControllerState
			  ) => Partial<LiveSelectedHourControllerMutableState>)
	): void {
		const patch = typeof updater === 'function' ? updater(state) : updater;
		state = deriveState({
			analysis: 'analysis' in patch ? patch.analysis ?? null : state.analysis,
			acceptedGpuResidentOutput:
				'acceptedGpuResidentOutput' in patch
					? patch.acceptedGpuResidentOutput ?? null
					: state.acceptedGpuResidentOutput,
			surfaceIdentity:
				'surfaceIdentity' in patch ? patch.surfaceIdentity ?? null : state.surfaceIdentity,
			loading: 'loading' in patch ? patch.loading ?? false : state.loading,
			error: 'error' in patch ? patch.error ?? null : state.error,
			renderTransport: patch.renderTransport ?? state.renderTransport,
			sameDeviceForComputeAndRender:
				'sameDeviceForComputeAndRender' in patch
					? patch.sameDeviceForComputeAndRender ?? null
					: state.sameDeviceForComputeAndRender,
			pendingRenderUpdateStartedAt:
				'pendingRenderUpdateStartedAt' in patch
					? patch.pendingRenderUpdateStartedAt
					: state.pendingRenderUpdateStartedAt,
			renderSurfaceDiagnostics:
				patch.renderSurfaceDiagnostics ?? state.renderSurfaceDiagnostics
		}, acceptedCpuPublication);
		emit();
	}

	function replaceAcceptedGpuResidentOutput(
		next: SelectedHourGpuResidentOutput | null,
		patch: Partial<LiveSelectedHourControllerMutableState>
	): void {
		const previous = state.acceptedGpuResidentOutput;
		setState({
			...patch,
			acceptedGpuResidentOutput: next
		});
		if (previous && previous.output !== next?.output) {
			disposeSelectedHourGpuResidentOutput(previous);
		}
	}

	function resetControllerState(): void {
		deferredCpuFallback = null;
		acceptedCpuPublication = null;
		const previous = state.acceptedGpuResidentOutput;
		if (previous) {
			disposeSelectedHourGpuResidentOutput(previous);
		}
		state = createInitialState();
		emit();
	}

	function disposeCurrentSession(): void {
		currentSessionAbortController?.abort();
		currentSessionAbortController = null;
		currentSessionPromise = null;
		currentSession?.dispose();
		currentSession = null;
		currentSessionKey = null;
		sessionEpoch += 1;
	}

	async function ensureSession(
		request: LiveSelectedHourControllerRequest
	): Promise<SelectedHourLiveSession> {
		if (disposed) {
			throw new DOMException('Aborted', 'AbortError');
		}

		if (currentSessionKey === request.sessionKey) {
			if (currentSession) return currentSession;
			if (currentSessionPromise) return currentSessionPromise;
		}

		disposeCurrentSession();
		resetControllerState();
		setState({ loading: true });

		const abortController = new AbortController();
		const requestedEpoch = sessionEpoch;
		currentSessionAbortController = abortController;
		currentSessionKey = request.sessionKey;
		const sessionPromise = prepareSession({
			...request.sessionConfig,
			signal: abortController.signal
		})
			.then((session) => {
				if (
					disposed ||
					abortController.signal.aborted ||
					requestedEpoch !== sessionEpoch ||
					currentSessionKey !== request.sessionKey
				) {
					session.dispose();
					throw new DOMException('Aborted', 'AbortError');
				}
				currentSession = session;
				return session;
			})
			.finally(() => {
				if (currentSessionPromise === sessionPromise) {
					currentSessionPromise = null;
				}
			});
		currentSessionPromise = sessionPromise;
		return sessionPromise;
	}

	function ownsRequest(requestToken: number): boolean {
		return !disposed && requestToken === activeRequestToken;
	}

	return {
		getState() {
			return cloneState(state);
		},

		subscribe(listener) {
			listeners.add(listener);
			return () => {
				listeners.delete(listener);
			};
		},

		async requestSelection(request) {
			const requestToken = ++activeRequestToken;
			deferredCpuFallback = null;
			setState({ loading: true, error: null });

			try {
				const session = await ensureSession(request);
				if (!ownsRequest(requestToken)) {
					return {
						accepted: false,
						reason: disposed ? 'disposed' : 'stale',
						state: cloneState(state)
					};
				}

				const result = await session.runSelectedHour({
					monthIndex: request.monthIndex,
					hourIndex: request.hourIndex,
					timeIndex: request.timeIndex,
					colorMode: request.colorMode,
					preferGpuResident: request.preferGpuResident,
					rendererDevice: request.rendererDevice
				});
				if (!ownsRequest(requestToken)) {
					disposeSelectedHourGpuResidentOutput(result.gpuResidentOutput);
					return {
						accepted: false,
						reason: disposed ? 'disposed' : 'stale',
						state: cloneState(state)
					};
				}

				const controllerRequestId = requestToken;
				const acceptedGpuResidentOutput = withControllerRequestId(
					result.gpuResidentOutput,
					controllerRequestId
				);
				const acceptedSelectionKey =
					request.selectionKey ??
					createSelectionKey({
						requestId: controllerRequestId,
						monthIndex: result.monthIndex,
						hourIndex: result.hourIndex,
						timeIndex: result.timeIndex
					});
				deferredCpuFallback = acceptedGpuResidentOutput
					? {
							requestId: controllerRequestId,
							monthIndex: result.monthIndex,
							hourIndex: result.hourIndex,
							timeIndex: result.timeIndex,
							analysis: result.analysis,
							loadCpuFallback: result.loadCpuFallback
						}
					: null;
				acceptedCpuPublication =
					result.renderTransport === 'cpu-uploaded-selected-hour'
						? {
								requestId: controllerRequestId,
								monthIndex: result.monthIndex,
								hourIndex: result.hourIndex,
								timeIndex: result.timeIndex,
								selectionKey: acceptedSelectionKey
							}
						: null;

				replaceAcceptedGpuResidentOutput(acceptedGpuResidentOutput, {
					analysis: result.analysis,
					surfaceIdentity: createSurfaceIdentity({
						requestId: controllerRequestId,
						monthIndex: result.monthIndex,
						hourIndex: result.hourIndex,
						timeIndex: result.timeIndex,
						selectionKey: acceptedSelectionKey,
						pendingRenderUpdateStartedAt:
							result.renderTransport === 'compute-buffer-selected-hour'
								? result.pendingRenderUpdateStartedAt
								: undefined,
						acceptedGpuResidentOutput: acceptedGpuResidentOutput
					}),
					loading: result.renderTransport === 'compute-buffer-selected-hour',
					error: null,
					renderTransport: result.renderTransport,
					sameDeviceForComputeAndRender: result.sameDeviceForComputeAndRender,
					pendingRenderUpdateStartedAt:
						result.renderTransport === 'compute-buffer-selected-hour'
							? result.pendingRenderUpdateStartedAt
							: undefined,
					renderSurfaceDiagnostics:
						result.renderTransport === 'compute-buffer-selected-hour'
							? {
									gpuResidentCopyStatus: 'pending',
									gpuResidentCopyRequestId: controllerRequestId
								}
							: EMPTY_SURFACE_DIAGNOSTICS
				});

				return { accepted: true, state: cloneState(state) };
			} catch (error) {
				if (isAbortError(error) || !ownsRequest(requestToken)) {
					return {
						accepted: false,
						reason: disposed ? 'disposed' : 'stale',
						state: cloneState(state)
					};
				}

				deferredCpuFallback = null;
				setState({
					loading: false,
					error: error instanceof Error ? error.message : 'Failed to compute live UTCI.',
					pendingRenderUpdateStartedAt: undefined,
					renderSurfaceDiagnostics:
						state.acceptedGpuResidentOutput == null
							? EMPTY_SURFACE_DIAGNOSTICS
							: state.renderSurfaceDiagnostics
				});
				return { accepted: false, state: cloneState(state) };
			}
		},

		async handleRenderSurfaceDiagnostics(diagnostics) {
			if (disposed) return;

			const acceptedRequestId = state.acceptedGpuResidentOutput?.requestId;
			const requestId = diagnostics.gpuResidentCopyRequestId;
			const nextDiagnostics = mergeRenderSurfaceDiagnostics(
				state.renderSurfaceDiagnostics,
				diagnostics,
				acceptedRequestId,
				acceptedCpuPublication
			);
			const acceptsGpuCompletion =
				diagnostics.gpuResidentCopyStatus === 'complete' &&
				requestId !== undefined &&
				acceptedRequestId === requestId &&
				state.sameDeviceForComputeAndRender === true &&
				state.renderTransport === 'compute-buffer-selected-hour' &&
				diagnostics.utciSurfaceSource === 'compute-buffer-selected-hour';
			if (
				diagnostics.gpuResidentCopyStatus === 'complete' &&
				!acceptsGpuCompletion
			) {
				return;
			}
			if (acceptsGpuCompletion) {
				deferredCpuFallback = null;
				acceptedCpuPublication = null;
				setState({
					surfaceIdentity: state.surfaceIdentity
						? {
								...state.surfaceIdentity,
								pendingRenderUpdateStartedAt: undefined,
								acceptedGpuResidentOutput: state.acceptedGpuResidentOutput
							}
						: null,
					renderSurfaceDiagnostics: nextDiagnostics,
					loading: false,
					renderTransport: 'compute-buffer-selected-hour',
					pendingRenderUpdateStartedAt: undefined
				});
				return;
			}

			const shouldHandleGpuFallback =
				diagnostics.gpuResidentCopyStatus === 'failed' &&
				requestId !== undefined &&
				deferredCpuFallback?.requestId === requestId;
			if (
				areDiagnosticsEqual(nextDiagnostics, state.renderSurfaceDiagnostics) &&
				!shouldHandleGpuFallback
			) {
				return;
			}

			setState({ renderSurfaceDiagnostics: nextDiagnostics });

			if (shouldHandleGpuFallback) {
				const fallbackRequest = deferredCpuFallback;
				if (fallbackRequest == null) {
					return;
				}
				const ownsDeferredFallback = () =>
					!disposed &&
					state.acceptedGpuResidentOutput?.requestId === requestId &&
					deferredCpuFallback?.requestId === requestId;
				let fallbackAnalysis = fallbackRequest.analysis;
				if (!fallbackAnalysis && fallbackRequest.loadCpuFallback) {
					try {
						const fallback = await fallbackRequest.loadCpuFallback();
						if (!ownsDeferredFallback()) {
							return;
						}
						fallbackAnalysis = fallback.analysis;
					} catch (error) {
						if (!ownsDeferredFallback()) {
							return;
						}
						deferredCpuFallback = null;
						replaceAcceptedGpuResidentOutput(null, {
							analysis: null,
							loading: false,
							error:
								error instanceof Error
									? `GPU copy failed and CPU fallback failed: ${error.message}`
									: 'GPU copy failed and CPU fallback failed.',
							renderTransport: 'cpu-uploaded-selected-hour',
							pendingRenderUpdateStartedAt: undefined
						});
						return;
					}
				}
				if (!fallbackAnalysis) {
					deferredCpuFallback = null;
					replaceAcceptedGpuResidentOutput(null, {
						analysis: null,
						loading: false,
						error: 'GPU copy failed and no CPU fallback analysis was available.',
						renderTransport: 'cpu-uploaded-selected-hour',
						pendingRenderUpdateStartedAt: undefined
					});
					return;
				}
				acceptedCpuPublication = {
					requestId: fallbackRequest.requestId,
					monthIndex: fallbackRequest.monthIndex,
					hourIndex: fallbackRequest.hourIndex,
					timeIndex: fallbackRequest.timeIndex,
					selectionKey:
						state.surfaceIdentity?.selectionKey ??
						createSelectionKey({
							requestId: fallbackRequest.requestId,
							monthIndex: fallbackRequest.monthIndex,
							hourIndex: fallbackRequest.hourIndex,
							timeIndex: fallbackRequest.timeIndex
						})
				};
				deferredCpuFallback = null;
				replaceAcceptedGpuResidentOutput(null, {
					analysis: fallbackAnalysis,
					surfaceIdentity: createSurfaceIdentity({
						requestId: acceptedCpuPublication.requestId,
						monthIndex: acceptedCpuPublication.monthIndex,
						hourIndex: acceptedCpuPublication.hourIndex,
						timeIndex: acceptedCpuPublication.timeIndex,
						selectionKey: acceptedCpuPublication.selectionKey,
						pendingRenderUpdateStartedAt: undefined,
						acceptedGpuResidentOutput: null
					}),
					loading: false,
					error: null,
					renderTransport: 'cpu-uploaded-selected-hour',
					pendingRenderUpdateStartedAt: undefined
				});
			}
		},

		dispose() {
			if (disposed) return;
			disposed = true;
			disposeCurrentSession();
			resetControllerState();
			listeners.clear();
		}
	};
}
