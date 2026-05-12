import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/selected-hour/liveSelectedHourSurfaceIdentity';
import type { SelectedHourGpuResidentOutput } from '$lib/compute/selected-hour/liveUtciSelectedHourSession';
import type { SelectedHourRenderTimingSubsteps } from '$lib/compute/on-demand/onDemandDiagnostics';
import {
	createAcceptedGpuResidentOutputReleaseNotifier,
	type AcceptedGpuResidentOutputReleaseCallback,
	type AcceptedGpuResidentOutputReleaseReason
} from '$lib/components/scene/acceptedGpuResidentOutputRelease';
import {
	getAcceptedGpuResidentKey,
	type GpuResidentCopyStatus
} from '$lib/components/scene/utciSurfaceSync';

export type AcceptedGpuResidentSurfaceSyncRun = {
	syncKey: string;
	syncRunKey: string;
	requestId: number;
	controllerIdentity: string;
	controllerInstanceId: number;
	copyRunToken: number;
	notifyAcceptedOutputRelease: (
		reason: AcceptedGpuResidentOutputReleaseReason
	) => boolean;
};

export type AcceptedGpuResidentSurfaceSyncLiveState = {
	acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null;
	liveSelectedHourSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
};

export type AcceptedGpuResidentSurfaceSyncTerminalResult =
	| 'complete'
	| 'failed'
	| 'superseded'
	| 'already-released';

type GpuResidentCopyDiagnosticsOptions = {
	error?: string;
	requestId?: number;
	renderTimings?: SelectedHourRenderTimingSubsteps;
};

export function createAcceptedGpuResidentSurfaceSync(params: {
	componentName: string;
	getOnAcceptedGpuResidentOutputRelease: () =>
		| AcceptedGpuResidentOutputReleaseCallback
		| undefined;
	setCopyDiagnostics: (
		status: GpuResidentCopyStatus,
		options?: GpuResidentCopyDiagnosticsOptions
	) => void;
}) {
	let activeSyncKey: string | null = null;
	let activeSyncRunKey: string | null = null;
	let activeCopyRunToken = 0;

	function getSyncRunKey(input: {
		acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null;
		liveSelectedHourSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
	}): string | null {
		if (!input.acceptedGpuResidentOutput) {
			return null;
		}
		const syncKey = getAcceptedGpuResidentKey(input.acceptedGpuResidentOutput);
		const controllerIdentity =
			input.liveSelectedHourSurfaceIdentity?.controllerIdentity;
		const controllerInstanceId =
			input.liveSelectedHourSurfaceIdentity?.controllerInstanceId;
		if (!syncKey || !controllerIdentity || controllerInstanceId == null) {
			return null;
		}
		return `${syncKey}|${controllerIdentity}|${controllerInstanceId}`;
	}

	function reset(options: { invalidateActiveRun?: boolean } = {}): void {
		if (options.invalidateActiveRun) {
			activeCopyRunToken += 1;
		}
		activeSyncKey = null;
		activeSyncRunKey = null;
		params.setCopyDiagnostics('idle');
	}

	function startSync(input: {
		acceptedOutput: SelectedHourGpuResidentOutput;
		liveSelectedHourSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
	}): AcceptedGpuResidentSurfaceSyncRun | null {
		const syncKey = getAcceptedGpuResidentKey(input.acceptedOutput);
		if (!syncKey) {
			return null;
		}

		const notifyAcceptedOutputRelease =
			createAcceptedGpuResidentOutputReleaseNotifier({
				callback: params.getOnAcceptedGpuResidentOutputRelease(),
				componentName: params.componentName,
				controllerIdentity:
					input.liveSelectedHourSurfaceIdentity?.controllerIdentity,
				controllerInstanceId:
					input.liveSelectedHourSurfaceIdentity?.controllerInstanceId,
				requestId: input.acceptedOutput.requestId,
				monthIndex: input.acceptedOutput.monthIndex,
				timeIndex: input.acceptedOutput.timeIndex
			});

		if (
			!input.liveSelectedHourSurfaceIdentity?.controllerIdentity ||
			input.liveSelectedHourSurfaceIdentity.controllerInstanceId == null
		) {
			reset({ invalidateActiveRun: true });
			return null;
		}

		const copyRunToken = ++activeCopyRunToken;
		const syncRunKey = `${syncKey}|${input.liveSelectedHourSurfaceIdentity.controllerIdentity}|${input.liveSelectedHourSurfaceIdentity.controllerInstanceId}`;
		activeSyncKey = syncKey;
		activeSyncRunKey = syncRunKey;
		params.setCopyDiagnostics('pending', {
			requestId: input.acceptedOutput.requestId
		});
		return {
			syncKey,
			syncRunKey,
			requestId: input.acceptedOutput.requestId,
			controllerIdentity: input.liveSelectedHourSurfaceIdentity.controllerIdentity,
			controllerInstanceId:
				input.liveSelectedHourSurfaceIdentity.controllerInstanceId,
			copyRunToken,
			notifyAcceptedOutputRelease
		};
	}

	function isSuperseded(
		run: AcceptedGpuResidentSurfaceSyncRun,
		liveState: AcceptedGpuResidentSurfaceSyncLiveState
	): boolean {
		return (
			run.copyRunToken !== activeCopyRunToken ||
			activeSyncKey !== run.syncKey ||
			activeSyncRunKey !== run.syncRunKey ||
			liveState.acceptedGpuResidentOutput?.requestId !== run.requestId ||
			liveState.liveSelectedHourSurfaceIdentity?.controllerIdentity !==
				run.controllerIdentity ||
			liveState.liveSelectedHourSurfaceIdentity?.controllerInstanceId !==
				run.controllerInstanceId
		);
	}

	function completeSync(
		run: AcceptedGpuResidentSurfaceSyncRun,
		paramsForCompletion: AcceptedGpuResidentSurfaceSyncLiveState & {
			renderTimings: SelectedHourRenderTimingSubsteps;
		}
	): AcceptedGpuResidentSurfaceSyncTerminalResult {
		if (isSuperseded(run, paramsForCompletion)) {
			run.notifyAcceptedOutputRelease('superseded');
			return 'superseded';
		}
		if (!run.notifyAcceptedOutputRelease('copy-complete')) {
			return 'already-released';
		}
		params.setCopyDiagnostics('complete', {
			requestId: run.requestId,
			renderTimings: paramsForCompletion.renderTimings
		});
		return 'complete';
	}

	function supersedeSync(
		run: AcceptedGpuResidentSurfaceSyncRun
	): AcceptedGpuResidentSurfaceSyncTerminalResult {
		return run.notifyAcceptedOutputRelease('superseded')
			? 'superseded'
			: 'already-released';
	}

	function failSync(
		run: AcceptedGpuResidentSurfaceSyncRun,
		paramsForFailure: AcceptedGpuResidentSurfaceSyncLiveState & {
			errorMessage: string;
		}
	): AcceptedGpuResidentSurfaceSyncTerminalResult {
		if (
			isSuperseded(run, paramsForFailure) ||
			paramsForFailure.errorMessage.includes('superseded')
		) {
			run.notifyAcceptedOutputRelease('superseded');
			return 'superseded';
		}
		if (!run.notifyAcceptedOutputRelease('copy-failed')) {
			return 'already-released';
		}
		params.setCopyDiagnostics('failed', {
			error: paramsForFailure.errorMessage,
			requestId: run.requestId
		});
		return 'failed';
	}

	return {
		completeSync,
		failSync,
		getActiveSyncKey: () => activeSyncKey,
		getActiveSyncRunKey: () => activeSyncRunKey,
		getActiveCopyRunToken: () => activeCopyRunToken,
		getSyncRunKey,
		isSuperseded,
		reset,
		startSync,
		supersedeSync
	};
}
