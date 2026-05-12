import { invokeDiagnosticsCallbackSafely } from '$lib/compute/on-demand/onDemandDiagnostics';

export type AcceptedGpuResidentOutputReleaseReason =
	| 'copy-complete'
	| 'copy-failed'
	| 'superseded';

export type AcceptedGpuResidentOutputReleasePayload = {
	controllerIdentity: string;
	controllerInstanceId: number;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
	reason: AcceptedGpuResidentOutputReleaseReason;
};

export type AcceptedGpuResidentOutputReleaseCallback = (
	params: AcceptedGpuResidentOutputReleasePayload
) => void | Promise<void>;

export function createAcceptedGpuResidentOutputReleaseNotifier(params: {
	callback: AcceptedGpuResidentOutputReleaseCallback | undefined;
	componentName: string;
	controllerIdentity: string | null | undefined;
	controllerInstanceId: number | null | undefined;
	requestId: number;
	monthIndex: number;
	timeIndex: number;
}): (reason: AcceptedGpuResidentOutputReleaseReason) => boolean {
	let releaseNotified = false;

	return (reason) => {
		if (releaseNotified) {
			return false;
		}
		if (!params.controllerIdentity || params.controllerInstanceId == null) {
			return false;
		}
		releaseNotified = true;
		invokeDiagnosticsCallbackSafely(
			params.callback,
			{
				controllerIdentity: params.controllerIdentity,
				controllerInstanceId: params.controllerInstanceId,
				requestId: params.requestId,
				monthIndex: params.monthIndex,
				timeIndex: params.timeIndex,
				reason
			},
			`${params.componentName} onAcceptedGpuResidentOutputRelease`
		);
		return true;
	};
}
