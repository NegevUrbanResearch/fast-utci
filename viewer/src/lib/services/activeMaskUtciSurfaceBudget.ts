import {
	INNOVATION_DISTRICT_05M_ACTIVE_MASK_SURFACE_SHAPE,
	type ActiveMaskUtciSurfaceShape
} from '$lib/services/activeMaskUtciSurfaceGeometry';
import {
	DEFAULT_ACTIVE_MASK_UTCI_SURFACE_BUDGET_LIMITS,
	estimateActiveMaskUtciSurfaceStrategies,
	type ActiveMaskUtciSurfaceBudgetEstimate,
	type ActiveMaskUtciSurfaceBudgetLimitInputs,
	type ActiveMaskUtciSurfaceBudgetLimits,
	type ActiveMaskUtciSurfaceStrategy
} from '$lib/services/activeMaskUtciSurfaceBudgetMath';

export { DEFAULT_ACTIVE_MASK_UTCI_SURFACE_BUDGET_LIMITS };
export type {
	ActiveMaskUtciSurfaceBudgetEstimate,
	ActiveMaskUtciSurfaceBudgetLimitInputs,
	ActiveMaskUtciSurfaceBudgetLimits,
	ActiveMaskUtciSurfaceStorageBufferEstimate,
	ActiveMaskUtciSurfaceStrategy
} from '$lib/services/activeMaskUtciSurfaceBudgetMath';

export type ActiveMaskUtciSurfaceBudgetDecision = {
	input: ActiveMaskUtciSurfaceShape;
	limits: ActiveMaskUtciSurfaceBudgetLimits;
	estimates: ActiveMaskUtciSurfaceBudgetEstimate[];
	selectedStrategy: ActiveMaskUtciSurfaceStrategy;
	selectionReason: string;
	planRevisionRequired: boolean;
	planRevisionText?: string;
};

function findEstimate(
	estimates: ActiveMaskUtciSurfaceBudgetEstimate[],
	strategy: ActiveMaskUtciSurfaceStrategy
): ActiveMaskUtciSurfaceBudgetEstimate {
	const estimate = estimates.find((candidate) => candidate.strategy === strategy);
	if (!estimate) {
		throw new Error(`Missing active-mask UTCI surface estimate for ${strategy}.`);
	}
	return estimate;
}

function buildPlanRevisionText(
	strategy: ActiveMaskUtciSurfaceStrategy
): string | undefined {
	if (strategy !== 'active-tiled-instanced-quads') {
		return undefined;
	}
	return [
		'Active instanced remains the only approved first-slice render path for Innovation District 0.5m.',
		'The current renderer or browser limits require an internal active-instanced tiling fallback review before allocation. Keep the tiny indexed quad plus per-instance active canonical cell indices, but split active canonical indices and selected-hour UTCI storage into safe tiles that fit the actual device limits.'
	].join(' ');
}

export function buildActiveMaskUtciSurfaceBudgetDecision(
	shape: ActiveMaskUtciSurfaceShape,
	limitInputs: ActiveMaskUtciSurfaceBudgetLimitInputs = {}
): ActiveMaskUtciSurfaceBudgetDecision {
	const { limits, estimates } = estimateActiveMaskUtciSurfaceStrategies(
		shape,
		limitInputs
	);
	const activeIndexed = findEstimate(estimates, 'active-indexed-quads');
	const activeInstanced = findEstimate(estimates, 'active-instanced-quads');
	const tiledInstanced = findEstimate(estimates, 'active-tiled-instanced-quads');
	const activeInstancedFits =
		activeInstanced.fits.jsLargestTypedArray &&
		activeInstanced.fits.jsTotalTypedArray &&
		activeInstanced.fits.maxBufferSize &&
		activeInstanced.fits.maxStorageBufferBindingSize;
	const activeIndexedComfortablyFits =
		activeIndexed.fits.comfortableJsLargestTypedArray &&
		activeIndexed.fits.comfortableJsTotalTypedArray &&
		activeIndexed.fits.maxBufferSize &&
		activeIndexed.fits.maxStorageBufferBindingSize;

	let selectedStrategy: ActiveMaskUtciSurfaceStrategy;
	let selectionReason: string;
	if (activeInstancedFits) {
		selectedStrategy = 'active-instanced-quads';
		selectionReason = activeIndexedComfortablyFits
			? 'Active indexed also fits this budget snapshot, but the revised first-slice implementation is locked to active instanced. Active instanced stays selected because it keeps the implementation on the approved renderer topology while remaining within the current JS typed-array and WebGPU device limits.'
			: 'Active indexed remains diagnostic-only because its largest single typed-array allocation is near or above the practical browser threshold; active instanced stays on the approved first-slice topology while keeping per-buffer allocations within the current JS typed-array and WebGPU device limits.';
	} else {
		selectedStrategy = 'active-tiled-instanced-quads';
		selectionReason =
			'Single active instancing exceeds at least one current JS typed-array or WebGPU device limit, so the approved active-instanced topology needs an internal tiling fallback before allocation.';
	}

	if (
		selectedStrategy === 'active-tiled-instanced-quads' &&
		!tiledInstanced.fits.maxStorageBufferBindingSize
	) {
		selectionReason +=
			' The tiled instanced estimate also exceeds storage binding limits and needs a smaller tile policy.';
	}

	return {
		input: { ...shape },
		limits,
		estimates,
		selectedStrategy,
		selectionReason,
		planRevisionRequired: selectedStrategy === 'active-tiled-instanced-quads',
		planRevisionText: buildPlanRevisionText(selectedStrategy)
	};
}

export function buildInnovationDistrict05mUtciSurfaceBudgetDecision(
	limitInputs: ActiveMaskUtciSurfaceBudgetLimitInputs = {}
): ActiveMaskUtciSurfaceBudgetDecision {
	return buildActiveMaskUtciSurfaceBudgetDecision(
		INNOVATION_DISTRICT_05M_ACTIVE_MASK_SURFACE_SHAPE,
		limitInputs
	);
}
