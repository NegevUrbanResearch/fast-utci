import type { Analysis } from '$lib/types/analysis';

export function createFullDayAnalysis(params: {
	label: string;
	sourceAnalysisId?: string;
	baseMin?: number;
	baseMax?: number;
}): Analysis {
	const baseMin = params.baseMin ?? 10;
	const baseMax = params.baseMax ?? 30;

	return {
		metadata: {
			analysis_type: 'full_day',
			num_positions: 2,
			hours: Array.from({ length: 24 }, (_, hour) => `${String(hour).padStart(2, '0')}:00`),
			utci_range: { min: baseMin, max: baseMax },
			grid_size: 2,
			coordinate_system: 'xy_ground',
			model_file: `${params.label}.glb`,
			source_analysis_id: params.sourceAnalysisId ?? params.label,
			num_months: 12,
			hour_statistics: Array.from({ length: 288 }, (_, index) => ({
				min: baseMin + (index % 24),
				max: baseMax + (index % 24),
				mean: (baseMin + baseMax) / 2 + (index % 24)
			}))
		},
		data: {
			numPositions: 2,
			numHours: 24,
			positions: new Float32Array([0, 0, 0, 1, 0, 0]),
			utciByHour: Array.from({ length: 288 }, () => new Float32Array([baseMin, baseMax]))
		}
	};
}
