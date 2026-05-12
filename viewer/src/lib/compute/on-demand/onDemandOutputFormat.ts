export type OnDemandOutputFormat = 'f32-utci' | 'packed-mrt-utci';

export interface OnDemandOutputFormatInfo {
	id: OnDemandOutputFormat;
	bytesPerPoint: number;
	includesMrt: boolean;
	requiresPacking: boolean;
	description: string;
}

export const ON_DEMAND_OUTPUT_FORMATS: Record<OnDemandOutputFormat, OnDemandOutputFormatInfo> =
	Object.freeze({
		'f32-utci': Object.freeze({
		id: 'f32-utci',
		bytesPerPoint: 4,
		includesMrt: false,
		requiresPacking: false,
		description: 'Baseline bridge format with one f32 UTCI value per point.'
		}),
		'packed-mrt-utci': Object.freeze({
			id: 'packed-mrt-utci',
			bytesPerPoint: 4,
			includesMrt: true,
			requiresPacking: true,
			description:
				'Experimental one u32 per point using pack2x16float(vec2<f32>(mrt, utci)).'
		})
	});

export function getOnDemandOutputFormat(id: OnDemandOutputFormat): OnDemandOutputFormatInfo {
	return ON_DEMAND_OUTPUT_FORMATS[id];
}
