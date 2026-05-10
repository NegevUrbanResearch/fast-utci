import { describe, expect, it } from 'vitest';
import { parseDebugWebgpuUtciQuery } from '$lib/debug/debugWebgpuUtciQuery';

function params(query: string): URLSearchParams {
	return new URLSearchParams(query);
}

describe('parseDebugWebgpuUtciQuery', () => {
	it('defaults the debug route to August Python bin comparison mode', () => {
		const state = parseDebugWebgpuUtciQuery(params(''));
		expect(state.parityMode).toBe(true);
		expect(state.collectMode).toBe('off');
		expect(state.debugOnDemandMode).toBe('f32');
		expect(state.binComparisonEnabled).toBe(true);
		expect(state.binComparisonValid).toBe(true);
	});

	it('keeps parity in f32 on-demand mode by default', () => {
		const state = parseDebugWebgpuUtciQuery(params('parity=1&timeIndex=23'));
		expect(state.parityMode).toBe(true);
		expect(state.collectMode).toBe('off');
		expect(state.debugOnDemandMode).toBe('f32');
		expect(state.binComparisonEnabled).toBe(true);
		expect(state.binComparisonValid).toBe(true);
	});

	it('allows parity f32 without an explicit month because debug defaults to August baseline', () => {
		const state = parseDebugWebgpuUtciQuery(params('parity=1&utciOnDemand=f32'));
		expect(state.debugOnDemandMode).toBe('f32');
		expect(state.binComparisonEnabled).toBe(true);
		expect(state.binComparisonValid).toBe(true);
	});

	it('allows explicit f32 on-demand in parity mode while keeping August validity separate', () => {
		const state = parseDebugWebgpuUtciQuery(params('parity=1&utciOnDemand=f32&monthIndex=7'));
		expect(state.debugOnDemandMode).toBe('f32');
		expect(state.binComparisonEnabled).toBe(true);
		expect(state.binComparisonValid).toBe(true);
	});

	it('does not claim bin validity for non-August parity months', () => {
		const state = parseDebugWebgpuUtciQuery(params('parity=1&utciOnDemand=f32&monthIndex=3'));
		expect(state.binComparisonEnabled).toBe(true);
		expect(state.binComparisonValid).toBe(false);
	});

	it('forces normal collect mode away from f32 on-demand by default', () => {
		const state = parseDebugWebgpuUtciQuery(params('collect=normal&monthIndex=3&hour=9'));
		expect(state.parityMode).toBe(false);
		expect(state.collectMode).toBe('normal');
		expect(state.debugOnDemandMode).toBe('off');
		expect(state.binComparisonEnabled).toBe(false);
	});

	it('preserves onDemandPrototype as an f32 opt-in', () => {
		const state = parseDebugWebgpuUtciQuery(params('parity=1&onDemandPrototype=1'));
		expect(state.debugOnDemandMode).toBe('f32');
	});

	it('allows explicit shared-host WebGPU mode by opting out of parity', () => {
		const state = parseDebugWebgpuUtciQuery(params('parity=0'));
		expect(state.parityMode).toBe(false);
		expect(state.debugOnDemandMode).toBe('f32');
		expect(state.binComparisonEnabled).toBe(false);
		expect(state.binComparisonValid).toBe(false);
	});
});
