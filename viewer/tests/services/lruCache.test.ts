import { describe, it, expect, beforeEach, vi } from 'vitest';
import { LRUCache } from '$lib/services/lruCache';

describe('LRUCache', () => {
	describe('Basic operations', () => {
		let cache: LRUCache<string>;

		beforeEach(() => {
			cache = new LRUCache<string>({ maxSize: 3 });
		});

		it('should store and retrieve values', () => {
			cache.set('key1', 'value1');
			expect(cache.get('key1')).toBe('value1');
		});

		it('should return undefined for non-existent keys', () => {
			expect(cache.get('nonexistent')).toBeUndefined();
		});

		it('should check if key exists', () => {
			cache.set('key1', 'value1');
			expect(cache.has('key1')).toBe(true);
			expect(cache.has('key2')).toBe(false);
		});

		it('should track cache size', () => {
			expect(cache.size).toBe(0);
			cache.set('key1', 'value1');
			expect(cache.size).toBe(1);
			cache.set('key2', 'value2');
			expect(cache.size).toBe(2);
		});

		it('should delete specific keys', () => {
			cache.set('key1', 'value1');
			cache.set('key2', 'value2');

			const deletedValue = cache.delete('key1');

			expect(deletedValue).toBe('value1');
			expect(cache.has('key1')).toBe(false);
			expect(cache.size).toBe(1);
		});

		it('should clear all items', () => {
			cache.set('key1', 'value1');
			cache.set('key2', 'value2');
			cache.set('key3', 'value3');

			cache.clear();

			expect(cache.size).toBe(0);
			expect(cache.has('key1')).toBe(false);
			expect(cache.has('key2')).toBe(false);
			expect(cache.has('key3')).toBe(false);
		});
	});

	describe('LRU eviction', () => {
		it('should evict least recently used item when capacity is reached', () => {
			const cache = new LRUCache<string>({ maxSize: 3 });

			cache.set('key1', 'value1'); // LRU order: key1
			cache.set('key2', 'value2'); // LRU order: key1, key2
			cache.set('key3', 'value3'); // LRU order: key1, key2, key3

			// Adding 4th item should evict key1 (least recently used)
			cache.set('key4', 'value4'); // LRU order: key2, key3, key4

			expect(cache.has('key1')).toBe(false);
			expect(cache.has('key2')).toBe(true);
			expect(cache.has('key3')).toBe(true);
			expect(cache.has('key4')).toBe(true);
			expect(cache.size).toBe(3);
		});

		it('should update LRU order when item is accessed', () => {
			const cache = new LRUCache<string>({ maxSize: 3 });

			cache.set('key1', 'value1'); // LRU order: key1
			cache.set('key2', 'value2'); // LRU order: key1, key2
			cache.set('key3', 'value3'); // LRU order: key1, key2, key3

			// Access key1 to make it most recently used
			cache.get('key1'); // LRU order: key2, key3, key1

			// Adding 4th item should now evict key2 (new least recently used)
			cache.set('key4', 'value4'); // LRU order: key3, key1, key4

			expect(cache.has('key1')).toBe(true);
			expect(cache.has('key2')).toBe(false);
			expect(cache.has('key3')).toBe(true);
			expect(cache.has('key4')).toBe(true);
		});

		it('should update LRU order when item is overwritten', () => {
			const cache = new LRUCache<string>({ maxSize: 3 });

			cache.set('key1', 'value1'); // LRU order: key1
			cache.set('key2', 'value2'); // LRU order: key1, key2
			cache.set('key3', 'value3'); // LRU order: key1, key2, key3

			// Overwrite key1 to make it most recently used
			cache.set('key1', 'value1-updated'); // LRU order: key2, key3, key1

			// Adding 4th item should now evict key2
			cache.set('key4', 'value4'); // LRU order: key3, key1, key4

			expect(cache.get('key1')).toBe('value1-updated');
			expect(cache.has('key2')).toBe(false);
			expect(cache.has('key3')).toBe(true);
			expect(cache.has('key4')).toBe(true);
		});
	});

	describe('Eviction callback', () => {
		it('should call onEvict when item is evicted due to capacity', () => {
			const onEvict = vi.fn();
			const cache = new LRUCache<string>({ maxSize: 2, onEvict });

			cache.set('key1', 'value1');
			cache.set('key2', 'value2');
			cache.set('key3', 'value3'); // Should evict key1

			expect(onEvict).toHaveBeenCalledWith('key1', 'value1');
			expect(onEvict).toHaveBeenCalledTimes(1);
		});

		it('should call onEvict when item is explicitly deleted', () => {
			const onEvict = vi.fn();
			const cache = new LRUCache<string>({ maxSize: 3, onEvict });

			cache.set('key1', 'value1');
			cache.set('key2', 'value2');
			cache.delete('key1');

			expect(onEvict).toHaveBeenCalledWith('key1', 'value1');
			expect(onEvict).toHaveBeenCalledTimes(1);
		});

		it('should call onEvict for all items when clearing', () => {
			const onEvict = vi.fn();
			const cache = new LRUCache<string>({ maxSize: 3, onEvict });

			cache.set('key1', 'value1');
			cache.set('key2', 'value2');
			cache.set('key3', 'value3');
			cache.clear();

			expect(onEvict).toHaveBeenCalledTimes(3);
			expect(onEvict).toHaveBeenCalledWith('key1', 'value1');
			expect(onEvict).toHaveBeenCalledWith('key2', 'value2');
			expect(onEvict).toHaveBeenCalledWith('key3', 'value3');
		});

		it('should not call onEvict when overwriting existing key', () => {
			const onEvict = vi.fn();
			const cache = new LRUCache<string>({ maxSize: 3, onEvict });

			cache.set('key1', 'value1');
			cache.set('key1', 'value1-updated');

			expect(onEvict).not.toHaveBeenCalled();
		});
	});

	describe('Edge cases', () => {
		it('should handle cache with size 1', () => {
			const cache = new LRUCache<string>({ maxSize: 1 });

			cache.set('key1', 'value1');
			expect(cache.size).toBe(1);

			cache.set('key2', 'value2');
			expect(cache.size).toBe(1);
			expect(cache.has('key1')).toBe(false);
			expect(cache.has('key2')).toBe(true);
		});

		it('should return keys in LRU order', () => {
			const cache = new LRUCache<string>({ maxSize: 3 });

			cache.set('key1', 'value1');
			cache.set('key2', 'value2');
			cache.set('key3', 'value3');

			// Access key1 to move it to end
			cache.get('key1');

			const keys = cache.keys();
			expect(keys).toEqual(['key2', 'key3', 'key1']);
		});

		it('should handle deletion of non-existent key', () => {
			const onEvict = vi.fn();
			const cache = new LRUCache<string>({ maxSize: 3, onEvict });

			const result = cache.delete('nonexistent');

			expect(result).toBeUndefined();
			expect(onEvict).not.toHaveBeenCalled();
		});
	});

	describe('Complex value types', () => {
		it('should work with object values', () => {
			interface TestObject {
				id: number;
				name: string;
			}

			const cache = new LRUCache<TestObject>({ maxSize: 2 });
			const obj1 = { id: 1, name: 'Object 1' };
			const obj2 = { id: 2, name: 'Object 2' };

			cache.set('obj1', obj1);
			cache.set('obj2', obj2);

			expect(cache.get('obj1')).toBe(obj1);
			expect(cache.get('obj2')).toBe(obj2);
		});

		it('should call eviction callback with correct object reference', () => {
			interface TestObject {
				dispose: () => void;
			}

			const onEvict = vi.fn();
			const cache = new LRUCache<TestObject>({ maxSize: 2, onEvict });

			const obj1 = { dispose: vi.fn() };
			const obj2 = { dispose: vi.fn() };
			const obj3 = { dispose: vi.fn() };

			cache.set('obj1', obj1);
			cache.set('obj2', obj2);
			cache.set('obj3', obj3); // Should evict obj1

			expect(onEvict).toHaveBeenCalledWith('obj1', obj1);
		});
	});
});

