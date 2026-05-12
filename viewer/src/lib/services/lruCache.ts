/**
 * Generic LRU (Least Recently Used) Cache Implementation
 * 
 * Provides a cache with a maximum capacity. When the cache is full, the least
 * recently used item is evicted. Supports optional disposal callbacks for cleanup.
 */

export interface LRUCacheOptions<V> {
	/** Maximum number of items to store */
	maxSize: number;
	/** Optional callback when an item is evicted */
	onEvict?: (key: string, value: V) => void;
}

export class LRUCache<V> {
	private maxSize: number;
	private cache: Map<string, V>;
	private onEvict?: (key: string, value: V) => void;

	constructor(options: LRUCacheOptions<V>) {
		this.maxSize = options.maxSize;
		this.onEvict = options.onEvict;
		this.cache = new Map();
	}

	/**
	 * Get a value from the cache
	 * Accessing an item marks it as recently used
	 */
	get(key: string): V | undefined {
		const value = this.cache.get(key);
		if (value !== undefined) {
			// Move to end (most recently used)
			this.cache.delete(key);
			this.cache.set(key, value);
		}
		return value;
	}

	/**
	 * Set a value in the cache
	 * If cache is full, evicts least recently used item
	 */
	set(key: string, value: V): void {
		// If key already exists, delete it first (will re-add at end)
		if (this.cache.has(key)) {
			this.cache.delete(key);
		}
		// If at capacity, evict least recently used (first item)
		else if (this.cache.size >= this.maxSize) {
			const firstKey = this.cache.keys().next().value;
			if (firstKey !== undefined) {
				const evictedValue = this.cache.get(firstKey)!;
				this.cache.delete(firstKey);

				// Call eviction callback if provided
				if (this.onEvict) {
					this.onEvict(firstKey, evictedValue);
				}
			}
		}

		// Add new item at end (most recently used)
		this.cache.set(key, value);
	}

	/**
	 * Check if a key exists in the cache (without affecting LRU order)
	 */
	has(key: string): boolean {
		return this.cache.has(key);
	}

	/**
	 * Remove a specific key from the cache
	 * Returns the removed value if it existed
	 */
	delete(key: string): V | undefined {
		const value = this.cache.get(key);
		if (value !== undefined && this.onEvict) {
			this.onEvict(key, value);
		}
		this.cache.delete(key);
		return value;
	}

	/**
	 * Clear all items from the cache
	 * Calls eviction callback for each item if provided
	 */
	clear(): void {
		if (this.onEvict) {
			this.cache.forEach((value, key) => {
				this.onEvict!(key, value);
			});
		}
		this.cache.clear();
	}

	/**
	 * Get current cache size
	 */
	get size(): number {
		return this.cache.size;
	}

	/**
	 * Get all keys in the cache (in LRU order, oldest first)
	 */
	keys(): string[] {
		return Array.from(this.cache.keys());
	}
}

