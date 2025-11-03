"""
Cache management for MRT calculations.

Provides centralized caching for expensive computations like Tregenza sky vectors
that can be reused across multiple positions and calculations.
"""

import numpy as np
from typing import Tuple, Optional
import threading


class CacheManager:
    """
    Thread-safe singleton cache manager for MRT calculations.
    
    Manages global caches for expensive computations that can be reused,
    such as Tregenza sky dome vectors and weights.
    """
    
    _instance: Optional['CacheManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize cache storage."""
        self._sky_vectors: Optional[np.ndarray] = None
        self._sky_weights: Optional[np.ndarray] = None
    
    def get_sky_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cached Tregenza sky dome vectors and weights.
        
        Computes and caches the vectors on first access for efficient reuse
        across all positions in the analysis.
        
        Returns:
            Tuple of (vectors, weights) where:
            - vectors: shape (145, 3) unit vectors pointing to sky patches
            - weights: shape (145,) solid angle weights for each patch
        """
        if self._sky_vectors is None:
            with self._lock:
                if self._sky_vectors is None:
                    from .solar import get_tregenza_dome_vectors
                    vectors, weights = get_tregenza_dome_vectors()
                    
                    # Ensure optimal memory layout for better cache performance
                    self._sky_vectors = np.ascontiguousarray(vectors, dtype=np.float64)
                    self._sky_weights = np.ascontiguousarray(weights, dtype=np.float64)
        
        return self._sky_vectors, self._sky_weights
    
    def clear(self):
        """Clear all caches. Useful for testing and memory management."""
        with self._lock:
            self._sky_vectors = None
            self._sky_weights = None
    
    @classmethod
    def get_instance(cls) -> 'CacheManager':
        """Get the singleton cache manager instance."""
        return cls()


# Global convenience function for backward compatibility
def get_cached_sky_vectors() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get cached sky vectors (convenience function).
    
    Returns:
        Tuple of (vectors, weights) from the cache manager
    """
    return CacheManager.get_instance().get_sky_vectors()

