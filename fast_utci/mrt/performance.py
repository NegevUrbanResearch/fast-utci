"""
Performance optimization utilities for MRT calculations.

Provides tools for optimizing batch sizes, memory management, and computational
efficiency based on system resources and workload characteristics.
"""

import numpy as np
from typing import Optional
from .config import DEFAULT_BATCH_SIZE


class PerformanceOptimizer:
    """
    Performance optimization utilities for ray tracing and MRT calculations.
    
    Dynamically calculates optimal batch sizes based on available memory,
    ray types, and system resources to balance performance and memory safety.
    """
    
    def __init__(self, 
                 memory_fraction: float = 0.25,
                 min_solar_batch: int = 100,
                 min_sky_batch: int = 500):
        """
        Initialize performance optimizer.
        
        Args:
            memory_fraction: Fraction of available memory to use for ray processing (0-1)
            min_solar_batch: Minimum batch size for solar rays
            min_sky_batch: Minimum batch size for sky rays
        """
        self.memory_fraction = memory_fraction
        self.min_solar_batch = min_solar_batch
        self.min_sky_batch = min_sky_batch
    
    def calculate_batch_size(self, 
                            n_rays: int, 
                            ray_type: str = "mixed",
                            available_memory: Optional[int] = None) -> int:
        """
        Calculate optimal batch size based on ray count, type, and available memory.
        
        Args:
            n_rays: Number of rays to process
            ray_type: Type of rays ("sky", "solar", or "mixed")
            available_memory: Override available memory in bytes (None = auto-detect)
            
        Returns:
            Optimal batch size that balances performance and memory safety
        """
        # Base batch sizes optimized for different ray types
        if ray_type == "sky":
            # Sky rays: typically 145 rays per position, can use larger batches
            base_batch_size = 20000
            min_batch_size = self.min_sky_batch
        elif ray_type == "solar":
            # Solar rays: typically 1 ray per position, use smaller batches
            base_batch_size = 5000
            min_batch_size = self.min_solar_batch
        else:
            # Mixed or unknown: use default
            base_batch_size = DEFAULT_BATCH_SIZE
            min_batch_size = min(self.min_solar_batch, self.min_sky_batch)
        
        try:
            # Get available memory
            if available_memory is None:
                available_memory = self.get_memory_info()
            
            # Estimate memory needed per ray
            # 3 floats for origin + 3 floats for direction + 1 bool for result = 7 * 8 bytes
            memory_per_ray = 7 * 8  # 56 bytes per ray
            
            # Use configured fraction of available memory for ray processing
            max_memory_for_rays = available_memory * self.memory_fraction
            
            # Calculate maximum safe batch size
            max_safe_batch = int(max_memory_for_rays / memory_per_ray)
            
            # Use the smaller of: base_batch_size, max_safe_batch, or n_rays
            optimal_batch_size = min(base_batch_size, max_safe_batch, n_rays)
            
            # Ensure minimum batch size for efficiency
            optimal_batch_size = max(min_batch_size, optimal_batch_size)
            
            return optimal_batch_size
            
        except Exception:
            # Fallback to conservative batch size if calculation fails
            return min(base_batch_size, n_rays, 5000)
    
    def get_memory_info(self) -> int:
        """
        Get available system memory in bytes.
        
        Returns:
            Available memory in bytes
        """
        try:
            import psutil
            return psutil.virtual_memory().available
        except Exception:
            # Fallback: assume 2GB available if psutil fails
            return 2 * 1024 * 1024 * 1024


# Global instance for convenience
_default_optimizer = PerformanceOptimizer()


def get_optimal_batch_size(n_rays: int, ray_type: str = "mixed") -> int:
    """
    Calculate optimal batch size (convenience function).
    
    Args:
        n_rays: Number of rays to process
        ray_type: Type of rays ("sky", "solar", or "mixed")
        
    Returns:
        Optimal batch size
    """
    return _default_optimizer.calculate_batch_size(n_rays, ray_type)

