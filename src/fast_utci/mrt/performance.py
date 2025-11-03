"""
Performance optimization utilities for MRT calculations.

Provides tools for optimizing batch sizes, memory management, and computational
efficiency based on system resources and workload characteristics.

Uses PerformanceConfig from shared utilities for configuration values.
"""

import numpy as np
from typing import Optional
from fast_utci.shared import MRTConfig


def _get_default_batch_size() -> int:
    """Fetch default batch size from TOML via load_config; fallback to 10000."""
    try:
        from fast_utci.shared import load_config
        return int(load_config().performance.batch_size)
    except Exception:
        return 10000


class PerformanceOptimizer:
    """
    Performance optimization utilities for ray tracing and MRT calculations.
    
    Dynamically calculates optimal batch sizes based on available memory,
    ray types, and system resources to balance performance and memory safety.
    
    Uses PerformanceConfig for configuration values (no hardcoded defaults).
    """
    
    def __init__(self, 
                 config: Optional[MRTConfig] = None,
                 memory_fraction: Optional[float] = None):
        """
        Initialize performance optimizer.
        
        Args:
            config: MRTConfig with performance settings (if None, loads from TOML)
            memory_fraction: Fraction of available memory to use (0-1). 
                          If None, uses 0.25 as default (not in config currently)
        """
        if config is None:
            try:
                from fast_utci.shared import load_config
                config = load_config().mrt
            except Exception:
                config = None
        
        self.config = config
        # Memory fraction not currently in config, use default or parameter
        self.memory_fraction = memory_fraction if memory_fraction is not None else 0.25
        # Minimum batch sizes - could be moved to config if needed
        self.min_solar_batch = 100
        self.min_sky_batch = 500
    
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
        # Use config batch_size as base, with multipliers for different ray types
        base_config_batch_size = self.config.performance.batch_size if self.config else _get_default_batch_size()
        
        if ray_type == "sky":
            # Sky rays: typically 145 rays per position, can use larger batches
            # Use 2x config batch_size for sky rays
            base_batch_size = base_config_batch_size * 2
            min_batch_size = self.min_sky_batch
        elif ray_type == "solar":
            # Solar rays: typically 1 ray per position, use smaller batches
            # Use 0.5x config batch_size for solar rays
            base_batch_size = base_config_batch_size // 2
            min_batch_size = self.min_solar_batch
        else:
            # Mixed or unknown: use configured default
            base_batch_size = base_config_batch_size
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


# Note: No global optimizer instance - must provide config when needed


def get_optimal_batch_size(n_rays: int, 
                          ray_type: str = "mixed",
                          config: Optional[MRTConfig] = None) -> int:
    """
    Calculate optimal batch size (convenience function).
    
    Args:
        n_rays: Number of rays to process
        ray_type: Type of rays ("sky", "solar", or "mixed")
        config: Optional MRTConfig (loads from TOML if not provided)
        
    Returns:
        Optimal batch size
    """
    optimizer = PerformanceOptimizer(config=config)
    return optimizer.calculate_batch_size(n_rays, ray_type)

