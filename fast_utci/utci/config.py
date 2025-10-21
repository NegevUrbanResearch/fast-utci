"""
Configuration parameters for UTCI calculations.

Centralized configuration for UTCI-specific parameters. Shared configuration
(parallel processing, performance) is imported from fast_utci.shared.
"""

from dataclasses import dataclass, field
from typing import Optional

from fast_utci.shared.config import ParallelConfig, get_bool_env


@dataclass
class UTCIConfig:
    """
    Configuration parameters for UTCI calculations.
    
    This class contains UTCI-specific calculation parameters. Shared settings
    like parallel processing are in separate config objects for reusability.
    """
    
    # Calculation settings
    enable_vectorized: bool = True  # Use vectorized UTCI calculations when possible
    
    # Result serialization (what to include in output)
    include_weather_in_results: bool = True  # Include air_temp, wind_speed, RH in results
    include_datetime_in_results: bool = True  # Include datetime information in results
    
    # Shared configuration (composition pattern)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    
    # File I/O parameters
    csv_encoding: str = 'utf-8'
    csv_index: bool = False  # Whether to include row indices in CSV exports
    
    # Backward compatibility properties
    @property
    def n_workers(self) -> Optional[int]:
        """Backward compatibility for n_workers access."""
        return self.parallel.n_workers
    
    @property
    def show_progress(self) -> bool:
        """Backward compatibility for show_progress access."""
        return self.parallel.show_progress
    
    @property
    def parallel_threshold(self) -> int:
        """Backward compatibility for parallel_threshold access."""
        return self.parallel.parallel_threshold
    
    @classmethod
    def from_environment(cls) -> 'UTCIConfig':
        """
        Create UTCIConfig from environment variables.
        
        Returns:
            UTCIConfig instance with settings from environment
        """
        return cls(
            enable_vectorized=get_bool_env("FAST_UTCI_VECTORIZED_UTCI", True),
            include_weather_in_results=get_bool_env("FAST_UTCI_INCLUDE_WEATHER_IN_RESULTS", True),
            include_datetime_in_results=get_bool_env("FAST_UTCI_INCLUDE_DATETIME_IN_RESULTS", True),
        )


# Default configuration instance
DEFAULT_CONFIG = UTCIConfig()

# Backward compatibility constants
CSV_ENCODING = DEFAULT_CONFIG.csv_encoding
CSV_INDEX = DEFAULT_CONFIG.csv_index
DEFAULT_N_WORKERS = DEFAULT_CONFIG.parallel.n_workers
DEFAULT_SHOW_PROGRESS = DEFAULT_CONFIG.parallel.show_progress
DEFAULT_PARALLEL_THRESHOLD = DEFAULT_CONFIG.parallel.parallel_threshold

