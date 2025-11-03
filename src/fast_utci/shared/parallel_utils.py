"""
Parallel processing utilities for MRT calculations.

Provides reusable patterns for parallel computation with chunking strategies,
progress tracking, and worker pool management.
"""

from __future__ import annotations
import numpy as np
from typing import List, Callable, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import time
import logging

logger = logging.getLogger(__name__)

# Import ParallelConfig for type hints (avoid circular import)
try:
    from .config import ParallelConfig
except ImportError:
    ParallelConfig = None  # Type: ignore


class ChunkStrategy(ABC):
    """Base class for data chunking strategies."""
    
    @abstractmethod
    def create_chunks(self, data: np.ndarray, n_workers: int) -> List[np.ndarray]:
        """
        Create data chunks for parallel processing.
        
        Args:
            data: Input data array
            n_workers: Number of parallel workers
            
        Returns:
            List of data chunks
        """
        pass


class BalancedChunkStrategy(ChunkStrategy):
    """Balanced chunking strategy - evenly distributes data across workers."""
    
    def create_chunks(self, data: np.ndarray, n_workers: int) -> List[np.ndarray]:
        """Create evenly balanced chunks."""
        n_items = len(data)
        items_per_worker = n_items // n_workers
        extra_items = n_items % n_workers
        
        chunks = []
        start_idx = 0
        
        for worker_id in range(n_workers):
            # Some workers get one extra item for better load balancing
            chunk_size = items_per_worker + (1 if worker_id < extra_items else 0)
            end_idx = start_idx + chunk_size
            
            if start_idx < n_items:
                chunks.append(data[start_idx:end_idx])
                start_idx = end_idx
        
        return chunks


class SpatialChunkStrategy(ChunkStrategy):
    """
    Spatial chunking strategy - sorts data spatially before chunking.
    
    Improves cache locality when accessing spatial data structures like BVH trees.
    """
    
    def create_chunks(self, data: np.ndarray, n_workers: int) -> List[np.ndarray]:
        """Create spatially sorted chunks for better cache locality."""
        # Sort positions spatially (X, Y, Z order)
        sorted_indices = np.lexsort((data[:, 2], data[:, 1], data[:, 0]))
        sorted_data = data[sorted_indices]
        
        # Then apply balanced chunking to sorted data
        balanced_strategy = BalancedChunkStrategy()
        return balanced_strategy.create_chunks(sorted_data, n_workers)


class ParallelProcessor:
    """
    Generic parallel processor with chunking, progress tracking, and pool management.
    
    Provides a reusable pattern for parallel computation that handles worker pool
    lifecycle, progress bars, and performance metrics.
    """
    
    def __init__(self, 
                 parallel_config: Optional['ParallelConfig'] = None,
                 n_workers: Optional[int] = None,
                 show_progress: Optional[bool] = None):
        """
        Initialize parallel processor.
        
        Args:
            parallel_config: ParallelConfig object (preferred, takes precedence)
            n_workers: Number of parallel workers (None = CPU count - 1) - deprecated, use parallel_config
            show_progress: Whether to show progress bar - deprecated, use parallel_config
        """
        if parallel_config is not None:
            self.n_workers = parallel_config.get_n_workers()
            self.show_progress = parallel_config.show_progress
        else:
            # Fallback for backward compatibility (will be removed eventually)
            if n_workers is None:
                n_workers = max(1, mp.cpu_count() - 1)
            self.n_workers = n_workers
            self.show_progress = show_progress if show_progress is not None else True
    
    def process_chunks(self,
                      data: Any,
                      worker_fn: Callable,
                      chunk_strategy: ChunkStrategy,
                      description: str = "Processing") -> List[Any]:
        """
        Process data in parallel using specified chunking strategy.
        
        Args:
            data: Input data to process
            worker_fn: Worker function that processes a chunk
            chunk_strategy: Strategy for creating chunks
            description: Description for progress bar
            
        Returns:
            List of results from all chunks
        """
        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Create chunks using the provided strategy
        chunks = chunk_strategy.create_chunks(data, self.n_workers)
        
        n_total = len(data) if hasattr(data, '__len__') else sum(len(c) for c in chunks)
        
        logger.debug(f"Processing {n_total} items with {self.n_workers} workers in {len(chunks)} chunks")
        
        # Process chunks in parallel
        results = []
        
        with Pool(processes=self.n_workers) as pool:
            if self.show_progress:
                start_time = time.time()
                
                with tqdm(total=n_total, desc=description, unit="item",
                         mininterval=1.0, maxinterval=5.0, smoothing=0.1, leave=True) as pbar:
                    
                    for chunk_result in pool.imap(worker_fn, chunks):
                        results.append(chunk_result)
                        chunk_size = len(chunk_result) if hasattr(chunk_result, '__len__') else 1
                        pbar.update(chunk_size)
                        
                        # Update description with performance metrics
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            rate = pbar.n / elapsed
                            eta = (n_total - pbar.n) / rate if rate > 0 else 0
                            pbar.set_description(f"{description} ({rate:.1f} item/s, ETA: {eta:.0f}s)")
            else:
                # No progress bar - simple processing
                results = pool.map(worker_fn, chunks)
        
        return results
    
    def process_indexed_chunks(self,
                               data: List[Tuple[int, Any]],
                               worker_fn: Callable,
                               description: str = "Processing") -> List[Any]:
        """
        Process indexed data in parallel (for maintaining order).
        
        Args:
            data: List of (index, item) tuples
            worker_fn: Worker function that processes indexed chunks
            description: Description for progress bar
            
        Returns:
            List of results from all chunks
        """
        # Create balanced chunks of indexed data
        n_items = len(data)
        items_per_worker = n_items // self.n_workers
        extra_items = n_items % self.n_workers
        
        chunks = []
        start_idx = 0
        
        for worker_id in range(self.n_workers):
            chunk_size = items_per_worker + (1 if worker_id < extra_items else 0)
            end_idx = start_idx + chunk_size
            
            if start_idx < n_items:
                chunks.append(data[start_idx:end_idx])
                start_idx = end_idx
        
        logger.debug(f"Processing {n_items} indexed items with {self.n_workers} workers in {len(chunks)} chunks")
        
        # Process chunks in parallel
        results = []
        
        with Pool(processes=self.n_workers) as pool:
            if self.show_progress:
                start_time = time.time()
                
                with tqdm(total=n_items, desc=description, unit="item",
                         mininterval=1.0, maxinterval=5.0, smoothing=0.1, leave=True) as pbar:
                    
                    for chunk_result in pool.imap(worker_fn, chunks):
                        results.append(chunk_result)
                        chunk_size = len(chunk_result)
                        pbar.update(chunk_size)
                        
                        # Update description with performance metrics
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            rate = pbar.n / elapsed
                            eta = (n_items - pbar.n) / rate if rate > 0 else 0
                            pbar.set_description(f"{description} ({rate:.1f} item/s, ETA: {eta:.0f}s)")
            else:
                # No progress bar - simple processing
                results = pool.map(worker_fn, chunks)
        
        return results
    
    def process_and_merge(self,
                          data: Any,
                          worker_fn: Callable,
                          chunk_strategy: ChunkStrategy,
                          description: str = "Processing",
                          merge_dict: bool = False,
                          merge_list: bool = False) -> Union[Dict[str, Any], List[Any], List[Dict[str, Any]]]:
        """
        Process data in parallel and automatically merge results.
        
        Args:
            data: Input data to process
            worker_fn: Worker function that processes a chunk
            chunk_strategy: Strategy for creating chunks
            description: Description for progress bar
            merge_dict: If True, merge chunk results as dictionaries (dict.update)
            merge_list: If True, merge chunk results as lists (list.extend)
            
        Returns:
            Merged results (dict if merge_dict=True, list if merge_list=True, otherwise list of chunk results)
        """
        chunk_results = self.process_chunks(data, worker_fn, chunk_strategy, description)
        
        if merge_dict:
            # Merge dictionaries
            merged = {}
            for chunk_result in chunk_results:
                if isinstance(chunk_result, dict):
                    merged.update(chunk_result)
                else:
                    # Handle case where chunk_result might be a list of dicts
                    if hasattr(chunk_result, '__iter__'):
                        for item in chunk_result:
                            if isinstance(item, dict):
                                merged.update(item)
            return merged
        elif merge_list:
            # Merge lists
            merged = []
            for chunk_result in chunk_results:
                if hasattr(chunk_result, '__iter__') and not isinstance(chunk_result, (str, bytes)):
                    merged.extend(chunk_result)
                else:
                    merged.append(chunk_result)
            return merged
        else:
            # Return raw chunk results
            return chunk_results


def create_balanced_chunks(data: List, n_workers: int) -> List[List]:
    """
    Create evenly balanced chunks (convenience function).
    
    Args:
        data: Input data list
        n_workers: Number of workers
        
    Returns:
        List of balanced chunks
    """
    strategy = BalancedChunkStrategy()
    return strategy.create_chunks(np.array(data), n_workers)


def create_spatial_chunks(positions: np.ndarray, n_workers: int) -> List[np.ndarray]:
    """
    Create spatially sorted chunks (convenience function).
    
    Args:
        positions: Position array (N, 3)
        n_workers: Number of workers
        
    Returns:
        List of spatially sorted chunks
    """
    strategy = SpatialChunkStrategy()
    return strategy.create_chunks(positions, n_workers)

