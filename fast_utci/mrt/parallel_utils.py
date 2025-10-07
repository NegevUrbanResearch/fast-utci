"""
Parallel processing utilities for MRT calculations.

Provides reusable patterns for parallel computation with chunking strategies,
progress tracking, and worker pool management.
"""

from __future__ import annotations
import numpy as np
from typing import List, Callable, Any, Optional, Tuple
from abc import ABC, abstractmethod
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import time
import logging

logger = logging.getLogger(__name__)


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
                 n_workers: Optional[int] = None,
                 show_progress: bool = True):
        """
        Initialize parallel processor.
        
        Args:
            n_workers: Number of parallel workers (None = CPU count - 1)
            show_progress: Whether to show progress bar
        """
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        
        self.n_workers = n_workers
        self.show_progress = show_progress
    
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
        # Create chunks
        if isinstance(data, np.ndarray) and len(data.shape) > 1:
            # Spatial data
            chunks = chunk_strategy.create_chunks(data, self.n_workers)
        else:
            # Generic data - use balanced chunking
            balanced = BalancedChunkStrategy()
            chunks = balanced.create_chunks(np.array(data), self.n_workers)
        
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

