import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from kernel import page_cache_update_kernel, page_cache_retrieve_kernel


class PageAttentionCacheManager:
    """
    Page-based cache manager that manages KV cache with page attention mechanisms.
    """
    
    def __init__(
        self,
        batch_size: int,
        page_size: int,  # Size of each page
        num_pages: int,  # Total number of pages in the cache
        kv_latent_rank: int,
        qk_rope_head_dim: int,
        max_seq_len: int = 4096 * 4,
        use_triton: bool = False,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        self.batch_size = batch_size
        self.page_size = page_size
        self.num_pages = num_pages
        self.kv_latent_rank = kv_latent_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.dtype = dtype
        self.device = device
        self.use_triton = use_triton
        
        # Allocate physical cache memory as pages
        self.kv_latent_pages = torch.zeros(
            num_pages, page_size, kv_latent_rank, 
            dtype=dtype, device=device
        )
        self.k_rope_pages = torch.zeros(
            num_pages, page_size, qk_rope_head_dim, 
            dtype=dtype, device=device
        )
        
        # Initialize page tables
        # Structure: {batch_idx: {logical_page_idx: physical_page_idx}}
        if self.use_triton:
            # For triton, store the page table in tensor format for faster access
            # NOTE: max_num_logical_page is the max number of logical pages could be
            # allocated for a sequence with max_seq_len
            self.max_num_logical_page = (max_seq_len + page_size - 1) // page_size
            self.page_tables = torch.full(
                (batch_size, self.max_num_logical_page), -1,
                dtype=torch.int32, device=device
            )
        else:
            self.page_tables: Dict[int, Dict[int, int]] = {i: {} for i in range(batch_size)}
        
        # Track free pages
        self.free_pages: Set[int] = set(range(num_pages))
        
        # Track allocated pages per batch for easy cleanup
        # Structure: {batch_idx: [physical_page_idxs]}
        self.batch_to_pages: Dict[int, List[int]] = {i: [] for i in range(batch_size)}
        
        # Cache metadata
        self.max_seq_len_per_batch: Dict[int, int] = {i: 0 for i in range(batch_size)}
    
    def _logical_to_physical(self, batch_idx: int, seq_pos: int) -> Tuple[int, int]:
        """
        Convert logical sequence position to physical page and offset, for tokens not 
        in the table, allocate page and add to the table.
        
        Args:
            batch_idx: Index of the batch
            seq_pos: Position in the sequence
            
        Returns:
            Tuple of (physical_page_idx, offset_in_page)
        """
        logical_page_idx = seq_pos // self.page_size
        offset_in_page = seq_pos % self.page_size
        
        # Check if this logical page exists in the page table
        if logical_page_idx not in self.page_tables[batch_idx]:
            # Allocate a new page if needed
            if not self.free_pages:
                raise RuntimeError("Cache out of memory: no free pages available")
            
            physical_page_idx = self.free_pages.pop()
            self.page_tables[batch_idx][logical_page_idx] = physical_page_idx
            self.batch_to_pages[batch_idx].append(physical_page_idx)
        else:
            physical_page_idx = self.page_tables[batch_idx][logical_page_idx]
        
        return physical_page_idx, offset_in_page
    
    def _logical_to_physical_triton(self, batch_idx: int, seq_pos: int) -> Tuple[int, int]:
        """
        For triton, we use tensor format for page table, so the logical to physical is different. 
        """
        logical_page_idx = seq_pos // self.page_size
        offset_in_page = seq_pos % self.page_size
        
        if self.page_tables[batch_idx, logical_page_idx] == -1:
            if not self.free_pages:
                raise RuntimeError("Cache out of memory: no free pages available")
            
            physical_page_idx = self.free_pages.pop()
            self.page_tables[batch_idx, logical_page_idx] = physical_page_idx
            self.batch_to_pages[batch_idx].append(physical_page_idx)
        else:
            physical_page_idx = self.page_tables[batch_idx, logical_page_idx]

        return physical_page_idx, offset_in_page


    def update(
        self, 
        batch_idx: int, 
        start_pos: int, 
        kv_latent: torch.Tensor, 
        k_rope: torch.Tensor
    ):
        """
        Update the cache with new KV data
        
        Args:
            batch_idx: Index of the batch
            start_pos: Starting position in the sequence
            kv_latent: KV latent tensor [seq_len, kv_latent_rank]
            k_rope: K rope tensor [seq_len, qk_rope_head_dim]
        """
        seq_len = kv_latent.shape[0]
        end_pos = start_pos + seq_len
        
        # Update max sequence length if needed
        if end_pos > self.max_seq_len_per_batch[batch_idx]:
            self.max_seq_len_per_batch[batch_idx] = end_pos
        
        # Update cache page by page
        for i in range(seq_len):
            seq_pos = start_pos + i
            physical_page_idx, offset_in_page = self._logical_to_physical(batch_idx, seq_pos)
            
            # Update the cache
            self.kv_latent_pages[physical_page_idx, offset_in_page] = kv_latent[i]
            self.k_rope_pages[physical_page_idx, offset_in_page] = k_rope[i]
    
    def update_batch(
        self,
        start_pos: int,
        kv_latent: torch.Tensor,
        k_rope: torch.Tensor,
    ):
        """
        Update the cache with new KV data for a batch, this use triton kernel to speed up. 
        """
        batch_size, seq_len, kv_latent_rank = kv_latent.shape
        _, _, qk_rope_head_dim = k_rope.shape
        page_size = kv_latent.shape[1]
        end_pos = start_pos + seq_len
        for batch_idx in range(batch_size):
            if end_pos > self.max_seq_len_per_batch[batch_idx]:
                self.max_seq_len_per_batch[batch_idx] = end_pos

        # This cannot be written into the triton kernel, 
        # because the parralel allocate page will cause the page table to be inconsistent.
        for batch_idx in range(batch_size):
            for seq_pos in range(start_pos, end_pos):
                _, _ = self._logical_to_physical_triton(batch_idx, seq_pos)

        grid = lambda meta: (
            batch_size,
            seq_len,
        )
        
        page_cache_update_kernel[grid](
            kv_latent, # [B, T, K]
            k_rope, # [B, T, R]
            self.page_tables, # (batch_size, max_num_logical_pages)
            self.kv_latent_pages, 
            self.k_rope_pages,
            seq_len,
            self.max_num_logical_page,
            page_size,
            kv_latent_rank,
            qk_rope_head_dim,
            start_pos,
        )

    def retrieve(
        self, 
        batch_idx: int, 
        start_pos: int, 
        end_pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve cached KV data for a specific sequence range
        
        Args:
            batch_idx: Index of the batch
            start_pos: Starting position in the sequence
            end_pos: Ending position in the sequence (exclusive)
            
        Returns:
            Tuple of (kv_latent, k_rope) tensors
        """
        seq_len = end_pos - start_pos
        
        # Initialize output tensors
        kv_latent = torch.zeros(
            seq_len, self.kv_latent_rank, 
            dtype=self.dtype, device=self.device
        )
        k_rope = torch.zeros(
            seq_len, self.qk_rope_head_dim, 
            dtype=self.dtype, device=self.device
        )
        
        # Gather data from pages
        for i in range(seq_len):
            seq_pos = start_pos + i
            logical_page_idx = seq_pos // self.page_size
            offset_in_page = seq_pos % self.page_size
            
            # Check if this logical page exists in the page table
            if logical_page_idx in self.page_tables[batch_idx]:
                physical_page_idx = self.page_tables[batch_idx][logical_page_idx]
                
                # Copy data from cache
                kv_latent[i] = self.kv_latent_pages[physical_page_idx, offset_in_page]
                k_rope[i] = self.k_rope_pages[physical_page_idx, offset_in_page]
        
        return kv_latent, k_rope

    def retrieve_batch(
        self,
        batch_size: int,
        start_pos: int,
        end_pos: int,
    ):
        seq_len = end_pos - start_pos

        # Initialize output tensors
        kv_latent = torch.zeros(
            batch_size, seq_len, self.kv_latent_rank, 
            dtype=self.dtype, device=self.device
        )
        k_rope = torch.zeros(
            batch_size, seq_len, self.qk_rope_head_dim, 
            dtype=self.dtype, device=self.device
        )

        grid = lambda meta: (
            batch_size,
            seq_len,
        )

        page_cache_retrieve_kernel[grid](
            self.kv_latent_pages,
            self.k_rope_pages,
            self.page_tables,
            kv_latent,
            k_rope,
            seq_len,
            self.max_num_logical_page,
            self.page_size,
            self.kv_latent_rank,
            self.qk_rope_head_dim,
            start_pos
        )
        
        return kv_latent, k_rope
    
    def clear_batch(self, batch_idx: int):
        """
        Clear all cache entries for a specific batch
        
        Args:
            batch_idx: Index of the batch to clear
        """
        # Return pages to the free pool
        for page_idx in self.batch_to_pages[batch_idx]:
            self.free_pages.add(page_idx)
        
        # Clear batch metadata
        self.batch_to_pages[batch_idx] = []
        self.page_tables[batch_idx] = {}
        self.max_seq_len_per_batch[batch_idx] = 0
    
    def clear_all(self):
        """Clear all cache entries"""
        for batch_idx in range(self.batch_size):
            self.clear_batch(batch_idx)
    
    def get_memory_usage(self) -> Dict:
        """
        Get cache memory usage statistics
        
        Returns:
            Dictionary with memory usage stats
        """
        total_pages = self.num_pages
        used_pages = total_pages - len(self.free_pages)
        
        # Calculate actual sequence tokens stored
        actual_tokens = sum(self.max_seq_len_per_batch.values())
        
        # Calculate capacity utilization and fragmentation
        page_capacity = total_pages * self.page_size
        internal_fragmentation = 0
        
        for batch_idx, max_len in self.max_seq_len_per_batch.items():
            pages_allocated = len(self.batch_to_pages[batch_idx])
            capacity_allocated = pages_allocated * self.page_size
            internal_fragmentation += capacity_allocated - max_len
        
        return {
            "total_pages": total_pages,
            "used_pages": used_pages,
            "free_pages": len(self.free_pages),
            "usage_percentage": (used_pages / total_pages) * 100 if total_pages > 0 else 0,
            "actual_tokens": actual_tokens,
            "page_capacity": page_capacity,
            "internal_fragmentation": internal_fragmentation,
            "internal_fragmentation_percentage": 
                (internal_fragmentation / (page_capacity - len(self.free_pages) * self.page_size)) * 100 
                if used_pages > 0 else 0
        }
    
    # def optimize_pages(self) -> int:
    #     """
    #     Optimize page allocation to reduce fragmentation
        
    #     Returns:
    #         Number of pages freed
    #     """
    #     # This is a simple defragmentation strategy:
    #     # 1. Identify pages with low utilization
    #     # 2. Consolidate data from these pages
    #     # 3. Free up unused pages
        
    #     pages_freed = 0
        
    #     # Implement defragmentation here
    #     # This is a placeholder for advanced page optimization logic
        
    #     return pages_freed