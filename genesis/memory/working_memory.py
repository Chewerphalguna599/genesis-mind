"""
Genesis Mind — Working Memory (Short-Term Buffer)

The human brain has a strict bottleneck: working memory holds ~7±2 items.
This constraint is not a flaw — it FORCES:
    - Chunking: grouping items into meaningful clusters
    - Prioritization: only the most relevant items survive
    - Abstraction: compressing detail into gist

Working memory is the gateway to long-term memory:
    1. New input enters the buffer (capacity limited)
    2. Items are rehearsed (repeated access keeps them alive)
    3. Unrehearsed items decay in ~20 seconds
    4. Sufficiently rehearsed items consolidate to long-term storage

This is fundamentally different from semantic/episodic memory:
    - Working memory: volatile, tiny, fast (~20s lifespan)
    - Short-term: survives minutes, needs consolidation
    - Long-term: persists indefinitely (but decays without retrieval)
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("genesis.memory.working_memory")

# Working memory capacity (Miller's Law: 7 ± 2)
WM_CAPACITY = 7
WM_DECAY_SECONDS = 20.0  # Items decay after 20s without rehearsal
STM_DECAY_SECONDS = 300.0  # Short-term: 5 minutes without consolidation


@dataclass
class WorkingMemoryItem:
    """A single item in working memory."""
    key: str
    content: Any
    embedding: Optional[np.ndarray] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    salience: float = 0.5  # How important/attention-worthy this item is
    emotional_weight: float = 0.0  # Emotional items persist longer

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    @property
    def time_since_access(self) -> float:
        return time.time() - self.last_accessed

    @property
    def effective_decay_time(self) -> float:
        """Emotionally charged items decay slower."""
        base = WM_DECAY_SECONDS
        # Emotional items get up to 3x lifespan
        return base * (1.0 + 2.0 * abs(self.emotional_weight))

    def is_decayed(self) -> bool:
        return self.time_since_access > self.effective_decay_time


class WorkingMemory:
    """
    Capacity-limited short-term buffer.

    Like a real brain: can hold ~7 items, items decay without
    rehearsal, and only sufficiently rehearsed items move to
    long-term storage.
    """

    def __init__(self, capacity: int = WM_CAPACITY):
        self.capacity = capacity
        self._buffer: OrderedDict[str, WorkingMemoryItem] = OrderedDict()
        self._total_items_processed = 0
        self._items_consolidated = 0
        self._items_forgotten = 0
        logger.info("Working memory initialized (capacity=%d, decay=%.0fs)",
                     capacity, WM_DECAY_SECONDS)

    def attend(self, key: str, content: Any, embedding: np.ndarray = None,
               salience: float = 0.5, emotional_weight: float = 0.0) -> Optional[str]:
        """
        Attend to something — put it in working memory.

        If buffer is full, the least salient item is evicted.
        Returns the key of any evicted item (or None).
        """
        evicted = None

        # If already in buffer, rehearse it (strengthens it)
        if key in self._buffer:
            item = self._buffer[key]
            item.last_accessed = time.time()
            item.access_count += 1
            item.salience = max(item.salience, salience)
            # Move to end (most recent)
            self._buffer.move_to_end(key)
            return None

        # First, prune decayed items
        self._prune_decayed()

        # If at capacity, evict least salient
        if len(self._buffer) >= self.capacity:
            evicted = self._evict_least_salient()

        # Add new item
        self._buffer[key] = WorkingMemoryItem(
            key=key,
            content=content,
            embedding=embedding,
            salience=salience,
            emotional_weight=emotional_weight,
        )
        self._total_items_processed += 1

        return evicted

    def rehearse(self, key: str) -> bool:
        """Rehearse an item — prevents decay, strengthens for consolidation."""
        if key not in self._buffer:
            return False
        item = self._buffer[key]
        item.last_accessed = time.time()
        item.access_count += 1
        self._buffer.move_to_end(key)
        return True

    def recall(self, key: str) -> Optional[WorkingMemoryItem]:
        """Try to recall something from working memory."""
        if key in self._buffer:
            item = self._buffer[key]
            if not item.is_decayed():
                item.last_accessed = time.time()
                item.access_count += 1
                return item
            else:
                # It decayed — forgotten
                del self._buffer[key]
                self._items_forgotten += 1
                return None
        return None

    def get_consolidation_candidates(self) -> List[WorkingMemoryItem]:
        """
        Get items ready for long-term consolidation.

        Items that have been rehearsed enough (accessed 3+ times)
        are candidates for transfer to long-term memory.
        """
        self._prune_decayed()
        candidates = []
        for item in self._buffer.values():
            if item.access_count >= 3 or item.emotional_weight > 0.5:
                candidates.append(item)
        return candidates

    def get_active_items(self) -> List[WorkingMemoryItem]:
        """Get all non-decayed items currently in working memory."""
        self._prune_decayed()
        return list(self._buffer.values())

    def _prune_decayed(self):
        """Remove items that have decayed from working memory."""
        to_remove = [k for k, v in self._buffer.items() if v.is_decayed()]
        for k in to_remove:
            del self._buffer[k]
            self._items_forgotten += 1

    def _evict_least_salient(self) -> Optional[str]:
        """Evict the least important item to make room."""
        if not self._buffer:
            return None
        # Find item with lowest salience × recency score
        worst_key = min(
            self._buffer.keys(),
            key=lambda k: self._buffer[k].salience * (1.0 / (1.0 + self._buffer[k].time_since_access))
        )
        evicted_key = worst_key
        del self._buffer[worst_key]
        self._items_forgotten += 1
        return evicted_key

    def get_stats(self) -> Dict:
        self._prune_decayed()
        return {
            "current_items": len(self._buffer),
            "capacity": self.capacity,
            "utilization": f"{len(self._buffer)}/{self.capacity}",
            "total_processed": self._total_items_processed,
            "consolidated": self._items_consolidated,
            "forgotten": self._items_forgotten,
        }

    def __len__(self) -> int:
        self._prune_decayed()
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"WorkingMemory({len(self)}/{self.capacity})"
