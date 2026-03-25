"""
Genesis Mind — Sleep & Memory Consolidation

When a human sleeps, the brain replays the day's experiences,
strengthens important memories, prunes weak ones, and forms
new associations. This is why sleep is essential for learning.

Genesis replicates this process:

    1. REPLAY:    Review all episodes from the current session
    2. STRENGTHEN: Boost frequently accessed concepts
    3. PRUNE:      Remove weak/unused memories
    4. SUMMARIZE:  Generate a daily summary narrative
    5. OPTIMIZE:   Reorganize the memory database

Sleep can be triggered manually ("Genesis, sleep") or automatically
after a configurable interval (default: every 8 hours of runtime).
"""

import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("genesis.growth.sleep")


class SleepCycle:
    """
    Memory consolidation system.

    Runs periodically to strengthen important memories,
    prune weak ones, and generate daily summaries.
    """

    def __init__(self, consolidation_strength_boost: float = 0.1,
                 pruning_threshold: float = 0.01,
                 decay_amount: float = 0.005):
        self.consolidation_strength_boost = consolidation_strength_boost
        self.pruning_threshold = pruning_threshold
        self.decay_amount = decay_amount
        self._sleep_count = 0
        self._last_sleep: Optional[str] = None

        logger.info("Sleep cycle initialized")

    def consolidate(self, semantic_memory, episodic_memory, phonetics_engine=None) -> Dict:
        """
        Run a full sleep consolidation cycle.

        This is the equivalent of a night's sleep. It processes the
        day's experiences and optimizes the memory system.

        Args:
            semantic_memory: The SemanticMemory instance
            episodic_memory: The EpisodicMemory instance
            phonetics_engine: Optional PhoneticsEngine instance

        Returns:
            A report of what happened during consolidation.
        """
        self._sleep_count += 1
        start_time = datetime.now()
        logger.info("╔══════════════════════════════════════════════╗")
        logger.info("║         ENTERING SLEEP CYCLE #%d             ║", self._sleep_count)
        logger.info("╚══════════════════════════════════════════════╝")

        report = {
            "sleep_number": self._sleep_count,
            "started_at": start_time.isoformat(),
            "concepts_before": semantic_memory.count(),
            "episodes_before": episodic_memory.count(),
        }

        # Step 1: DECAY — Apply forgetting curve to all concepts
        logger.info("Step 1/4: Applying forgetting curve...")
        semantic_memory.decay_all(amount=self.decay_amount)
        if phonetics_engine:
            phonetics_engine.decay_all(amount=self.decay_amount)

        # Step 2: REPLAY — Review recent episodes and reinforce related concepts
        logger.info("Step 2/4: Replaying recent experiences...")
        recent_episodes = episodic_memory.get_today()
        concepts_reinforced = set()

        for episode in recent_episodes:
            # Reinforce concepts that were activated during the day
            for word in episode.concepts_activated + episode.concepts_learned:
                concept = semantic_memory.recall_concept(word)
                if concept:
                    concepts_reinforced.add(word)

        report["concepts_reinforced"] = len(concepts_reinforced)
        logger.info("  Reinforced %d concepts from today's experiences", len(concepts_reinforced))

        # Step 3: PRUNE — Remove dead concepts
        logger.info("Step 3/4: Pruning weak memories...")
        pruned_count = semantic_memory.prune_dead_concepts(threshold=self.pruning_threshold)
        report["concepts_pruned"] = pruned_count

        # Step 4: SUMMARIZE — Generate daily summary
        logger.info("Step 4/4: Generating daily summary...")
        summary = episodic_memory.get_daily_summary()
        report["daily_summary"] = summary

        # Final stats
        report["concepts_after"] = semantic_memory.count()
        report["duration_sec"] = (datetime.now() - start_time).total_seconds()

        self._last_sleep = datetime.now().isoformat()

        logger.info("╔══════════════════════════════════════════════╗")
        logger.info("║         SLEEP CYCLE COMPLETE                 ║")
        logger.info("║  Concepts: %d → %d                          ║",
                     report["concepts_before"], report["concepts_after"])
        logger.info("║  Reinforced: %d | Pruned: %d                ║",
                     report["concepts_reinforced"], report["concepts_pruned"])
        logger.info("╚══════════════════════════════════════════════╝")

        return report

    @property
    def sleep_count(self) -> int:
        return self._sleep_count

    @property
    def last_sleep(self) -> Optional[str]:
        return self._last_sleep

    def __repr__(self) -> str:
        return f"SleepCycle(count={self._sleep_count}, last='{self._last_sleep}')"
