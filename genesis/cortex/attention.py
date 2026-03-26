"""
Genesis Mind — Attention & Salience Filter

The human brain cannot process everything equally. Attention is a
bottleneck that FORCES prioritization:

    1. BOTTOM-UP (stimulus-driven): Novel, loud, moving, or
       emotionally charged stimuli grab attention involuntarily.

    2. TOP-DOWN (goal-driven): Current drives and goals bias
       attention toward relevant stimuli.

    3. HABITUATION: Repeated identical stimuli lose potency.
       The 100th presentation of "apple" is boring.

Without attention, everything is noise. With it, the brain
operates on a bandwidth budget — like a real organism.
"""

import logging
import time
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("genesis.cortex.attention")


@dataclass
class AttentionResult:
    """Result of the attention/salience computation."""
    salience: float           # 0.0 = ignore, 1.0 = maximum attention
    should_process: bool      # Whether to run full neural cascade
    processing_depth: str     # "deep", "shallow", or "ignore"
    novelty: float            # How novel is this stimulus
    emotional_relevance: float  # How emotionally charged
    drive_relevance: float    # How relevant to current drives
    habituation_factor: float  # How habituated to this stimulus (0=fresh, 1=boring)


class AttentionSystem:
    """
    The attention bottleneck — not everything gets processed equally.

    High-salience stimuli (novel, emotional, drive-relevant) get
    deep processing through the full neural cascade.
    Low-salience stimuli get shallow processing or are ignored entirely.
    """

    def __init__(self, habituation_rate: float = 0.1,
                 habituation_recovery: float = 0.002,
                 salience_threshold: float = 0.3):
        self.habituation_rate = habituation_rate
        self.habituation_recovery = habituation_recovery
        self.salience_threshold = salience_threshold

        # Habituation map: stimulus_key -> (exposure_count, last_seen_time)
        self._exposure_history: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, 0.0))

        # Attention stats
        self._total_stimuli = 0
        self._deep_processed = 0
        self._shallow_processed = 0
        self._ignored = 0

        logger.info("Attention system initialized (threshold=%.2f, habituation_rate=%.2f)",
                     salience_threshold, habituation_rate)

    def compute_salience(self, stimulus_key: str,
                         novelty: float = 0.5,
                         emotional_intensity: float = 0.0,
                         drive_states: Optional[Dict] = None,
                         stimulus_category: str = "general") -> AttentionResult:
        """
        Compute salience score for an incoming stimulus.

        Args:
            stimulus_key: Unique identifier for this stimulus
            novelty: How novel (0=familiar, 1=completely new)
            emotional_intensity: Emotional charge (0=neutral, 1=extreme)
            drive_states: Current drive levels (curiosity, social, etc.)
            stimulus_category: Category for drive relevance matching
        """
        self._total_stimuli += 1

        # 1. Habituation — repeated exposure reduces salience
        count, last_time = self._exposure_history[stimulus_key]
        time_since = time.time() - last_time if last_time > 0 else float('inf')

        # Habituation builds with exposure, recovers with time
        raw_habituation = 1.0 - math.exp(-self.habituation_rate * count)
        # Recovery: habituation fades when stimulus is absent
        recovery = 1.0 - math.exp(-self.habituation_recovery * time_since)
        habituation = raw_habituation * (1.0 - recovery)
        habituation = max(0.0, min(1.0, habituation))

        # Update exposure history
        self._exposure_history[stimulus_key] = (count + 1, time.time())

        # 2. Bottom-up salience (stimulus-driven)
        # Novel and emotional stimuli are inherently salient
        bottom_up = (novelty * 0.6 + emotional_intensity * 0.4)

        # 3. Top-down salience (goal-driven)
        drive_relevance = 0.0
        if drive_states:
            # High drives make related stimuli more salient
            for drive_name, drive_info in drive_states.items():
                if isinstance(drive_info, dict) and 'level' in drive_info:
                    level = drive_info['level']
                    drive_relevance = max(drive_relevance, level)
                elif isinstance(drive_info, (int, float)):
                    try:
                        drive_relevance = max(drive_relevance, float(drive_info))
                    except (ValueError, TypeError):
                        pass

        top_down = drive_relevance * 0.5

        # 4. Combined salience (attenuated by habituation)
        raw_salience = bottom_up * 0.5 + top_down * 0.3 + novelty * 0.2
        salience = raw_salience * (1.0 - habituation * 0.8)  # Habituation reduces but never fully blocks

        # Emotional stimuli break through habituation
        if emotional_intensity > 0.7:
            salience = max(salience, emotional_intensity * 0.8)

        salience = max(0.0, min(1.0, salience))

        # Determine processing depth
        if salience >= 0.6:
            depth = "deep"
            should_process = True
            self._deep_processed += 1
        elif salience >= self.salience_threshold:
            depth = "shallow"
            should_process = True
            self._shallow_processed += 1
        else:
            depth = "ignore"
            should_process = False
            self._ignored += 1

        return AttentionResult(
            salience=salience,
            should_process=should_process,
            processing_depth=depth,
            novelty=novelty,
            emotional_relevance=emotional_intensity,
            drive_relevance=drive_relevance,
            habituation_factor=habituation,
        )

    def get_habituation(self, stimulus_key: str) -> float:
        """Get current habituation level for a stimulus."""
        count, last_time = self._exposure_history[stimulus_key]
        time_since = time.time() - last_time if last_time > 0 else float('inf')
        raw = 1.0 - math.exp(-self.habituation_rate * count)
        recovery = 1.0 - math.exp(-self.habituation_recovery * time_since)
        return max(0.0, raw * (1.0 - recovery))

    def reset_habituation(self, stimulus_key: str):
        """Reset habituation for a specific stimulus."""
        if stimulus_key in self._exposure_history:
            del self._exposure_history[stimulus_key]

    def get_stats(self) -> Dict:
        return {
            "total_stimuli": self._total_stimuli,
            "deep_processed": self._deep_processed,
            "shallow_processed": self._shallow_processed,
            "ignored": self._ignored,
            "unique_stimuli_tracked": len(self._exposure_history),
            "deep_pct": f"{self._deep_processed / max(1, self._total_stimuli) * 100:.0f}%",
        }

    def __repr__(self) -> str:
        return f"AttentionSystem(stimuli={self._total_stimuli}, deep={self._deep_processed}, ignored={self._ignored})"
