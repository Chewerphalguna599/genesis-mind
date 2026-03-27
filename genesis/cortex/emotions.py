"""
Genesis Mind — The Emotional Evaluator (Limbic-Driven)

Every piece of data that enters Genesis is evaluated through the limbic
system's neurochemical response. There are NO hardcoded word lists.

The limbic system maps raw sensory features → 4 neurochemicals:
    dopamine  → reward/pleasure signal
    cortisol  → stress/threat signal
    serotonin → stability/confidence signal
    oxytocin  → bonding/trust signal

Emotional evaluation is derived purely from these learned neurochemical
levels, not from English keyword matching. Genesis must LEARN what is
positive or negative through experience and limbic training.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
from collections import deque

logger = logging.getLogger("genesis.cortex.emotions")


@dataclass
class EmotionalState:
    """
    The current emotional state of Genesis — purely numerical.

    No English descriptions. The 4-dim neurochemical vector IS the
    emotional state. Labels are internal developer identifiers only.
    """
    valence: float = 0.0          # -1.0 (negative) to +1.0 (positive)
    arousal: float = 0.0          # 0.0 (calm) to 1.0 (excited)
    dopamine: float = 0.5         # Current dopamine level
    cortisol: float = 0.2         # Current cortisol level
    serotonin: float = 0.5        # Current serotonin level
    oxytocin: float = 0.3         # Current oxytocin level
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def as_vector(self):
        """Return the emotional state as a 6-dim numerical vector."""
        return [
            self.valence, self.arousal,
            self.dopamine, self.cortisol,
            self.serotonin, self.oxytocin,
        ]


class EmotionsEngine:
    """
    The emotional evaluation engine — powered by limbic neurochemistry.

    Instead of matching keywords against hardcoded English word lists,
    this engine derives emotional state from the limbic system's learned
    neurochemical responses to sensory input.

    The limbic system learns through supervised training: when the
    conscious mind evaluates something as positive/negative, the limbic
    network is trained to produce the appropriate neurochemical response.
    Over time, Genesis develops genuine emotional "instincts."
    """

    def __init__(self, memory_window: int = 20):
        self._state = EmotionalState()
        self._recent_valences: deque = deque(maxlen=memory_window)
        self._evaluation_count = 0

        logger.info("Emotions engine initialized")

    @property
    def current_state(self) -> EmotionalState:
        """Get the current emotional state."""
        return self._state

    def evaluate_from_limbic(self, neurochemicals: Dict[str, float]) -> Dict:
        """
        Evaluate emotional state from limbic system output.

        This is the PRIMARY evaluation method. The limbic system has already
        processed the raw sensory input and produced neurochemicals. We
        derive the emotional valence from those levels.

        Args:
            neurochemicals: Dict with dopamine, cortisol, serotonin, oxytocin (0-1)

        Returns:
            Dict with valence, arousal, and neurochemical levels
        """
        dopamine = neurochemicals.get("dopamine", 0.5)
        cortisol = neurochemicals.get("cortisol", 0.2)
        serotonin = neurochemicals.get("serotonin", 0.5)
        oxytocin = neurochemicals.get("oxytocin", 0.3)

        # Valence = (positive signals) - (negative signals)
        # Dopamine + oxytocin = positive; cortisol = negative
        valence = ((dopamine + oxytocin) / 2.0) - cortisol
        valence = max(-1.0, min(1.0, valence))

        # Arousal = how "activated" the system is (high chemicals = high arousal)
        arousal = min(1.0, (abs(dopamine - 0.5) + abs(cortisol - 0.2) +
                            abs(serotonin - 0.5) + abs(oxytocin - 0.3)) / 2.0)

        # Update running state
        self._recent_valences.append(valence)
        self._update_state(dopamine, cortisol, serotonin, oxytocin)
        self._evaluation_count += 1

        return {
            "valence": valence,
            "arousal": arousal,
            "dopamine": dopamine,
            "cortisol": cortisol,
            "serotonin": serotonin,
            "oxytocin": oxytocin,
        }

    def evaluate(self, text: str) -> Dict:
        """
        Backward-compatible evaluate method.

        Since we no longer use keyword matching, this returns a neutral
        baseline. The REAL evaluation happens through evaluate_from_limbic().
        This method exists to avoid breaking code that still calls it —
        but it produces no English labels.
        """
        # No keyword matching — return neutral baseline
        self._recent_valences.append(0.0)
        self._update_state(0.5, 0.2, 0.5, 0.3)
        self._evaluation_count += 1

        return {
            "valence": 0.0,
            "arousal": 0.0,
            "dopamine": 0.5,
            "cortisol": 0.2,
            "serotonin": 0.5,
            "oxytocin": 0.3,
        }

    def _update_state(self, dopamine: float, cortisol: float,
                      serotonin: float, oxytocin: float):
        """Update the emotional state from neurochemical levels."""
        if not self._recent_valences:
            return

        avg_valence = sum(self._recent_valences) / len(self._recent_valences)

        # Arousal from variance in recent valences
        if len(self._recent_valences) > 1:
            variance = sum(
                (v - avg_valence) ** 2 for v in self._recent_valences
            ) / len(self._recent_valences)
            arousal = min(1.0, variance * 5)
        else:
            arousal = 0.0

        self._state = EmotionalState(
            valence=avg_valence,
            arousal=arousal,
            dopamine=dopamine,
            cortisol=cortisol,
            serotonin=serotonin,
            oxytocin=oxytocin,
        )

    def get_emotional_vector(self) -> list:
        """Get the emotional state as a numerical vector for neural processing."""
        return self._state.as_vector()

    def reset(self):
        """Reset emotional state (e.g., after sleep)."""
        self._recent_valences.clear()
        self._state = EmotionalState()
        logger.info("Emotional state reset to baseline")

    def __repr__(self) -> str:
        return (
            f"EmotionsEngine(valence={self._state.valence:.2f}, "
            f"arousal={self._state.arousal:.2f})"
        )
