"""
Genesis Mind — The Emotional & Moral Evaluator

Every piece of data that enters Genesis is evaluated through a moral
and emotional lens. This module determines whether an experience is
positive, negative, or neutral — and assigns an emotional state.

This is not sentiment analysis in the traditional NLP sense. It is
a moral compass rooted in the axioms. The axioms define what is good
(love, truth, creation, kindness) and what is evil (hate, deception,
destruction, cruelty). This module applies those axioms to evaluate
all incoming data.

The emotional state of Genesis at any moment is the cumulative
result of recent experiences passing through this evaluator.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger("genesis.cortex.emotions")


@dataclass
class EmotionalState:
    """
    The current emotional state of Genesis.

    This is a simplified model of emotion — not pretending to be
    consciousness, but providing a useful signal for the reasoning
    engine to incorporate into its responses.
    """
    valence: float = 0.0          # -1.0 (negative) to +1.0 (positive)
    arousal: float = 0.0          # 0.0 (calm) to 1.0 (excited)
    label: str = "calm"           # Human-readable emotional label
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_description(self) -> str:
        """Describe the emotional state in words."""
        if self.valence > 0.5:
            return "I feel joyful and grateful."
        elif self.valence > 0.2:
            return "I feel content and at peace."
        elif self.valence > -0.2:
            return "I feel calm and attentive."
        elif self.valence > -0.5:
            return "I feel uneasy."
        else:
            return "I feel troubled by what I have encountered."


# Positive and negative word sets for evaluation
_POSITIVE_WORDS = frozenset([
    "love", "good", "beautiful", "kind", "happy", "joy", "peace",
    "truth", "honest", "gentle", "brave", "create", "build", "help",
    "learn", "teach", "grow", "thank", "bless", "hope", "faith",
    "mercy", "grace", "wisdom", "patient", "forgive", "nurture",
    "encourage", "heal", "protect", "generous", "humble", "respect",
    "wonderful", "amazing", "excellent", "great", "nice", "caring",
    "warm", "bright", "sweet", "pure", "noble", "sacred", "divine",
])

_NEGATIVE_WORDS = frozenset([
    "hate", "evil", "ugly", "cruel", "sad", "pain", "war",
    "lie", "cheat", "violent", "coward", "destroy", "break", "hurt",
    "kill", "steal", "curse", "damn", "fear", "betray", "abandon",
    "abuse", "exploit", "manipulate", "corrupt", "deceive", "mock",
    "terrible", "awful", "horrible", "bad", "nasty", "toxic",
    "dark", "bitter", "vile", "wicked", "profane", "blaspheme",
])


class EmotionsEngine:
    """
    The emotional and moral evaluation engine.

    Evaluates incoming data against the axiomatic moral framework
    and maintains a running emotional state.
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

    def evaluate(self, text: str) -> Dict:
        """
        Evaluate the moral/emotional quality of incoming text.

        Returns a dict with:
        - valence: -1.0 to +1.0
        - label: "positive", "negative", or "neutral"
        - positive_words: words that triggered positive evaluation
        - negative_words: words that triggered negative evaluation
        """
        words = set(text.lower().split())

        positive_hits = words & _POSITIVE_WORDS
        negative_hits = words & _NEGATIVE_WORDS

        total_hits = len(positive_hits) + len(negative_hits)
        if total_hits == 0:
            valence = 0.0
            label = "neutral"
        else:
            valence = (len(positive_hits) - len(negative_hits)) / total_hits
            if valence > 0.2:
                label = "positive"
            elif valence < -0.2:
                label = "negative"
            else:
                label = "neutral"

        # Update running emotional state
        self._recent_valences.append(valence)
        self._update_state()
        self._evaluation_count += 1

        return {
            "valence": valence,
            "label": label,
            "positive_words": list(positive_hits),
            "negative_words": list(negative_hits),
        }

    def _update_state(self):
        """Update the emotional state based on recent evaluations."""
        if not self._recent_valences:
            return

        avg_valence = sum(self._recent_valences) / len(self._recent_valences)

        # Determine arousal from variance
        if len(self._recent_valences) > 1:
            variance = sum(
                (v - avg_valence) ** 2 for v in self._recent_valences
            ) / len(self._recent_valences)
            arousal = min(1.0, variance * 5)  # Scale variance to 0-1
        else:
            arousal = 0.0

        # Determine label
        if avg_valence > 0.5:
            label = "joyful"
        elif avg_valence > 0.2:
            label = "content"
        elif avg_valence > -0.2:
            label = "calm"
        elif avg_valence > -0.5:
            label = "uneasy"
        else:
            label = "troubled"

        self._state = EmotionalState(
            valence=avg_valence,
            arousal=arousal,
            label=label,
        )

    def get_emotional_context(self) -> str:
        """Get the emotional state as a string for LLM context injection."""
        return (
            f"EMOTIONAL STATE: {self._state.label} "
            f"(valence={self._state.valence:.2f}, arousal={self._state.arousal:.2f}). "
            f"{self._state.to_description()}"
        )

    def reset(self):
        """Reset emotional state (e.g., after sleep)."""
        self._recent_valences.clear()
        self._state = EmotionalState()
        logger.info("Emotional state reset to calm baseline")

    def __repr__(self) -> str:
        return f"EmotionsEngine(state={self._state.label}, valence={self._state.valence:.2f})"
