"""
Genesis Mind — Metacognition (Thinking About Thinking)

Metacognition is the ability to monitor your own cognitive processes:
    - "I don't understand this" (gap detection)
    - "I'm confident about this" (confidence calibration)
    - "I should focus on this" (learning strategy)
    - "This is hard" (difficulty assessment)

Without metacognition, a learner doesn't know WHAT to learn.
With it, learning becomes targeted and efficient.
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("genesis.cortex.metacognition")


class Metacognition:
    """
    Self-monitoring of cognitive processes.

    Tracks what Genesis knows well, what it's uncertain about,
    and what it doesn't know at all. Drives targeted learning.
    """

    def __init__(self):
        # Confidence map: concept -> confidence score
        self._confidence: Dict[str, float] = defaultdict(float)
        # Retrieval success rate: concept -> (successes, attempts)
        self._retrieval_history: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
        # Knowledge gaps: things we've been asked about but couldn't answer
        self._knowledge_gaps: Dict[str, int] = defaultdict(int)
        # Learning difficulty: concept -> how many attempts needed
        self._learning_difficulty: Dict[str, int] = defaultdict(int)
        # Overall calibration error (how accurate our confidence is)
        self._calibration_errors: List[float] = []

        logger.info("Metacognition initialized — self-monitoring active")

    def on_learn(self, concept: str, success: bool = True, attempts: int = 1):
        """Record a learning event."""
        if success:
            # Confidence grows with successful learning
            self._confidence[concept] = min(1.0,
                self._confidence[concept] + 0.2 / math.sqrt(attempts))
        self._learning_difficulty[concept] += attempts

    def on_recall_attempt(self, concept: str, success: bool):
        """Record a recall attempt — did we remember correctly?"""
        successes, total = self._retrieval_history[concept]
        if success:
            self._retrieval_history[concept] = (successes + 1, total + 1)
            # Successful recall boosts confidence
            self._confidence[concept] = min(1.0,
                self._confidence[concept] + 0.05)
        else:
            self._retrieval_history[concept] = (successes, total + 1)
            # Failed recall hurts confidence
            self._confidence[concept] = max(0.0,
                self._confidence[concept] - 0.1)

    def on_question_failed(self, topic: str):
        """Record that we couldn't answer a question about a topic."""
        self._knowledge_gaps[topic] += 1

    def get_confidence(self, concept: str) -> float:
        """Get confidence level for a specific concept."""
        return self._confidence.get(concept, 0.0)

    def get_weakest_concepts(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the N concepts we're least confident about."""
        if not self._confidence:
            return []
        sorted_concepts = sorted(self._confidence.items(), key=lambda x: x[1])
        return sorted_concepts[:n]

    def get_knowledge_gaps(self) -> List[str]:
        """Get topics we know we don't know about."""
        return sorted(self._knowledge_gaps.keys(),
                       key=lambda k: self._knowledge_gaps[k], reverse=True)

    def get_learning_strategy(self) -> str:
        """Suggest what to focus on next based on metacognitive state."""
        gaps = self.get_knowledge_gaps()
        weak = self.get_weakest_concepts(3)

        if gaps:
            return f"I should learn about: {gaps[0]}"
        elif weak:
            return f"I should practice recalling: {weak[0][0]} (confidence: {weak[0][1]:.0%})"
        else:
            return "I should explore something new"

    def get_overall_confidence(self) -> float:
        """Get average confidence across all known concepts."""
        if not self._confidence:
            return 0.0
        return float(np.mean(list(self._confidence.values())))

    def get_stats(self) -> Dict:
        return {
            "concepts_tracked": len(self._confidence),
            "avg_confidence": round(self.get_overall_confidence(), 3),
            "knowledge_gaps": len(self._knowledge_gaps),
            "top_gaps": self.get_knowledge_gaps()[:3],
            "weakest": [(k, round(v, 2)) for k, v in self.get_weakest_concepts(3)],
            "strategy": self.get_learning_strategy(),
        }
