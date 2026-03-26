"""
Genesis Mind — Theory of Mind

By age 4, human children develop Theory of Mind (ToM):
the ability to understand that OTHER people have beliefs,
desires, and knowledge different from their own.

Genesis models the user by tracking:
    1. What the user has taught (knowledge state)
    2. What the user seems interested in (goals)
    3. How the user communicates (style)
    4. Interaction patterns (frequency, patience)
    5. Emotional patterns (detected from input)
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("genesis.cortex.theory_of_mind")


@dataclass
class UserModel:
    """A model of the user's mental state."""
    # What the user has taught
    topics_taught: List[str] = field(default_factory=list)
    # Estimated user knowledge areas
    knowledge_areas: Dict[str, float] = field(default_factory=dict)
    # User emotional tone history
    sentiment_history: List[float] = field(default_factory=list)
    # Interaction stats
    total_interactions: int = 0
    avg_message_length: float = 0.0
    patience_score: float = 0.5  # 0=impatient, 1=very patient
    # Timing
    last_interaction_time: float = 0.0
    interaction_frequency: float = 0.0  # interactions per minute


class TheoryOfMind:
    """
    Models other minds — tracks what the user knows, wants, and feels.

    Only unlocked at Phase 3+ (Child). Younger phases cannot
    model other minds — they are purely egocentric.
    """

    def __init__(self):
        self._user_model = UserModel()
        self._interaction_times: List[float] = []
        self._topic_frequency: Dict[str, int] = defaultdict(int)
        self._enabled = False
        logger.info("Theory of Mind initialized (dormant until Phase 3)")

    def enable(self):
        """Enable ToM (triggered at Phase 3)."""
        self._enabled = True
        logger.info("Theory of Mind ACTIVATED — can now model other minds")

    @property
    def is_active(self) -> bool:
        return self._enabled

    def observe_interaction(self, user_input: str, topic: str = "general",
                            sentiment: float = 0.0):
        """
        Observe a user interaction and update the model.

        Args:
            user_input: What the user said
            topic: What topic this relates to
            sentiment: Detected sentiment (-1 to +1)
        """
        if not self._enabled:
            return

        now = time.time()
        model = self._user_model

        model.total_interactions += 1
        model.last_interaction_time = now
        self._interaction_times.append(now)

        # Track message length patterns
        length = len(user_input.split())
        model.avg_message_length = (
            model.avg_message_length * 0.9 + length * 0.1
        )

        # Track sentiment
        model.sentiment_history.append(sentiment)
        if len(model.sentiment_history) > 50:
            model.sentiment_history = model.sentiment_history[-50:]

        # Track topics
        if topic != "general":
            model.topics_taught.append(topic)
            self._topic_frequency[topic] += 1
            model.knowledge_areas[topic] = min(1.0,
                model.knowledge_areas.get(topic, 0) + 0.1)

        # Estimate patience from interaction timing
        if len(self._interaction_times) >= 2:
            gaps = [self._interaction_times[i] - self._interaction_times[i-1]
                    for i in range(1, len(self._interaction_times[-10:]))]
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                # Very fast interactions = impatient, slow = patient
                model.patience_score = min(1.0, avg_gap / 30.0)
                model.interaction_frequency = 60.0 / max(0.1, avg_gap)

    def predict_user_interest(self) -> Optional[str]:
        """Predict what topic the user is most interested in."""
        if not self._enabled or not self._topic_frequency:
            return None
        return max(self._topic_frequency.items(), key=lambda x: x[1])[0]

    def estimate_user_sentiment(self) -> float:
        """Estimate the user's current emotional state."""
        if not self._enabled or not self._user_model.sentiment_history:
            return 0.0
        # Weighted average, recent more important
        recent = self._user_model.sentiment_history[-5:]
        weights = [0.1, 0.15, 0.2, 0.25, 0.3][:len(recent)]
        return sum(s * w for s, w in zip(recent, weights)) / sum(weights)

    def what_user_knows(self) -> List[str]:
        """Return list of topics the user has taught (user's perceived knowledge)."""
        if not self._enabled:
            return []
        return list(set(self._user_model.topics_taught))

    def what_user_doesnt_know_i_know(self, my_concepts: List[str]) -> List[str]:
        """Find concepts I know but the user didn't teach (learned from dreams/associations)."""
        if not self._enabled:
            return []
        taught = set(self._user_model.topics_taught)
        return [c for c in my_concepts if c not in taught]

    def get_status(self) -> Dict:
        if not self._enabled:
            return {"active": False, "reason": "Requires Phase 3+ (Child)"}
        m = self._user_model
        return {
            "active": True,
            "user_interactions": m.total_interactions,
            "topics_taught": len(set(m.topics_taught)),
            "user_patience": round(m.patience_score, 2),
            "user_sentiment": round(self.estimate_user_sentiment(), 2),
            "predicted_interest": self.predict_user_interest(),
            "interaction_rate": f"{m.interaction_frequency:.1f}/min",
        }
