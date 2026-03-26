"""
Genesis Mind — Play Behavior (Creative Exploration)

Play is not optional — it is how children learn:
    - Combinatorial play: "What if I mix apple + banana?"
    - Exploratory play: "What happens if I recall X near Y?"
    - Repetitive play: Rehearsing the same concept over and over
    - Social play: Turn-taking interactions

Play generates novel associations that wouldn't arise from
structured teaching alone. It is driven by the curiosity
and novelty drives.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("genesis.cortex.play")


class PlayBehavior:
    """
    Creative, exploratory learning through play.

    Combines known concepts in novel ways, tests associations,
    and rehearses recently learned material — autonomously.
    """

    def __init__(self):
        self._play_sessions = 0
        self._discoveries = 0
        self._combinations_tried: List[Tuple[str, str]] = []
        self._favorite_concepts: Dict[str, int] = {}
        logger.info("Play behavior initialized — creative exploration enabled")

    def should_play(self, curiosity_level: float, novelty_level: float,
                    concept_count: int, phase: int) -> bool:
        """Determine if conditions are right for spontaneous play."""
        if concept_count < 2:
            return False  # Need at least 2 concepts to play with
        if phase < 1:
            return False  # Newborns don't play yet

        # Play probability increases with curiosity and novelty drives
        play_urge = curiosity_level * 0.4 + novelty_level * 0.4 + random.random() * 0.2
        threshold = 0.6 - (phase * 0.05)  # Older = more likely to play
        return play_urge > threshold

    def play_combine(self, concepts: List[str],
                     get_embedding_fn, semantic_memory) -> Optional[Dict]:
        """
        Combinatorial play: pick two concepts and see how they relate.

        Returns a discovery dict if the combination is novel.
        """
        if len(concepts) < 2:
            return None

        # Pick two random concepts
        a, b = random.sample(concepts, 2)
        combo = (min(a, b), max(a, b))

        # Have we tried this combination before?
        if combo in self._combinations_tried:
            return None

        self._combinations_tried.append(combo)
        self._play_sessions += 1

        # Get embeddings and compute similarity
        try:
            emb_a = get_embedding_fn(a)
            emb_b = get_embedding_fn(b)
            if emb_a is not None and emb_b is not None:
                similarity = float(np.dot(emb_a, emb_b) /
                    (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8))
            else:
                similarity = 0.0
        except Exception:
            similarity = 0.0

        # Track favorites (concepts played with most)
        self._favorite_concepts[a] = self._favorite_concepts.get(a, 0) + 1
        self._favorite_concepts[b] = self._favorite_concepts.get(b, 0) + 1

        # Consider it a discovery if similarity is in the interesting range
        is_discovery = 0.2 < similarity < 0.8

        if is_discovery:
            self._discoveries += 1

        return {
            "concept_a": a,
            "concept_b": b,
            "similarity": similarity,
            "is_discovery": is_discovery,
            "play_type": "combinatorial",
        }

    def play_rehearse(self, concepts: List[str], semantic_memory) -> Optional[str]:
        """
        Repetitive play: rehearse a recently learned concept.

        Children repeat things over and over — this strengthens memory.
        """
        if not concepts:
            return None

        # Prefer recently learned or weak concepts
        chosen = random.choice(concepts)
        self._play_sessions += 1

        # Track as favorite
        self._favorite_concepts[chosen] = self._favorite_concepts.get(chosen, 0) + 1

        return chosen

    def get_favorite_concept(self) -> Optional[str]:
        """What does Genesis like to play with most?"""
        if not self._favorite_concepts:
            return None
        return max(self._favorite_concepts.items(), key=lambda x: x[1])[0]

    def get_stats(self) -> Dict:
        return {
            "play_sessions": self._play_sessions,
            "discoveries": self._discoveries,
            "combinations_tried": len(self._combinations_tried),
            "favorite_concept": self.get_favorite_concept(),
        }
