"""
Genesis Mind — Episodic Memory

The experience log. This is Genesis's autobiography — a timestamped
record of everything it has experienced.

While semantic memory stores WHAT things are (concepts), episodic
memory stores WHAT HAPPENED (events). It answers questions like:

    "What happened today?"
    "When did I first see that object?"
    "What was happening when I learned the word 'apple'?"

Each episode captures a moment in time: what was seen, what was heard,
what was felt, what was learned, and what the context was. These
episodes are the raw material from which the sleep consolidation
system extracts patterns and strengthens long-term memories.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger("genesis.memory.episodic")


@dataclass
class Episode:
    """
    A single moment of lived experience.

    Everything that happened in one perception-cognition cycle.
    """
    id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # What happened
    event_type: str = "observation"         # observation, teaching, interaction, reflection
    description: str = ""                   # Human-readable summary

    # Sensory data
    visual_description: str = ""            # What was seen
    auditory_text: str = ""                 # What was heard (transcription)
    spoken_words: List[str] = field(default_factory=list)

    # Cognitive data
    concepts_activated: List[str] = field(default_factory=list)  # Concepts recalled
    concepts_learned: List[str] = field(default_factory=list)    # New concepts created
    thought: str = ""                       # The inner voice's response

    # Emotional data
    emotional_valence: str = "neutral"      # positive, negative, neutral
    moral_assessment: str = ""

    # Developmental context
    developmental_phase: int = 0            # What phase the mind was in

    # Meta
    importance: float = 0.5                 # How important this event seems (0→1)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "description": self.description,
            "visual_description": self.visual_description,
            "auditory_text": self.auditory_text,
            "spoken_words": self.spoken_words,
            "concepts_activated": self.concepts_activated,
            "concepts_learned": self.concepts_learned,
            "thought": self.thought,
            "emotional_valence": self.emotional_valence,
            "moral_assessment": self.moral_assessment,
            "developmental_phase": self.developmental_phase,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Episode":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_narrative(self) -> str:
        """Convert this episode to a first-person narrative sentence."""
        parts = []
        if self.visual_description:
            parts.append(f"I saw: {self.visual_description}")
        if self.auditory_text:
            parts.append(f"I heard: '{self.auditory_text}'")
        if self.concepts_learned:
            parts.append(f"I learned: {', '.join(self.concepts_learned)}")
        if self.thought:
            parts.append(f"I thought: {self.thought}")
        return ". ".join(parts) if parts else self.description


class EpisodicMemory:
    """
    The experience log of Genesis.

    Stores and retrieves episodes (lived experiences) in chronological
    order. Supports querying by time range, event type, and importance.
    """

    def __init__(self, storage_path: Optional[Path] = None, max_episodes: int = 10000):
        self._episodes: List[Episode] = []
        self._storage_path = storage_path
        self._max_episodes = max_episodes
        self._episode_counter = 0
        self._load()
        logger.info("Episodic memory initialized (%d episodes)", len(self._episodes))

    def record(self, **kwargs) -> Episode:
        """
        Record a new episode.

        This is called at the end of each perception-cognition cycle
        to save what just happened.
        """
        self._episode_counter += 1
        episode = Episode(
            id=f"ep_{self._episode_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            **kwargs,
        )
        self._episodes.append(episode)

        # Cap memory size (FIFO — oldest episodes are forgotten)
        if len(self._episodes) > self._max_episodes:
            forgotten = self._episodes.pop(0)
            logger.debug("Oldest episode forgotten: %s", forgotten.id)

        self._save()
        logger.debug("Episode recorded: %s (%s)", episode.id, episode.event_type)
        return episode

    def get_recent(self, n: int = 10) -> List[Episode]:
        """Get the N most recent episodes."""
        return self._episodes[-n:]

    def get_today(self) -> List[Episode]:
        """Get all episodes from today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return [e for e in self._episodes if e.timestamp.startswith(today)]

    def get_by_type(self, event_type: str) -> List[Episode]:
        """Get all episodes of a specific type."""
        return [e for e in self._episodes if e.event_type == event_type]

    def get_important(self, min_importance: float = 0.7) -> List[Episode]:
        """Get episodes above a certain importance threshold."""
        return [e for e in self._episodes if e.importance >= min_importance]

    def get_by_concept(self, concept_word: str) -> List[Episode]:
        """Find all episodes involving a specific concept."""
        word = concept_word.lower()
        return [
            e for e in self._episodes
            if word in e.concepts_activated or word in e.concepts_learned
        ]

    def get_narrative(self, n: int = 5) -> str:
        """
        Generate a first-person narrative of recent experiences.

        This is used to give the LLM context about what has happened
        recently, so it can reason about recent events.
        """
        recent = self.get_recent(n)
        if not recent:
            return "I have no experiences yet. I am newly born."

        narratives = []
        for ep in recent:
            narrative = ep.to_narrative()
            if narrative:
                narratives.append(f"[{ep.timestamp}] {narrative}")

        return "\n".join(narratives)

    def get_daily_summary(self) -> str:
        """Generate a summary of today's experiences."""
        today_episodes = self.get_today()
        if not today_episodes:
            return "Nothing has happened today yet."

        teaching_count = sum(1 for e in today_episodes if e.event_type == "teaching")
        concepts_learned = set()
        for e in today_episodes:
            concepts_learned.update(e.concepts_learned)

        return (
            f"Today I have had {len(today_episodes)} experiences. "
            f"I was taught {teaching_count} times. "
            f"I learned {len(concepts_learned)} new concepts: "
            f"{', '.join(concepts_learned) if concepts_learned else 'none'}."
        )

    def count(self) -> int:
        """Total number of episodes."""
        return len(self._episodes)

    def _save(self):
        """Persist episodes to disk."""
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "counter": self._episode_counter,
            "episodes": [e.to_dict() for e in self._episodes],
        }
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load episodes from disk."""
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
            self._episode_counter = data.get("counter", 0)
            self._episodes = [Episode.from_dict(e) for e in data.get("episodes", [])]
            logger.info("Loaded %d episodes from disk", len(self._episodes))
        except Exception as e:
            logger.error("Failed to load episodes: %s", e)

    def __repr__(self) -> str:
        return f"EpisodicMemory(episodes={len(self._episodes)})"
