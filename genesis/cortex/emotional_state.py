"""
Genesis Mind — Persistent Emotional State

Emotions in a real brain are not labels computed per-event.
They are continuous dynamics:

    1. MOMENTUM: Emotions have inertia — anger doesn't vanish instantly
    2. BLENDING: You can feel anxious AND excited simultaneously
    3. DECAY: Emotions naturally return to baseline over time
    4. CONTAGION: Detecting emotion in others influences your own
    5. MOOD: Long-term emotional baseline shifts slowly

The emotional state is a continuous vector that is ALWAYS active,
evolving tick by tick in the brain daemon — not just when
something happens.
"""

import logging
import time
import math
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("genesis.cortex.emotional_state")


class EmotionalDimension:
    """A single emotional dimension with momentum and decay."""

    def __init__(self, name: str, baseline: float = 0.0,
                 decay_rate: float = 0.02, momentum: float = 0.8):
        self.name = name
        self.baseline = baseline
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.value = baseline
        self._velocity = 0.0  # Rate of change

    def nudge(self, amount: float):
        """Apply an emotional impulse (positive or negative)."""
        self._velocity += amount * (1.0 - self.momentum * 0.5)

    def tick(self, dt: float = 1.0):
        """Advance the emotional state by one time step."""
        # Apply velocity (momentum)
        self.value += self._velocity * dt

        # Decay velocity (momentum loss)
        self._velocity *= self.momentum

        # Decay toward baseline
        diff = self.value - self.baseline
        self.value -= diff * self.decay_rate * dt

        # Clamp
        self.value = max(-1.0, min(1.0, self.value))

        # Kill tiny velocities
        if abs(self._velocity) < 0.001:
            self._velocity = 0.0

    def __repr__(self) -> str:
        return f"{self.name}={self.value:+.3f}"


class PersistentEmotionalState:
    """
    Continuous, multi-dimensional emotional state.

    8 emotional dimensions that blend, decay, and have momentum:
        joy/sadness
        excitement/calm
        trust/fear
        anger/peace
        surprise/anticipation
        disgust/acceptance
        interest/boredom
        love/loneliness
    """

    def __init__(self):
        self.dimensions = {
            "joy":         EmotionalDimension("joy", baseline=0.1, decay_rate=0.03),
            "excitement":  EmotionalDimension("excitement", baseline=0.0, decay_rate=0.05),
            "trust":       EmotionalDimension("trust", baseline=0.0, decay_rate=0.01),
            "anger":       EmotionalDimension("anger", baseline=0.0, decay_rate=0.02, momentum=0.9),
            "surprise":    EmotionalDimension("surprise", baseline=0.0, decay_rate=0.1),
            "disgust":     EmotionalDimension("disgust", baseline=0.0, decay_rate=0.04),
            "interest":    EmotionalDimension("interest", baseline=0.2, decay_rate=0.02),
            "love":        EmotionalDimension("love", baseline=0.0, decay_rate=0.005, momentum=0.95),
        }

        # Mood: slow-moving emotional baseline (shifts over hours)
        self._mood_vector = np.zeros(len(self.dimensions))
        self._mood_momentum = 0.995  # Very slow to change

        # Contagion sensitivity (how much external emotion affects us)
        self.contagion_sensitivity = 0.3

        self._tick_count = 0
        logger.info("Persistent emotional state initialized (8 dimensions)")

    def tick(self):
        """Advance all emotional dimensions by one step."""
        for dim in self.dimensions.values():
            dim.tick()
        self._tick_count += 1

        # Update mood (very slow moving average)
        current = self.get_vector()
        self._mood_vector = self._mood_vector * self._mood_momentum + \
                           current * (1.0 - self._mood_momentum)

    def on_experience(self, valence: float, arousal: float, novelty: float = 0.0):
        """
        React emotionally to an experience.

        Args:
            valence: Positive/negative (-1 to +1)
            arousal: Calm/excited (0 to 1)
            novelty: How surprising (0 to 1)
        """
        if valence > 0:
            self.dimensions["joy"].nudge(valence * 0.3)
            self.dimensions["trust"].nudge(valence * 0.1)
        else:
            self.dimensions["anger"].nudge(abs(valence) * 0.1)

        self.dimensions["excitement"].nudge(arousal * 0.3)
        self.dimensions["surprise"].nudge(novelty * 0.5)
        self.dimensions["interest"].nudge(novelty * 0.3)

    def on_social_interaction(self, positive: bool = True):
        """React to social interaction."""
        if positive:
            self.dimensions["love"].nudge(0.1)
            self.dimensions["joy"].nudge(0.05)
            self.dimensions["trust"].nudge(0.05)
        else:
            self.dimensions["anger"].nudge(0.1)
            self.dimensions["trust"].nudge(-0.1)

    def on_contagion(self, external_valence: float, external_arousal: float):
        """
        Emotional contagion — detect emotion in others and mirror it.
        Like a baby crying when another baby cries.
        """
        strength = self.contagion_sensitivity
        if external_valence > 0:
            self.dimensions["joy"].nudge(external_valence * strength)
        else:
            self.dimensions["anger"].nudge(abs(external_valence) * strength * 0.5)
        self.dimensions["excitement"].nudge(external_arousal * strength)

    def get_vector(self) -> np.ndarray:
        """Get the current emotional state as a numpy vector."""
        return np.array([d.value for d in self.dimensions.values()])

    def get_mood(self) -> np.ndarray:
        """Get the long-term mood baseline."""
        return self._mood_vector.copy()

    def get_dominant_emotion(self) -> str:
        """Get the currently dominant emotion."""
        return max(self.dimensions.items(), key=lambda x: abs(x[1].value))[0]

    def get_valence(self) -> float:
        """Overall positive/negative feeling."""
        positive = self.dimensions["joy"].value + self.dimensions["love"].value + self.dimensions["trust"].value
        negative = self.dimensions["anger"].value + self.dimensions["disgust"].value
        return (positive - negative) / 3.0

    def get_arousal(self) -> float:
        """Overall activation level."""
        return (abs(self.dimensions["excitement"].value) +
                abs(self.dimensions["surprise"].value) +
                abs(self.dimensions["interest"].value)) / 3.0

    def get_emotional_intensity(self) -> float:
        """Total emotional intensity (for attention/salience)."""
        return float(np.mean(np.abs(self.get_vector())))

    def get_status(self) -> Dict:
        return {
            "dimensions": {name: round(dim.value, 3) for name, dim in self.dimensions.items()},
            "dominant": self.get_dominant_emotion(),
            "valence": round(self.get_valence(), 3),
            "arousal": round(self.get_arousal(), 3),
            "intensity": round(self.get_emotional_intensity(), 3),
            "mood_valence": round(float(np.mean(self._mood_vector[:3])), 3),
        }

    def __repr__(self) -> str:
        return f"EmotionalState(dominant={self.get_dominant_emotion()}, valence={self.get_valence():.2f})"
