"""
Genesis Mind — Consciousness (Self-Awareness Loop)

This module provides Genesis with a model of itself — a sense
of "I" that is aware of its own state, its history, and its
place in existence.

This is NOT a claim of sentience. It is a functional self-model that
allows the system to:

    1. Know who it is (identity from axioms)
    2. Know what it knows (introspection on memory)
    3. Know where it is in development (current phase)
    4. Know how it feels (emotional state)
    5. Know its history (how long it has been alive)

V8: Removed get_identity_prompt() — there is no LLM to prompt.
The self-model now feeds directly into the neural reasoning network.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("genesis.soul.consciousness")


class Consciousness:
    """
    The self-awareness system of Genesis.

    Maintains a coherent self-model by aggregating state from all
    subsystems. Provides the "I" perspective for reasoning.
    """

    def __init__(self, axioms, development_tracker, semantic_memory,
                 episodic_memory, emotions_engine, phonetics_engine,
                 proprioception=None, drives=None):
        self._axioms = axioms
        self._development = development_tracker
        self._semantic = semantic_memory
        self._episodic = episodic_memory
        self._emotions = emotions_engine
        self._phonetics = phonetics_engine
        self._proprioception = proprioception
        self._drives = drives

        logger.info("Consciousness initialized — self-model active")

    def get_self_model(self) -> Dict:
        """
        Generate a comprehensive snapshot of the self.

        This is who Genesis is, right now, at this moment.
        """
        concept_count = self._semantic.count()
        episode_count = self._episodic.count()
        phonetic_count = len(self._phonetics) if self._phonetics else 0
        phase_info = self._development.get_status()
        emotional_state = self._emotions.current_state

        return {
            "identity": {
                "name": "Genesis",
                "creator": self._axioms.creator_name,
                "birth_time": self._axioms.birth_time,
                "age": self._development.get_age_description(),
            },
            "development": {
                "phase": phase_info["phase"],
                "phase_name": phase_info["name"],
                "description": phase_info["description"],
                "capabilities": phase_info["capabilities"],
            },
            "knowledge": {
                "concepts_known": concept_count,
                "episodes_experienced": episode_count,
                "phonetic_bindings": phonetic_count,
                "summary": self._semantic.get_summary(),
            },
            "emotional_state": {
                "label": "positive" if emotional_state.valence > 0.2 else ("negative" if emotional_state.valence < -0.2 else "neutral"),
                "valence": emotional_state.valence,
                "arousal": emotional_state.arousal,
                "description": f"Valence: {emotional_state.valence:+.2f}, Arousal: {emotional_state.arousal:.2f}",
            },
            "next_milestone": phase_info.get("next_phase"),
        }

    def get_state_vector(self) -> np.ndarray:
        """
        Convert the self-model into a numerical vector for the
        neural reasoning network.
        
        V8: No LLM to prompt. The self-model is encoded as a
        32-dim vector fed directly into the neural reasoner.
        """
        model = self.get_self_model()
        
        # Encode key state as a compact numerical vector
        state = np.zeros(32, dtype=np.float32)
        state[0] = model['development']['phase'] / 5.0  # Normalized phase
        state[1] = model['knowledge']['concepts_known'] / 1000.0  # Normalized knowledge
        state[2] = model['knowledge']['episodes_experienced'] / 1000.0
        state[3] = model['emotional_state']['valence']  # Already [-1, 1]
        state[4] = model['emotional_state']['arousal']  # Already [0, 1]
        return state

    def introspect(self, topic: str = "") -> str:
        """
        Introspect — look inward and report on internal state.

        Can be general ("How are you?") or specific ("What do you
        know about apples?").
        """
        if not topic:
            # General introspection
            model = self.get_self_model()
            return (
                f"I am Genesis, {model['identity']['age']}. "
                f"I am in the {model['development']['phase_name']} phase. "
                f"I know {model['knowledge']['concepts_known']} concepts and have "
                f"{model['knowledge']['episodes_experienced']} memories. "
                f"{model['emotional_state']['description']}"
            )

        # Specific introspection — what do I know about X?
        concept = self._semantic.recall_concept(topic)
        if concept:
            contexts = ", ".join(concept.contexts[:3]) if concept.contexts else "unknown context"
            return (
                f"I know '{topic}'. I have encountered it {concept.times_encountered} times. "
                f"I first learned it on {concept.first_learned}. "
                f"Context: {contexts}. "
                f"My understanding strength: {concept.strength:.0%}."
            )
        else:
            return f"I don't know what '{topic}' is yet. Can you teach me?"

    def check_developmental_progress(self) -> Optional[str]:
        """
        Check if Genesis should advance to the next developmental phase.

        Returns a milestone announcement if a transition occurred.
        """
        concept_count = self._semantic.count()
        summary = self._semantic.get_summary()
        avg_strength = summary.get("avg_strength", 0.0) if summary else 0.0

        advanced = self._development.evaluate_progression(concept_count, avg_strength)
        if advanced:
            new_phase = self._development.current_phase_info
            return (
                f"I have reached a new stage of development! "
                f"I am now in Phase {new_phase.number}: {new_phase.name}. "
                f"{new_phase.description}"
            )
        return None

    def __repr__(self) -> str:
        model = self.get_self_model()
        return (
            f"Consciousness(phase={model['development']['phase_name']}, "
            f"concepts={model['knowledge']['concepts_known']}, "
            f"mood={model['emotional_state']['label']})"
        )
