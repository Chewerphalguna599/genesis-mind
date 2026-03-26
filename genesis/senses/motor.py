"""
Genesis Mind — Simulated Motor System

A real brain learns through ACTION — reaching, grasping, pointing.
Motor actions generate sensory feedback that drives cognitive
development. The body IS the mind.

Since Genesis runs on a laptop without physical actuators, this
module provides SIMULATED motor affordances:

    1. LOOK_AT: Direct visual attention to a concept/object
    2. VOCALIZE: Produce speech (tied to voice system)
    3. REACH: Attempt to interact with a concept (triggers recall)
    4. POINT: Direct others' attention (social gesture)
    5. GESTURE: Express emotional state physically

Each action has consequences that feed back into perception,
creating a sensorimotor loop even without physical hardware.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

logger = logging.getLogger("genesis.senses.motor")


@dataclass
class MotorAction:
    """A single motor action with consequences."""
    action_type: str
    target: str
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    feedback: str = ""


class SimulatedMotor:
    """
    Simulated motor system — actions with sensory consequences.

    Each motor action triggers feedback through the sensory
    systems, creating a perception-action loop.
    """

    def __init__(self):
        self._action_history: List[MotorAction] = []
        self._action_counts: Dict[str, int] = {}
        self._motor_development: float = 0.0  # 0=newborn, 1=fully coordinated
        self._fatigue: float = 0.0
        self._callbacks: Dict[str, Optional[Callable]] = {
            "look": None,
            "vocalize": None,
            "reach": None,
        }
        logger.info("Simulated motor system initialized")

    def set_callback(self, action_type: str, callback: Callable):
        """Register a callback for motor actions (connects to senses)."""
        self._callbacks[action_type] = callback

    def develop(self, phase: int):
        """Motor development follows cognitive development."""
        # Motor coordination improves with phase
        self._motor_development = min(1.0, phase * 0.2)

    def can_perform(self, action_type: str, phase: int) -> bool:
        """Check if the current developmental phase supports this action."""
        action_phases = {
            "look": 0,      # Even newborns can look
            "vocalize": 0,  # Can babble from birth
            "reach": 1,     # Reaching from infant stage
            "point": 2,     # Pointing from toddler stage
            "gesture": 2,   # Gesturing from toddler stage
        }
        return phase >= action_phases.get(action_type, 0)

    def execute(self, action_type: str, target: str, phase: int) -> MotorAction:
        """
        Execute a motor action.

        Returns the action with feedback. May fail based on
        motor development (like a baby's clumsy reaching).
        """
        if not self.can_perform(action_type, phase):
            return MotorAction(
                action_type=action_type, target=target,
                success=False, feedback="Motor skill not yet developed"
            )

        # Motor success probability increases with development
        import random
        coordination = self._motor_development + (phase * 0.15)
        success = random.random() < min(0.95, coordination)

        # Build feedback
        if action_type == "look":
            feedback = f"Looking at {target}" if success else f"Can't focus on {target}"
        elif action_type == "vocalize":
            if phase == 0:
                feedback = "...babble..."
            elif phase == 1:
                feedback = f"...{target[:4]}..."
            else:
                feedback = f"Saying '{target}'"
        elif action_type == "reach":
            feedback = f"Reaching for {target}" if success else f"Missed {target}"
        elif action_type == "point":
            feedback = f"Pointing at {target}" if success else f"Pointing vaguely"
        elif action_type == "gesture":
            feedback = f"Expressing about {target}" if success else f"Fidgeting"
        else:
            feedback = "Unknown action"

        action = MotorAction(
            action_type=action_type, target=target,
            success=success, feedback=feedback
        )

        self._action_history.append(action)
        self._action_counts[action_type] = self._action_counts.get(action_type, 0) + 1

        # Motor actions increase fatigue slightly
        self._fatigue = min(1.0, self._fatigue + 0.01)

        # Trigger callback if registered
        cb = self._callbacks.get(action_type)
        if cb and success:
            try:
                cb(target)
            except Exception as e:
                logger.debug("Motor callback error for %s: %s", action_type, e)

        return action

    def get_stats(self) -> Dict:
        return {
            "motor_development": round(self._motor_development, 2),
            "fatigue": round(self._fatigue, 2),
            "total_actions": len(self._action_history),
            "action_counts": dict(self._action_counts),
        }
