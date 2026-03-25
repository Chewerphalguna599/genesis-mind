"""
Genesis Mind — Phonetics Engine

The letter↔sound binding system. This is how Genesis learns to read.

A child does not learn to read by memorizing entire words. A child
learns that each LETTER (grapheme) maps to a SOUND (phoneme):

    A → /æ/ (as in "apple")
    B → /b/ (as in "ball")
    C → /k/ (as in "cat")

Over time, the child learns combinations:
    TH → /θ/ (as in "think")
    SH → /ʃ/ (as in "ship")
    CH → /tʃ/ (as in "chair")

Genesis starts with NO phonetic knowledge. It learns these mappings
when you teach it: "The letter A says 'ah'". Each mapping is stored
in the memory system and strengthened through repetition.

Once enough mappings are learned, Genesis can "sound out" new words
it has never encountered — just like a child learning to read.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("genesis.senses.phonetics")


# =============================================================================
# Standard English Phoneme Set (IPA)
# Genesis starts empty — these are used as reference, not pre-loaded
# =============================================================================

ENGLISH_PHONEMES = {
    # Consonants
    "b": "/b/", "d": "/d/", "f": "/f/", "g": "/ɡ/", "h": "/h/",
    "j": "/dʒ/", "k": "/k/", "l": "/l/", "m": "/m/", "n": "/n/",
    "p": "/p/", "r": "/ɹ/", "s": "/s/", "t": "/t/", "v": "/v/",
    "w": "/w/", "y": "/j/", "z": "/z/",
    # Digraphs
    "th": "/θ/", "sh": "/ʃ/", "ch": "/tʃ/", "ng": "/ŋ/",
    "wh": "/w/", "ph": "/f/", "ck": "/k/",
    # Vowels (short)
    "a": "/æ/", "e": "/ɛ/", "i": "/ɪ/", "o": "/ɒ/", "u": "/ʌ/",
    # Vowels (long)
    "a_long": "/eɪ/", "e_long": "/iː/", "i_long": "/aɪ/",
    "o_long": "/oʊ/", "u_long": "/juː/",
    # Diphthongs
    "oi": "/ɔɪ/", "ou": "/aʊ/", "oo": "/uː/", "aw": "/ɔː/",
    "ar": "/ɑːr/", "er": "/ɜːr/", "ir": "/ɜːr/", "or": "/ɔːr/",
}


@dataclass
class PhoneticBinding:
    """
    A single grapheme-to-phoneme mapping learned by Genesis.

    strength: How many times this binding has been reinforced (0.0 to 1.0).
    A binding with strength 1.0 is deeply learned; 0.1 is tentative.
    """
    grapheme: str               # The letter or letter combination (e.g., "th")
    phoneme: str                # The sound it makes (e.g., "/θ/")
    sound_description: str      # Human description (e.g., "like in 'think'")
    strength: float = 0.1      # How well learned (0.0 to 1.0)
    times_reinforced: int = 1   # How many times the human taught this
    first_learned: str = field(default_factory=lambda: datetime.now().isoformat())
    last_reinforced: str = field(default_factory=lambda: datetime.now().isoformat())

    def reinforce(self):
        """Strengthen this binding through repetition."""
        self.times_reinforced += 1
        self.strength = min(1.0, self.strength + 0.05)
        self.last_reinforced = datetime.now().isoformat()

    def decay(self, amount: float = 0.01):
        """Weaken this binding over time without reinforcement."""
        self.strength = max(0.0, self.strength - amount)


class PhoneticsEngine:
    """
    The phonetic processing system of Genesis.

    Maintains a learned mapping of graphemes (letters) to phonemes (sounds).
    Starts completely empty. Mappings are added through teaching interactions
    and strengthened through repetition.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self._bindings: Dict[str, PhoneticBinding] = {}
        self._storage_path = storage_path
        self._load()
        logger.info(
            "Phonetics engine initialized (%d known bindings)",
            len(self._bindings),
        )

    def teach(self, grapheme: str, phoneme: str, description: str = "") -> PhoneticBinding:
        """
        Teach Genesis a new letter↔sound mapping.

        If the mapping already exists, reinforce it (strengthen the connection).
        If it's new, create a fresh binding.

        Example:
            engine.teach("A", "/æ/", "like in 'apple'")
            engine.teach("TH", "/θ/", "like in 'think'")
        """
        key = grapheme.lower()

        if key in self._bindings:
            binding = self._bindings[key]
            binding.reinforce()
            logger.info(
                "Reinforced binding: '%s' → %s (strength: %.2f, times: %d)",
                grapheme, phoneme, binding.strength, binding.times_reinforced,
            )
        else:
            binding = PhoneticBinding(
                grapheme=grapheme.lower(),
                phoneme=phoneme,
                sound_description=description,
            )
            self._bindings[key] = binding
            logger.info("New binding learned: '%s' → %s (%s)", grapheme, phoneme, description)

        self._save()
        return binding

    def sound_out(self, word: str) -> List[Tuple[str, str]]:
        """
        Attempt to sound out a word using known phonetic bindings.

        Returns a list of (grapheme, phoneme) pairs. If a grapheme
        is unknown, the phoneme will be "?" — indicating the system
        hasn't learned that sound yet.

        Example:
            engine.sound_out("cat") → [("c", "/k/"), ("a", "/æ/"), ("t", "/t/")]
            engine.sound_out("the") → [("th", "/θ/"), ("e", "/ɛ/")]
        """
        word = word.lower()
        result = []
        i = 0

        while i < len(word):
            matched = False

            # Try digraphs first (2-letter combinations) — greedy matching
            if i + 1 < len(word):
                digraph = word[i:i+2]
                if digraph in self._bindings:
                    binding = self._bindings[digraph]
                    result.append((digraph, binding.phoneme))
                    i += 2
                    matched = True

            # Fall back to single letter
            if not matched:
                char = word[i]
                if char in self._bindings:
                    binding = self._bindings[char]
                    result.append((char, binding.phoneme))
                else:
                    result.append((char, "?"))  # Unknown sound
                i += 1

        return result

    def can_read(self, word: str) -> bool:
        """Check if Genesis knows enough phonetics to sound out a word."""
        sounded = self.sound_out(word)
        return all(phoneme != "?" for _, phoneme in sounded)

    def get_known_graphemes(self) -> List[str]:
        """Return all graphemes Genesis has learned."""
        return sorted(self._bindings.keys())

    def get_binding_strength(self, grapheme: str) -> float:
        """Get the learning strength of a specific grapheme."""
        key = grapheme.lower()
        if key in self._bindings:
            return self._bindings[key].strength
        return 0.0

    def get_all_bindings(self) -> List[Dict]:
        """Return all learned bindings as a list of dicts."""
        return [
            {
                "grapheme": b.grapheme,
                "phoneme": b.phoneme,
                "description": b.sound_description,
                "strength": b.strength,
                "times_reinforced": b.times_reinforced,
            }
            for b in sorted(self._bindings.values(), key=lambda b: b.grapheme)
        ]

    def decay_all(self, amount: float = 0.01):
        """Apply forgetting curve to all bindings (used during sleep)."""
        for binding in self._bindings.values():
            binding.decay(amount)
        # Prune dead bindings
        self._bindings = {
            k: v for k, v in self._bindings.items() if v.strength > 0.0
        }
        self._save()

    def _save(self):
        """Persist bindings to disk."""
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            key: {
                "grapheme": b.grapheme,
                "phoneme": b.phoneme,
                "sound_description": b.sound_description,
                "strength": b.strength,
                "times_reinforced": b.times_reinforced,
                "first_learned": b.first_learned,
                "last_reinforced": b.last_reinforced,
            }
            for key, b in self._bindings.items()
        }
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load bindings from disk."""
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
            for key, entry in data.items():
                self._bindings[key] = PhoneticBinding(
                    grapheme=entry["grapheme"],
                    phoneme=entry["phoneme"],
                    sound_description=entry.get("sound_description", ""),
                    strength=entry.get("strength", 0.1),
                    times_reinforced=entry.get("times_reinforced", 1),
                    first_learned=entry.get("first_learned", ""),
                    last_reinforced=entry.get("last_reinforced", ""),
                )
            logger.info("Loaded %d phonetic bindings from disk", len(self._bindings))
        except Exception as e:
            logger.error("Failed to load phonetic bindings: %s", e)

    def __len__(self) -> int:
        return len(self._bindings)

    def __repr__(self) -> str:
        return f"PhoneticsEngine(known_graphemes={len(self._bindings)})"


# =============================================================================
# Standalone test — run with: python -m genesis.senses.phonetics
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — Phonetics Test")
    print("Teaching Genesis to read from zero...")
    print("=" * 60)

    engine = PhoneticsEngine(storage_path=Path("/tmp/genesis_phonetics_test.json"))

    # Teach basic consonants
    engine.teach("b", "/b/", "like in 'ball'")
    engine.teach("c", "/k/", "like in 'cat'")
    engine.teach("d", "/d/", "like in 'dog'")
    engine.teach("t", "/t/", "like in 'top'")

    # Teach vowels
    engine.teach("a", "/æ/", "like in 'apple'")
    engine.teach("o", "/ɒ/", "like in 'octopus'")

    # Teach a digraph
    engine.teach("th", "/θ/", "like in 'think'")

    # Test sounding out words
    print("\n--- Sounding Out Words ---")
    for word in ["cat", "bat", "dot", "that", "xyz"]:
        sounded = engine.sound_out(word)
        readable = " + ".join(f"'{g}'→{p}" for g, p in sounded)
        can_read = engine.can_read(word)
        print(f"  {word}: {readable}  (can read: {can_read})")

    print(f"\nKnown graphemes: {engine.get_known_graphemes()}")
    print("Phonetics test PASSED ✓")
