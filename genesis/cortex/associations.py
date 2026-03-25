"""
Genesis Mind — The Multimodal Association Engine

This is the heart of the system — the engine that creates UNDERSTANDING.

When Genesis sees an apple (visual) while hearing "apple" (audio),
this engine binds those two representations together into a single
concept. This is called "multimodal grounding" and it is the
fundamental difference between Genesis and a traditional LLM.

An LLM knows "apple" → "red, fruit, tree" because those words
appear near each other in text. Genesis knows "apple" because
it has SEEN an apple and HEARD the word — the two are bound
together through shared experience.

The binding process:
1. Visual input arrives as a 512-dim CLIP embedding
2. Auditory input arrives as transcribed text + phonemes
3. Text is converted to a 384-dim sentence embedding
4. The engine computes cross-modal alignment scores
5. If alignment is high enough, a Binding is created/strengthened
6. The Binding is stored in the semantic memory as a Concept
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("genesis.cortex.associations")


@dataclass
class Binding:
    """
    A multimodal binding — the link between what is seen and what is heard.

    This is the atomic unit of understanding. When a Binding exists,
    Genesis literally "knows" that a visual pattern and an auditory
    pattern refer to the same thing.
    """
    word: str                               # The word being bound
    visual_embedding: Optional[List[float]] = None   # What it looks like
    text_embedding: Optional[List[float]] = None      # Linguistic representation
    alignment_score: float = 0.0            # How well the modalities match
    source_context: str = ""                # Where/when binding was created
    strength: float = 0.1                   # Binding strength (0→1)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class AssociationEngine:
    """
    The multimodal binding engine of Genesis.

    Creates and manages cross-modal associations — the connections
    between what is seen, heard, and spoken. This is the engine
    of understanding.
    """

    def __init__(self):
        self._text_model = None
        self._bindings: Dict[str, Binding] = {}
        logger.info("Association engine initialized")

    def _load_text_model(self):
        """Lazy-load the sentence transformer for text embeddings."""
        if self._text_model is not None:
            return

        logger.info("Loading text embedding model (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        self._text_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("Text embedding model loaded")

    def embed_text(self, text: str) -> np.ndarray:
        """Convert text to a 384-dim embedding vector."""
        self._load_text_model()
        embedding = self._text_model.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Convert multiple texts to embedding vectors."""
        self._load_text_model()
        embeddings = self._text_model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def compute_cross_modal_similarity(
        self,
        visual_embedding: np.ndarray,
        text: str,
        clip_text_embedding_fn=None,
    ) -> float:
        """
        Compute how well a visual percept matches a word.

        Uses CLIP's shared embedding space: if the image of a dog
        and the text "dog" are close in CLIP space, they likely
        refer to the same concept.

        Args:
            visual_embedding: CLIP visual embedding (512-dim)
            text: The word or phrase to compare against
            clip_text_embedding_fn: Function to embed text in CLIP space
                                    (provided by Eyes.embed_text)
        """
        if clip_text_embedding_fn is None:
            # Can't compute true cross-modal similarity without CLIP text encoder
            logger.debug("No CLIP text encoder available — skipping cross-modal alignment")
            return 0.0

        # Embed text in CLIP's visual-linguistic space
        text_emb = clip_text_embedding_fn(text)

        # Cosine similarity in CLIP space
        similarity = float(
            np.dot(visual_embedding, text_emb) /
            (np.linalg.norm(visual_embedding) * np.linalg.norm(text_emb) + 1e-8)
        )

        return similarity

    def create_binding(
        self,
        word: str,
        visual_embedding: Optional[np.ndarray] = None,
        context: str = "",
        clip_text_embedding_fn=None,
    ) -> Binding:
        """
        Create or strengthen a multimodal binding.

        This is called when Genesis experiences a teaching moment:
        someone shows it an object (visual) and says its name (audio).

        The binding connects the visual representation to the word,
        creating genuine understanding.
        """
        key = word.lower().strip()

        # Compute text embedding
        text_embedding = self.embed_text(word).tolist()

        # Compute cross-modal alignment if visual data is available
        alignment = 0.0
        if visual_embedding is not None and clip_text_embedding_fn:
            alignment = self.compute_cross_modal_similarity(
                visual_embedding, word, clip_text_embedding_fn,
            )

        if key in self._bindings:
            # Strengthen existing binding
            binding = self._bindings[key]
            binding.strength = min(1.0, binding.strength + 0.05)
            binding.alignment_score = max(binding.alignment_score, alignment)
            if visual_embedding is not None:
                binding.visual_embedding = visual_embedding.tolist() if isinstance(visual_embedding, np.ndarray) else visual_embedding
            binding.text_embedding = text_embedding
            logger.info(
                "Binding strengthened: '%s' (strength=%.2f, alignment=%.2f)",
                word, binding.strength, binding.alignment_score,
            )
        else:
            # Create new binding — a new connection formed!
            binding = Binding(
                word=key,
                visual_embedding=visual_embedding.tolist() if isinstance(visual_embedding, np.ndarray) else visual_embedding,
                text_embedding=text_embedding,
                alignment_score=alignment,
                source_context=context,
            )
            self._bindings[key] = binding
            logger.info(
                "NEW binding created: '%s' (alignment=%.2f)",
                word, alignment,
            )

        return binding

    def find_best_match(
        self,
        visual_embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Given a visual input, find the words that best match it.

        This is how Genesis answers "What is this?" — it compares
        what it sees to all the visual embeddings in its bindings.
        """
        if not self._bindings:
            return []

        scores = []
        for word, binding in self._bindings.items():
            if binding.visual_embedding is not None:
                stored = np.array(binding.visual_embedding)
                similarity = float(
                    np.dot(visual_embedding, stored) /
                    (np.linalg.norm(visual_embedding) * np.linalg.norm(stored) + 1e-8)
                )
                scores.append((word, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def find_most_similar_words(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find words with the most similar text embeddings."""
        if word.lower() not in self._bindings:
            return []

        query_emb = np.array(self._bindings[word.lower()].text_embedding)
        scores = []

        for other_word, binding in self._bindings.items():
            if other_word == word.lower() or binding.text_embedding is None:
                continue
            stored = np.array(binding.text_embedding)
            similarity = float(
                np.dot(query_emb, stored) /
                (np.linalg.norm(query_emb) * np.linalg.norm(stored) + 1e-8)
            )
            scores.append((other_word, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_binding_count(self) -> int:
        """Total number of multimodal bindings."""
        return len(self._bindings)

    def __repr__(self) -> str:
        return f"AssociationEngine(bindings={len(self._bindings)})"
