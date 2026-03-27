"""
Genesis Mind — The Multimodal Association Engine (V8: No Pretrained Models)

Creates UNDERSTANDING by binding visual and auditory experiences.

V8 CHANGE: Removed SentenceTransformer/SBERT (33M pretrained params).
Text embeddings now come from the from-scratch PhonemeEmbedder (~10K params).
Cross-modal similarity uses direct cosine similarity between visual and
phoneme embeddings (both 64-dim) instead of CLIP's shared space.

The binding process (V8):
1. Visual input arrives as a 64-dim VisualCortex embedding
2. Auditory input arrives as phonemes/text
3. Text is converted to a 64-dim PhonemeEmbedder vector
4. Cross-modal alignment: cosine similarity between visual and phoneme vectors
5. If alignment improves with training, binding is strengthened
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("genesis.cortex.associations")


@dataclass
class Binding:
    """A multimodal binding — linking vision to sound/text."""
    word: str
    visual_embedding: Optional[List[float]] = None   # 64-dim from VisualCortex
    text_embedding: Optional[List[float]] = None      # 64-dim from PhonemeEmbedder
    alignment_score: float = 0.0
    source_context: str = ""
    strength: float = 0.1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class AssociationEngine:
    """
    The multimodal binding engine of Genesis (V8: no pretrained models).

    Uses the from-scratch PhonemeEmbedder for text embeddings and
    computes cross-modal alignment directly in the shared 64-dim space.
    """

    def __init__(self, phoneme_embedder=None):
        self._phoneme_embedder = phoneme_embedder
        self._bindings: Dict[str, Binding] = {}
        logger.info("Association engine initialized (from-scratch)")

    def set_phoneme_embedder(self, embedder):
        """Set the phoneme embedder reference (for late binding)."""
        self._phoneme_embedder = embedder

    def embed_text(self, text: str) -> np.ndarray:
        """Convert text to a 64-dim embedding via the from-scratch PhonemeEmbedder."""
        if self._phoneme_embedder is None:
            logger.warning("No phoneme embedder — returning zero embedding")
            return np.zeros(64, dtype=np.float32)
        return self._phoneme_embedder.encode(text)

    def encode_text(self, text: str) -> np.ndarray:
        """Alias for embed_text (backward compatibility)."""
        return self.embed_text(text)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Convert multiple texts to embedding vectors."""
        return np.array([self.embed_text(t) for t in texts], dtype=np.float32)

    def compute_cross_modal_similarity(
        self,
        visual_embedding: np.ndarray,
        text: str,
        **kwargs,  # Accept and ignore clip_text_embedding_fn
    ) -> float:
        """
        Compute how well a visual percept matches a word.

        V8: Direct cosine similarity between VisualCortex (64-dim) and
        PhonemeEmbedder (64-dim) vectors. No CLIP shared space needed.
        """
        text_emb = self.embed_text(text)
        
        visual = np.array(visual_embedding, dtype=np.float32).flatten()
        text_e = np.array(text_emb, dtype=np.float32).flatten()
        
        # Project to common dim
        min_dim = min(len(visual), len(text_e))
        v = visual[:min_dim]
        t = text_e[:min_dim]
        
        nv, nt = np.linalg.norm(v), np.linalg.norm(t)
        if nv > 0 and nt > 0:
            return float(np.dot(v, t) / (nv * nt))
        return 0.0

    def create_binding(
        self,
        word: str,
        visual_embedding: Optional[np.ndarray] = None,
        context: str = "",
        **kwargs,  # Accept and ignore clip_text_embedding_fn
    ) -> Binding:
        """
        Create or strengthen a multimodal binding.

        Also trains the PhonemeEmbedder to align the text embedding
        with the visual embedding (contrastive learning).
        """
        key = word.lower().strip()

        # Compute text embedding
        text_embedding = self.embed_text(word).tolist()

        # Compute cross-modal alignment
        alignment = 0.0
        if visual_embedding is not None:
            alignment = self.compute_cross_modal_similarity(
                visual_embedding, word,
            )
            # Train the phoneme embedder to align with the visual embedding
            if self._phoneme_embedder is not None:
                self._phoneme_embedder.train_contrastive(word, visual_embedding)

        if key in self._bindings:
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
            binding = Binding(
                word=key,
                visual_embedding=visual_embedding.tolist() if isinstance(visual_embedding, np.ndarray) else visual_embedding,
                text_embedding=text_embedding,
                alignment_score=alignment,
                source_context=context,
            )
            self._bindings[key] = binding
            logger.info("NEW binding created: '%s' (alignment=%.2f)", word, alignment)

        return binding

    def find_best_match(
        self,
        visual_embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Given a visual input, find the words that best match it."""
        if not self._bindings:
            return []

        scores = []
        for word, binding in self._bindings.items():
            if binding.visual_embedding is not None:
                stored = np.array(binding.visual_embedding)
                visual = np.array(visual_embedding).flatten()
                min_dim = min(len(visual), len(stored))
                v, s = visual[:min_dim], stored[:min_dim]
                nv, ns = np.linalg.norm(v), np.linalg.norm(s)
                if nv > 0 and ns > 0:
                    similarity = float(np.dot(v, s) / (nv * ns))
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
            min_dim = min(len(query_emb), len(stored))
            q, s = query_emb[:min_dim], stored[:min_dim]
            nq, ns = np.linalg.norm(q), np.linalg.norm(s)
            if nq > 0 and ns > 0:
                similarity = float(np.dot(q, s) / (nq * ns))
                scores.append((other_word, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_binding_count(self) -> int:
        return len(self._bindings)

    def __repr__(self) -> str:
        return f"AssociationEngine(bindings={len(self._bindings)}, from_scratch=True)"
