"""
Genesis Mind — Phoneme Embedder (From-Scratch Text Representation)

Replaces SBERT/SentenceTransformer (33M+ pretrained params) with a
tiny character-level GRU that learns to embed phoneme sequences.

Architecture:
    Character embedding (vocab_size=64, embed_dim=16)
    → GRU (hidden_dim=32)
    → Linear → 64-dim output

Training: Contrastive learning — phoneme sequences that co-occur with
the same visual input get similar embeddings. This is how infants learn
that the sound "dah" goes with the visual experience of "dad".

Total params: ~10K (vs SBERT's 33M). All learned from scratch.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("genesis.neural.phoneme_embedder")


# Character vocabulary — maps characters to indices
# Covers all phonemes from BabblingEngine + basic letters
CHAR_VOCAB = {c: i + 1 for i, c in enumerate(
    "abcdefghijklmnopqrstuvwxyz "
    "0123456789.,!?'-"
)}
CHAR_VOCAB["<PAD>"] = 0
VOCAB_SIZE = len(CHAR_VOCAB) + 1  # +1 for unknown chars
MAX_SEQ_LEN = 64


class PhonemeGRU(nn.Module):
    """
    Character-level GRU: text/phoneme string → 64-dim embedding.
    
    Unlike SBERT (which uses a pretrained transformer with full
    WordPiece tokenization), this is a tiny GRU that learns
    character-level patterns from scratch.
    """
    
    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 16,
                 hidden_dim: int = 32, output_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) integer tensor of character indices
        Returns:
            (batch, output_dim) embedding vector
        """
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)  # hidden: (1, batch, hidden_dim)
        hidden = hidden.squeeze(0)      # (batch, hidden_dim)
        return self.fc(hidden)           # (batch, output_dim)


class PhonemeEmbedder:
    """
    Genesis's text/phoneme representation system — learns from scratch.
    
    Converts character sequences into 64-dim vectors through a learned
    GRU. Initially random — identical strings won't even have similar
    embeddings. Learns through contrastive pairing with visual/auditory
    embeddings during teaching.
    """
    
    def __init__(self, output_dim: int = 64, learning_rate: float = 1e-3,
                 storage_path: Optional[Path] = None):
        self.output_dim = output_dim
        self._storage_path = storage_path
        
        self.network = PhonemeGRU(output_dim=output_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self._train_steps = 0
        self._encode_count = 0
        
        self._load()
        
        total_params = sum(p.numel() for p in self.network.parameters())
        logger.info(
            "Phoneme embedder initialized (output_dim=%d, params=%d, trained=%d)",
            output_dim, total_params, self._train_steps,
        )
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a text/phoneme string into a 64-dim vector.
        
        Args:
            text: Any string (word, phoneme sequence, sentence)
            
        Returns:
            64-dim numpy embedding
        """
        self.network.eval()
        indices = self._text_to_indices(text)
        with torch.no_grad():
            embedding = self.network(indices)
        self._encode_count += 1
        return embedding.squeeze(0).cpu().numpy()
    
    def train_contrastive(self, text: str, target_embedding: np.ndarray,
                          margin: float = 0.5) -> float:
        """
        Train to align text embedding with a target (visual/auditory) embedding.
        
        When Genesis sees an object and hears a word at the same time,
        this trains the phoneme embedder to produce a similar vector for
        the word as the visual cortex produced for the object.
        
        Args:
            text: The word/phonemes to embed
            target_embedding: The visual or auditory embedding to align with
            margin: Contrastive margin
            
        Returns:
            Loss value
        """
        self.network.train()
        
        indices = self._text_to_indices(text)
        target = torch.tensor(target_embedding, dtype=torch.float32).unsqueeze(0)
        
        # Truncate or pad target to match output_dim
        if target.shape[1] != self.output_dim:
            if target.shape[1] > self.output_dim:
                target = target[:, :self.output_dim]
            else:
                pad = torch.zeros(1, self.output_dim - target.shape[1])
                target = torch.cat([target, pad], dim=1)
        
        # Forward pass
        embedding = self.network(indices)
        
        # Cosine similarity loss — push embedding toward target
        loss = 1.0 - F.cosine_similarity(embedding, target, dim=1).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self._train_steps += 1
        
        if self._train_steps % 50 == 0:
            logger.info("Phoneme embedder: %d train steps, loss=%.4f", 
                        self._train_steps, loss.item())
            self._save()
        
        return loss.item()
    
    def _text_to_indices(self, text: str) -> torch.Tensor:
        """Convert text string to padded integer tensor."""
        text = text.lower()[:MAX_SEQ_LEN]
        indices = [CHAR_VOCAB.get(c, VOCAB_SIZE - 1) for c in text]
        # Pad to MAX_SEQ_LEN
        indices += [0] * (MAX_SEQ_LEN - len(indices))
        return torch.tensor([indices], dtype=torch.long)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two phoneme embeddings."""
        e1 = np.array(emb1, dtype=np.float32).flatten()
        e2 = np.array(emb2, dtype=np.float32).flatten()
        n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
        if n1 > 0 and n2 > 0:
            return float(np.dot(e1, e2) / (n1 * n2))
        return 0.0
    
    def get_stats(self) -> dict:
        return {
            "train_steps": self._train_steps,
            "encode_count": self._encode_count,
            "output_dim": self.output_dim,
            "total_params": sum(p.numel() for p in self.network.parameters()),
        }
    
    def _save(self):
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self._train_steps,
        }, self._storage_path)
    
    def _load(self):
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            checkpoint = torch.load(self._storage_path, map_location="cpu", weights_only=False)
            self.network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self._train_steps = checkpoint.get("train_steps", 0)
            logger.info("Phoneme embedder loaded (%d train steps)", self._train_steps)
        except Exception as e:
            logger.error("Failed to load phoneme embedder: %s", e)
    
    def save(self):
        self._save()
    
    def __repr__(self) -> str:
        return f"PhonemeEmbedder(dim={self.output_dim}, steps={self._train_steps})"
