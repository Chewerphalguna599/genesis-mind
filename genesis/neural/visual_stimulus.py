"""
Genesis Mind — Visual Stimulus Analyzer

Human-like visual processing with differentiated responses to:

    1. MOTION: Frame-to-frame pixel differences (V5/MT area analog)
       → Triggers arousal, alertness, orienting response
    
    2. NOVELTY: How different is this frame from recent history?
       → Triggers curiosity, hippocampal encoding
    
    3. COMPLEXITY: Edge density / spatial frequency (V1/V2 analog)  
       → High complexity = interesting scene, low = boring wall
    
    4. LUMINANCE CHANGE: Brightness shifts (pupillary response analog)
       → Sudden darkness = fear, sudden brightness = surprise

All computed from raw frames — no neural networks needed.
These saliency signals feed into the limbic system and drives
to create differentiated emotional responses to different scenes.
"""

import logging
from collections import deque
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("genesis.neural.visual_stimulus")


class VisualStimulusAnalyzer:
    """
    Extracts perceptual saliency features from raw visual frames.
    
    Computes motion, novelty, complexity, and luminance signals
    that modulate emotional and drive responses — like how a human
    brain reacts differently to a face vs a wall vs a sudden movement.
    """

    def __init__(self, history_size: int = 30):
        # Frame history for novelty computation
        self._frame_history = deque(maxlen=history_size)
        self._prev_frame = None
        
        # Running stats
        self._avg_luminance = 0.5
        self._avg_complexity = 0.0
        self._frames_analyzed = 0
        
        # Exponential moving averages for adaptation
        self._ema_motion = 0.0
        self._ema_complexity = 0.0
        self._ema_luminance = 0.5
        
        logger.info("Visual stimulus analyzer initialized (history=%d)", history_size)

    def analyze(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Analyze a raw image frame and return saliency signals.
        
        Args:
            frame: Raw image as numpy array (H, W, 3) with values [0, 255]
            
        Returns:
            Dict with:
                - motion: 0.0 (still) to 1.0 (rapid movement)
                - novelty: 0.0 (seen this before) to 1.0 (completely new)
                - complexity: 0.0 (flat wall) to 1.0 (rich detailed scene)
                - luminance_change: -1.0 (went dark) to 1.0 (went bright)
                - overall_saliency: Weighted combination
        """
        # Normalize to [0, 1] float
        if frame.dtype == np.uint8:
            frame_f = frame.astype(np.float32) / 255.0
        else:
            frame_f = frame.astype(np.float32)
        
        # Downsample for efficiency (analyze at 16x16)
        h, w = frame_f.shape[:2]
        if h > 16 or w > 16:
            # Simple block averaging
            bh, bw = max(1, h // 16), max(1, w // 16)
            frame_small = frame_f[:bh*16, :bw*16].reshape(16, bh, 16, bw, -1).mean(axis=(1, 3))
        else:
            frame_small = frame_f
        
        # Convert to grayscale for analysis
        if len(frame_small.shape) == 3 and frame_small.shape[-1] >= 3:
            gray = np.mean(frame_small[:, :, :3], axis=-1)
        else:
            gray = frame_small.squeeze() if len(frame_small.shape) > 2 else frame_small
        
        # ─── 1. MOTION DETECTION (V5/MT analog) ───────────────
        motion = self._compute_motion(gray)
        
        # ─── 2. NOVELTY SCORING (hippocampal novelty) ─────────
        novelty = self._compute_novelty(gray)
        
        # ─── 3. COMPLEXITY / EDGE DENSITY (V1/V2 analog) ──────
        complexity = self._compute_complexity(gray)
        
        # ─── 4. LUMINANCE CHANGE (pupillary response) ─────────
        luminance_change = self._compute_luminance_change(gray)
        
        # ─── 5. OVERALL SALIENCY ──────────────────────────────
        # Motion is most alerting, then novelty, then complexity
        overall = (
            motion * 0.35 +
            novelty * 0.30 +
            complexity * 0.20 +
            abs(luminance_change) * 0.15
        )
        
        # Update running stats
        self._prev_frame = gray.copy()
        self._frame_history.append(gray.flatten())
        self._frames_analyzed += 1
        
        # Update EMAs (for adaptation — habituate to constant stimulation)
        alpha = 0.1
        self._ema_motion = self._ema_motion * (1 - alpha) + motion * alpha
        self._ema_complexity = self._ema_complexity * (1 - alpha) + complexity * alpha
        self._ema_luminance = self._ema_luminance * (1 - alpha) + np.mean(gray) * alpha
        
        return {
            "motion": float(np.clip(motion, 0, 1)),
            "novelty": float(np.clip(novelty, 0, 1)),
            "complexity": float(np.clip(complexity, 0, 1)),
            "luminance_change": float(np.clip(luminance_change, -1, 1)),
            "overall_saliency": float(np.clip(overall, 0, 1)),
        }

    def _compute_motion(self, gray: np.ndarray) -> float:
        """Frame-to-frame pixel difference — like V5/MT motion area."""
        if self._prev_frame is None:
            return 0.0
        
        # Mean absolute difference
        diff = np.abs(gray - self._prev_frame)
        raw_motion = float(np.mean(diff))
        
        # Normalize: typical indoor scene drift is ~0.01-0.03
        # Fast motion is ~0.1+
        motion = min(1.0, raw_motion * 10.0)
        
        # Subtract adapted baseline (habituate to constant motion like webcam jitter)
        adapted = max(0.0, motion - self._ema_motion * 0.5)
        return adapted

    def _compute_novelty(self, gray: np.ndarray) -> float:
        """How different is this frame from recent history?"""
        if len(self._frame_history) < 2:
            return 1.0  # Everything is novel at first
        
        current = gray.flatten()
        
        # Compare to average of recent frames
        history_array = np.array(list(self._frame_history))
        mean_history = np.mean(history_array, axis=0)
        
        # Cosine distance from history mean
        dot = np.dot(current, mean_history)
        norm_c = np.linalg.norm(current) + 1e-8
        norm_h = np.linalg.norm(mean_history) + 1e-8
        cosine_sim = dot / (norm_c * norm_h)
        
        # Convert similarity to novelty (1 - similarity)
        novelty = 1.0 - max(0.0, cosine_sim)
        
        # Scale up — most scenes are fairly similar (cosine > 0.9)
        return min(1.0, novelty * 5.0)

    def _compute_complexity(self, gray: np.ndarray) -> float:
        """
        Edge density — how visually complex is the scene?
        A blank wall = low complexity, a bookshelf = high complexity.
        Simple Sobel-like gradient computation.
        """
        # Horizontal and vertical gradients
        gx = np.diff(gray, axis=1)  # horizontal edges
        gy = np.diff(gray, axis=0)  # vertical edges
        
        # Edge magnitude
        edge_h = float(np.mean(np.abs(gx)))
        edge_v = float(np.mean(np.abs(gy)))
        
        # Combined edge density
        edge_density = (edge_h + edge_v) / 2.0
        
        # Normalize: flat scene ~0.01, complex scene ~0.1+
        complexity = min(1.0, edge_density * 8.0)
        return complexity

    def _compute_luminance_change(self, gray: np.ndarray) -> float:
        """
        Sudden brightness change — like pupillary light reflex.
        Negative = went dark (potentially scary), positive = went bright.
        """
        current_lum = float(np.mean(gray))
        change = current_lum - self._ema_luminance
        
        # Scale: typical indoor variance is ~0.01-0.05
        return np.clip(change * 5.0, -1.0, 1.0)

    def get_stats(self) -> Dict:
        return {
            "frames_analyzed": self._frames_analyzed,
            "avg_motion": round(self._ema_motion, 4),
            "avg_complexity": round(self._ema_complexity, 4),
            "avg_luminance": round(self._ema_luminance, 4),
        }
