"""
Genesis Mind — The Eyes

The visual perception system. This is how Genesis sees the world.

It opens the laptop webcam, captures frames, detects meaningful changes
in the visual field, and converts what it sees into mathematical
representations (embeddings) that can be stored in memory and compared
to other things it has seen before.

Like a newborn's eyes — it sees everything but understands nothing.
Understanding comes later, when the cortex binds what it sees to what
it hears and what it is taught.
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger("genesis.senses.eyes")


@dataclass
class VisualPercept:
    """
    A single moment of seeing.

    Contains the raw image, its mathematical embedding, and metadata
    about when and how it was captured.
    """
    image: np.ndarray                       # Raw image (H, W, 3) BGR
    embedding: Optional[np.ndarray] = None  # CLIP embedding (1, 512)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""                   # Human-readable description (if available)
    motion_score: float = 0.0               # How much changed since last frame
    is_significant: bool = False            # Whether this percept is worth remembering


class Eyes:
    """
    The visual perception system of Genesis.

    Opens the laptop's webcam and observes the world. Generates
    mathematical embeddings of what it sees using a lightweight
    CLIP vision model.
    """

    def __init__(self, camera_index: int = 0, image_size: Tuple[int, int] = (224, 224),
                 motion_threshold: float = 0.05):
        self.camera_index = camera_index
        self.image_size = image_size
        self.motion_threshold = motion_threshold

        self._camera = None
        self._clip_model = None
        self._clip_preprocess = None
        self._tokenizer = None
        self._last_frame = None
        self._device = "cpu"  # Always CPU — we don't need a GPU for perception

        logger.info("Eyes initialized (camera_index=%d, size=%s)", camera_index, image_size)

    def open(self):
        """Open the eyes — activate the webcam."""
        import cv2

        self._camera = cv2.VideoCapture(self.camera_index)
        if not self._camera.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.camera_index}. "
                "Ensure a webcam is connected and accessible."
            )
        logger.info("Eyes opened — camera %d is active", self.camera_index)

    def close(self):
        """Close the eyes — release the webcam."""
        if self._camera is not None:
            self._camera.release()
            self._camera = None
            logger.info("Eyes closed")

    def _load_clip_model(self):
        """Lazy-load the CLIP vision model (only when first needed)."""
        if self._clip_model is not None:
            return

        logger.info("Loading vision model (CLIP ViT-B/32) onto CPU...")
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self._device
        )
        model.eval()
        self._clip_model = model
        self._clip_preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        logger.info("Vision model loaded successfully")

    def _compute_motion(self, frame: np.ndarray) -> float:
        """
        Compute how much the visual field has changed since the last frame.

        Returns a float between 0 (identical) and 1 (completely different).
        This prevents the system from processing identical frames and wasting CPU.
        """
        if self._last_frame is None:
            return 1.0  # First frame is always significant

        # Convert to grayscale and compute absolute difference
        import cv2
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_previous = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        diff = np.mean(np.abs(gray_current - gray_previous))
        return float(diff)

    def look(self) -> Optional[VisualPercept]:
        """
        Take a single look at the world.

        Captures a frame from the webcam, checks if anything meaningful
        has changed, and if so, returns a VisualPercept with the image data.
        """
        import cv2

        if self._camera is None:
            self.open()

        ret, frame = self._camera.read()
        if not ret:
            logger.warning("Failed to capture frame from camera")
            return None

        # Resize for processing
        frame_resized = cv2.resize(frame, self.image_size)

        # Check motion
        motion = self._compute_motion(frame_resized)
        is_significant = motion > self.motion_threshold

        percept = VisualPercept(
            image=frame_resized,
            motion_score=motion,
            is_significant=is_significant,
        )

        self._last_frame = frame_resized.copy()
        return percept

    def embed(self, percept: VisualPercept) -> np.ndarray:
        """
        Convert what the eyes see into a mathematical representation.

        This embedding is a 512-dimensional vector that captures the
        *essence* of the image — what objects are in it, their shapes,
        colors, and spatial relationships.

        Two images of the same object will have similar embeddings,
        even if taken from different angles or lighting conditions.
        """
        import torch
        from PIL import Image

        self._load_clip_model()

        # Convert BGR numpy array to PIL Image
        image_rgb = percept.image[:, :, ::-1]  # BGR -> RGB
        pil_image = Image.fromarray(image_rgb)

        # Preprocess and compute embedding
        image_tensor = self._clip_preprocess(pil_image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self._clip_model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize

        percept.embedding = embedding.cpu().numpy().flatten()
        return percept.embedding

    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert a word/phrase into the same embedding space as images.

        This is the key to multimodal binding: the text "dog" and an
        image of a dog will have similar embeddings in CLIP space.
        """
        import torch

        self._load_clip_model()

        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            text_embedding = self._clip_model.encode_text(tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        return text_embedding.cpu().numpy().flatten()

    def show_preview(self, duration_sec: float = 10.0):
        """Show a live preview of what the eyes see (for debugging/demo)."""
        import cv2

        if self._camera is None:
            self.open()

        start = time.time()
        logger.info("Showing camera preview for %.1f seconds...", duration_sec)

        while time.time() - start < duration_sec:
            ret, frame = self._camera.read()
            if not ret:
                break
            cv2.imshow("Genesis Eyes — Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# Standalone test — run with: python -m genesis.senses.eyes
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — Eyes Test")
    print("Opening camera... Press 'q' to stop.")
    print("=" * 60)

    with Eyes() as eyes:
        eyes.show_preview(duration_sec=30.0)
        print("\nTaking a snapshot and computing embedding...")
        percept = eyes.look()
        if percept:
            embedding = eyes.embed(percept)
            print(f"  Image shape: {percept.image.shape}")
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
            print(f"  Motion score: {percept.motion_score:.4f}")
            print("Eyes test PASSED ✓")
        else:
            print("ERROR: Could not capture frame!")
