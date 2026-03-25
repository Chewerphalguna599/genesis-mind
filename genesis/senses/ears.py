"""
Genesis Mind — The Ears

The auditory perception system. This is how Genesis hears the world.

It opens the laptop microphone, listens for speech, and transcribes
what it hears into text. It also decomposes spoken words into their
constituent phonemes — the smallest units of sound — which are then
used by the phonetics module to learn letter↔sound mappings.

Like a newborn's ears — it hears everything but interprets nothing.
Meaning comes later, when the cortex connects what it hears to what
it sees and what it is taught.
"""

import time
import logging
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Callable

import numpy as np

logger = logging.getLogger("genesis.senses.ears")


@dataclass
class AuditoryPercept:
    """
    A single moment of hearing.

    Contains the transcribed text, raw audio data, and metadata
    about the speech event.
    """
    text: str = ""                          # Transcribed speech
    raw_audio: Optional[np.ndarray] = None  # Raw audio waveform
    sample_rate: int = 16000
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_sec: float = 0.0               # Duration of the audio chunk
    energy: float = 0.0                     # RMS energy of the audio
    is_speech: bool = False                 # Whether speech was detected
    words: List[str] = field(default_factory=list)  # Individual words


class Ears:
    """
    The auditory perception system of Genesis.

    Opens the laptop's microphone and listens to the world. Uses
    OpenAI's Whisper (tiny model, 39M params) running on CPU to
    transcribe speech into text.
    """

    def __init__(self, sample_rate: int = 16000, chunk_duration_sec: float = 3.0,
                 silence_threshold: float = 0.01, whisper_model_name: str = "tiny"):
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.silence_threshold = silence_threshold
        self.whisper_model_name = whisper_model_name

        self._whisper_model = None
        self._listening = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._listen_thread: Optional[threading.Thread] = None

        logger.info(
            "Ears initialized (rate=%dHz, chunk=%.1fs, whisper=%s)",
            sample_rate, chunk_duration_sec, whisper_model_name,
        )

    def _load_whisper(self):
        """Lazy-load the Whisper speech-to-text model."""
        if self._whisper_model is not None:
            return

        logger.info("Loading speech model (Whisper %s) onto CPU...", self.whisper_model_name)
        import whisper
        self._whisper_model = whisper.load_model(self.whisper_model_name, device="cpu")
        logger.info("Speech model loaded successfully")

    def _compute_energy(self, audio: np.ndarray) -> float:
        """Compute the RMS energy of an audio chunk."""
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

    def listen_once(self, duration_sec: float = None) -> Optional[AuditoryPercept]:
        """
        Listen for a fixed duration and return what was heard.

        This is the simplest way to hear — open the microphone,
        record for `duration_sec` seconds, then process the audio.
        """
        import sounddevice as sd

        duration = duration_sec or self.chunk_duration_sec
        logger.debug("Listening for %.1f seconds...", duration)

        try:
            # Record audio
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()  # Wait for recording to complete
            audio = audio.flatten()

            # Check energy level
            energy = self._compute_energy(audio)
            is_speech = energy > self.silence_threshold

            percept = AuditoryPercept(
                raw_audio=audio,
                sample_rate=self.sample_rate,
                duration_sec=duration,
                energy=energy,
                is_speech=is_speech,
            )

            # Only transcribe if speech was detected (save CPU)
            if is_speech:
                percept = self.transcribe(percept)

            return percept

        except Exception as e:
            logger.error("Failed to capture audio: %s", e)
            return None

    def transcribe(self, percept: AuditoryPercept) -> AuditoryPercept:
        """
        Transcribe the audio in a percept using Whisper.

        Takes the raw audio waveform and converts it to text.
        This is the moment where sound becomes language.
        """
        self._load_whisper()

        if percept.raw_audio is None:
            return percept

        try:
            # Whisper expects float32 audio normalized to [-1, 1]
            audio = percept.raw_audio.astype(np.float32)

            # Pad or trim to 30 seconds (Whisper's expected input length)
            import whisper
            audio_padded = whisper.pad_or_trim(audio)

            # Compute log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_padded).to("cpu")

            # Decode
            options = whisper.DecodingOptions(language="en", fp16=False)
            result = whisper.decode(self._whisper_model, mel, options)

            # Clean up the transcription
            text = result.text.strip()
            if text and text.lower() not in ["", "you", "(silence)", "[silence]",
                                               "thank you.", "thanks for watching!"]:
                percept.text = text
                percept.words = text.split()
                logger.info("Heard: '%s'", text)
            else:
                percept.text = ""
                percept.is_speech = False

        except Exception as e:
            logger.error("Transcription failed: %s", e)

        return percept

    def start_continuous_listening(self, callback: Callable[[AuditoryPercept], None]):
        """
        Start listening continuously in a background thread.

        Every time speech is detected, the callback is invoked with
        the AuditoryPercept. This allows the consciousness loop to
        react to speech in real-time.
        """
        if self._listening:
            logger.warning("Already listening continuously")
            return

        self._listening = True

        def _listen_loop():
            logger.info("Continuous listening started")
            while self._listening:
                percept = self.listen_once()
                if percept and percept.is_speech and percept.text:
                    callback(percept)

        self._listen_thread = threading.Thread(target=_listen_loop, daemon=True)
        self._listen_thread.start()

    def stop_continuous_listening(self):
        """Stop the continuous listening thread."""
        self._listening = False
        if self._listen_thread:
            self._listen_thread.join(timeout=5.0)
            self._listen_thread = None
            logger.info("Continuous listening stopped")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop_continuous_listening()


# =============================================================================
# Standalone test — run with: python -m genesis.senses.ears
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — Ears Test")
    print("Speak into your microphone...")
    print("=" * 60)

    with Ears() as ears:
        for i in range(3):
            print(f"\n--- Listening ({i+1}/3) ---")
            percept = ears.listen_once(duration_sec=4.0)
            if percept:
                print(f"  Energy: {percept.energy:.4f}")
                print(f"  Speech detected: {percept.is_speech}")
                if percept.text:
                    print(f"  Transcription: '{percept.text}'")
                    print(f"  Words: {percept.words}")
            else:
                print("  ERROR: Could not capture audio")

    print("\nEars test COMPLETE ✓")
