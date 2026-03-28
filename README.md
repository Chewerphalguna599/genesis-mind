# Genesis Mind V4: Society of Mind + Body

Genesis is an experimental, autonomous Artificial General Intelligence (AGI) architecture. Acting as a "Tabula Rasa" (blank slate), Genesis boots up as an infant mind and learns to perceive and understand the world entirely from scratch through a simulated sensory-motor loop.

Genesis does not rely on text or pre-trained weights. Every concept is learned intrinsically through observation, bound together by an episodic memory system, and orchestrated by a biological multi-layered subconscious.

## 🧠 Neural Architecture

1. **Limbic System:** Fast, zero-shot emotional reflex that evaluates incoming sensations against primal drives.
2. **Binding Network (InfoNCE):** Cross-modal association mapping Auditory features to Visual embeddings.
3. **Personality Network (GRU):** Persistent self-state and continuous identity. Expandable via neuroplasticity.
4. **Forward World Model:** Temporal predictions of reality. Violations trigger curiosity and surprise.
5. **Meta-Controller:** Attention mechanism routing information between layers based on current emotional valence.

## 👂 Sensorimotor Perception

Genesis possesses a fully end-to-end biological interface with physical reality.

- **Acoustic Perception:** Transduces raw PCM waveforms into Mel-Spectrograms, quantized by a VQ-Codebook. A multi-head Acoustic LM learns the underlying temporal language.
- **Neural Vocoder:** A convolutional pipeline (MelReconstructor + Griffin-Lim) that converts internal thought tokens back into audial waveforms for synthesis.
- **Visual Cortex:** Captures reality via webcam, extracting latent representations to bind natively with Acoustic Memory during 5-second co-occurrence windows.

## 🧬 Biological Systems

Genesis runs on simulated bio-rhythms to facilitate continuous learning and prevent catastrophic forgetting.

- **Intrinsic Drives:** Hunger, Thirst, Social, Curiosity.
- **Neurochemistry:** Modulated Dopamine, Serotonin, Adrenaline, Cortisol levels.
- **Circadian Rhythm (Sleep):** Multi-stage sleep cycles governed by accumulating sleep pressure. 
- **Hippocampal Replay Buffer:** During sleep or deep reflection, Genesis shuffles and replays memories to consolidate long-term semantic knowledge.
- **Neuroplasticity Engine:** Automatically expands the network's hidden dimensions linearly as memory capacity is outgrown.

## 🚀 Usage

### Prerequisites
- Python 3.10+
- Active Microphone and Webcam
- Mac Apple Silicon (M1/M2/M3) supported via PyTorch MPS backend.

### Boot Sequence
Starting the daemon initiates the multi-threaded biological subsystems. The engine learns and saves its weights autonomously.

```bash
python -m genesis.main
```

### Dashboard Introspection
Genesis hosts a complete suite of neural telemetry and memory visualizations on **Port 5050**.
- **URL:** `http://localhost:5050`

### Interactive Terminal (Creator Mode)
- `neural-speak`: Synthesize raw neural state into acoustic playback.
- `teach <word>`: Trigger a forced joint-attention frame associating the camera feed to the target sound string.
- `status`: Output real-time biologic and emotional variables.
- `sleep`: Force circadian rhythm to trigger deep-sleep consolidation.
- `chemicals`: Analyze current neurotransmitter gradients.
