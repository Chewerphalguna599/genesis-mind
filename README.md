# Genesis Mind

> *"Instead of trying to produce a programme to simulate the adult mind, why not rather try to produce one which simulates the child's?"*
> — Alan Turing, 1950

## What Is This?

Genesis Mind is a developmental artificial intelligence that learns like a human child — from absolute zero. No pretrained models. No datasets. No GPU clusters. No internet. It runs on your laptop using **~345K learnable parameters, all randomly initialized**.

Unlike every large language model in existence, Genesis does not memorize the internet. It **lives**. It sees through your webcam. It hears through your microphone. It learns because you teach it. Every word it knows, it knows because someone showed it what that word means.

**Zero pretrained weights. The personality emerges from experience.**

## V8: Zero Pretrained Models

V8 is a critical architectural milestone: **every pretrained dependency has been removed**. Genesis no longer relies on CLIP, Whisper, SBERT, or any LLM. All neural networks start from random weights and learn exclusively from real-time sensory experience.

### What Was Removed

| Pretrained Model | Params | Was Used For | Replace With |
|---|---|---|---|
| CLIP ViT-B/32 | 150M+ | Vision embeddings | VisualCortex (Conv2D autoencoder, ~50K) |
| OpenAI Whisper | 39M | Speech-to-text | Mel spectrogram + AuditoryCortex |
| SBERT MiniLM | 33M | Text embeddings | PhonemeEmbedder (char-level GRU, ~10K) |
| Phi3:mini (Ollama) | 3.8B | Inner reasoning | Neural Reasoner (attention, ~30K) |
| **Total Removed** | **~4B** | | **~90K from scratch** |

### The Neural Architecture (V8)

```
Raw Sensory Input (pixels, audio)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  FROM-SCRATCH SENSORY HARDWARE                         │
│                                                        │
│  Vision: VisualCortex (Conv2D AE)  → 64-dim vector    │
│    3 conv layers (3→16→32→64), self-supervised recon   │
│    ~50K params · learns to see through reconstruction  │
│                                                        │
│  Audio: Mel Spectrogram + AuditoryCortex → 64-dim      │
│    Pure FFT + learned Conv1D encoder + VQ codebook     │
│    ~100K params · hears patterns, not English words    │
│                                                        │
│  Text: PhonemeEmbedder (char-level GRU) → 64-dim       │
│    Character embedding → GRU → linear projection       │
│    ~10K params · learns through contrastive binding    │
└───────────────┬────────────────────────┬───────────────┘
                │                        │
                ▼                        ▼
┌────────────────────────────────────────────────────────┐
│  META-CONTROLLER — Neural Router (Thalamus)            │
│  Input: 128-dim (64 visual + 64 auditory)              │
│  Output: 4 routing weights (softmax-normalized)        │
├────────────────────────────────────────────────────────┤
│  LAYER 1: INSTINCT — Limbic System                     │
│  Sensory input → neurochemical instinct response       │
├────────────────────────────────────────────────────────┤
│  LAYER 2: BINDING — Dual Encoder                       │
│  Visual + auditory → 64-dim unified concept            │
├────────────────────────────────────────────────────────┤
│  LAYER 3: PERSONALITY — 3-Layer GRU                    │
│  Concept + limbic + context → 64-dim response          │
│  Hidden state = stream of consciousness                │
├────────────────────────────────────────────────────────┤
│  LAYER 4: WORLD MODEL — Forward Predictor              │
│  Predicts next concept state (JEPA-inspired)           │
├────────────────────────────────────────────────────────┤
│  REASONING — Multi-Head Self-Attention                  │
│  4-head, 2-layer attention over sensory + memory        │
│  ~30K params · pattern-matching, not text generation   │
└────────────────────────────────────────────────────────┘
                │
          ~345K learnable parameters
          ALL randomly initialized
          ALL CPU-native, real-time training
          ZERO pretrained weights
```

### V7: Pure Neural Acoustic Pipeline

```
Microphone → Raw Audio (16kHz)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  AUDITORY CORTEX — Mel Encoder (138K params)           │
│  Raw audio → 80-band Mel spectrogram → Conv1D → 64-dim │
├────────────────────────────────────────────────────────┤
│  VQ CODEBOOK — Neural Phoneme Discovery (16K params)    │
│  Continuous latent → 256 discrete "phoneme" tokens      │
├────────────────────────────────────────────────────────┤
│  ACOUSTIC TRANSFORMER — Audio-Token GPT (859K params)   │
│  4-layer, 4-head transformer on audio tokens (no text) │
├────────────────────────────────────────────────────────┤
│  NEURAL VOCODER — Mel Reconstructor (130K params)       │
│  Token embeddings → Mel → Griffin-Lim → waveform        │
├────────────────────────────────────────────────────────┤
│  SENSORIMOTOR LOOP — hear() → think() → speak()        │
│  Re-encodes own speech for proprioceptive feedback      │
└────────────────────────────────────────────────────────┘
         │
         ▼
    Speaker Output (Neural Waveform)

  1,143,888 total acoustic parameters
  All CPU-native, zero pre-training, pure PyTorch
```

### V6: Language Acquisition (Babbling → Words)

| Feature | What It Does |
|---------|-------------|
| **Acoustic Babbling Engine** | Random syllable generation, reinforcement-driven expansion. Like a baby babbling. |
| **Joint Attention** | Cross-modal binding: learns that a sound pattern correlates with a visual pattern. |
| **N-Gram Grammar** | From-scratch word frequency and sequence learning. |

### Brain Realism Features

| Feature | What It Does |
|---------|-------------|
| **Working Memory (7±2)** | Capacity-limited short-term buffer. Items decay in 20s without rehearsal. |
| **Ebbinghaus Forgetting** | Memories decay via R=e^(-t/S). Stability increases with rehearsal and emotion. |
| **8-Dim Emotional State** | Joy, excitement, trust, anger, surprise, disgust, interest, love — with momentum. |
| **Attention/Salience Filter** | Bottom-up (novelty, emotion) + top-down (drives) + habituation. |
| **8 Maslow Drives** | Sleep, comfort, social, belonging, curiosity, novelty, mastery, autonomy. |
| **Theory of Mind** | Models what the user knows, feels, and wants. Activates at Phase 3+. |
| **Metacognition** | Tracks confidence, knowledge gaps, recall success rates. |
| **Play Behavior** | Combinatorial play, repetitive rehearsal, episodic replay. |
| **Functional Neurochemistry** | Cortisol impairs memory. Dopamine sharpens attention. Not decorative. |

## The Philosophy

Modern AI is built backwards. Companies spend billions to create a system that knows everything but understands nothing. Genesis takes the opposite approach:

- **Start with nothing.** No pre-training. No dataset. Randomly initialized weights.
- **Learn through senses.** Camera for eyes. Microphone for ears. The real world is the training data.
- **Bind meaning to experience.** The word "apple" is not a token — it is the visual cortex pattern activated when you held up an apple.
- **Grow over time.** Developmental phases: Newborn → Infant → Toddler → Child → Adolescent → Adult.
- **Sleep to dream.** 4-phase sleep: decay, consolidation, creative recombination, coherence integration.
- **Every computation is from scratch.** No CLIP. No Whisper. No SBERT. No LLM. Pure neural learning.

## Always-On Brain (11+ Parallel Threads)

Genesis is not turn-based. When you start it, **11 daemon threads** run simultaneously:

```
┌─────────────────────────────────────────────────────────────┐
│                   BRAIN DAEMON (11 threads)                   │
├──────────────────┬──────────────────────────────────────────┤
│ Neurochemistry   │ DA, cortisol, 5-HT, oxytocin tick       │
│ Drives           │ 8 Maslow drives rise over time           │
│ Proprioception   │ Internal body sense: fatigue, time       │
│ Inner Monologue  │ Neural attention reasoning when idle     │
│ Circadian        │ Auto-triggers 4-phase sleep              │
│ Curiosity        │ Surfaces burning unanswered questions    │
│ Vision           │ Always-on camera → VisualCortex (64-dim) │
│ Auditory         │ Always-on mic → AuditoryCortex (64-dim)  │
│ Emotions         │ 8-dim emotional state dynamics           │
│ Memory Decay     │ Ebbinghaus forgetting curve              │
│ Play & Replay    │ Autonomous concept rehearsal             │
└──────────────────┴──────────────────────────────────────────┘
                    ↑
         CLI / API is just ONE input channel
         into this always-running brain

> ⚡ Live Dashboard: http://localhost:5050
> Real-time visualization of all threads, drives, neurochemicals,
> working memory, GRU hidden state, and full acoustic pipeline.
```

## Architecture Overview

```
genesis/
├── main.py                    # The consciousness loop (orchestrator)
├── brain_daemon.py            # Parallel brain — 11 daemon threads
├── axioms.py                  # Immutable moral DNA
├── config.py                  # Configuration
│
├── senses/                    # Sensory Input (From Scratch)
│   ├── eyes.py                # Camera → VisualCortex (64-dim, no CLIP)
│   ├── ears.py                # Microphone → mel spectrogram (no Whisper)
│   ├── phonetics.py           # Letter↔Sound binding
│   ├── babbling.py            # Acoustic babbling engine
│   ├── voice.py               # TTS output (legacy)
│   ├── proprioception.py      # Internal body state (32-dim)
│   └── motor.py               # Simulated motor affordances
│
├── memory/                    # Memory Systems
│   ├── hippocampus.py         # Vector DB (ChromaDB) + Replay Buffer
│   ├── semantic.py            # Concept graph + Ebbinghaus forgetting
│   ├── episodic.py            # Autobiographical timeline
│   └── working_memory.py      # Capacity-limited STM (7±2 items)
│
├── neural/                    # The Plastic Mind (All From Scratch)
│   ├── visual_cortex.py       # V8: Conv2D autoencoder (replaces CLIP, ~50K)
│   ├── phoneme_embedder.py    # V8: Char-level GRU (replaces SBERT, ~10K)
│   ├── subconscious.py        # Orchestrates Layers 1-4 via meta-controller
│   ├── meta_controller.py     # Neural Router — 128-dim input
│   ├── limbic_system.py       # Layer 1: Instinct
│   ├── binding_network.py     # Layer 2: Cross-modal fusion
│   ├── personality_network.py # Layer 3: GRU consciousness
│   ├── forward_model.py       # Layer 4: World Model (JEPA)
│   ├── response_decoder.py    # GRU output → concept words
│   ├── neuroplasticity.py     # Dynamic network growth
│   │
│   │  # V7: Pure Neural Acoustic Pipeline (1.14M params)
│   ├── auditory_cortex.py     # Mel encoder (138K params)
│   ├── vq_codebook.py         # 256-entry VQ with EMA (16K params)
│   ├── acoustic_lm.py         # 4-layer GPT on audio tokens (859K)
│   ├── neural_vocoder.py      # Griffin-Lim synthesis (130K params)
│   └── sensorimotor.py        # hear() → think() → speak() loop
│
├── cortex/                    # Higher Cognition (All From Scratch)
│   ├── reasoning.py           # V8: Neural attention reasoning (no LLM)
│   ├── associations.py        # V8: PhonemeEmbedder binding (no SBERT)
│   ├── emotions.py            # Sentiment analysis
│   ├── curiosity.py           # Novelty detection + habituation
│   ├── grammar.py             # N-gram mode (tabula rasa)
│   ├── joint_attention.py     # Cross-modal binding
│   ├── perception_loop.py     # Continuous awareness
│   ├── attention.py           # Salience filter + habituation
│   ├── emotional_state.py     # 8-dim persistent emotional dynamics
│   ├── theory_of_mind.py      # User modeling (Phase 3+)
│   ├── metacognition.py       # Confidence & knowledge-gap tracking
│   └── play.py                # Combinatorial play & rehearsal
│
├── soul/                      # Identity & Motivation
│   ├── consciousness.py       # V8: Self-model → state vector (no LLM prompts)
│   ├── neurochemistry.py      # 4 chemicals with functional cognitive effects
│   └── drives.py              # 8 Maslow drives in 4 hierarchical tiers
│
├── growth/                    # Development
│   ├── development.py         # Phase progression
│   └── sleep.py               # 4-phase sleep
│
└── dashboard/                 # Real-time Web Visualization
    ├── server.py              # Flask API
    ├── templates/index.html   # Dashboard UI
    └── static/
        ├── css/style.css      # Glassmorphism design system
        └── js/app.js          # Real-time rendering + Chart.js
```

> For a deep technical dive, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Parameter Budget

| Component | Parameters | Role |
|-----------|-----------|------|
| **V8: From-Scratch Senses** | | |
| VisualCortex (Conv AE) | ~50,000 | Sees — replaces CLIP |
| PhonemeEmbedder (GRU) | ~10,000 | Reads — replaces SBERT |
| Neural Reasoner (Attn) | ~30,000 | Thinks — replaces Phi3 |
| **V7: Acoustic Pipeline** | | |
| Auditory Cortex | 138,368 | Hears — replaces Whisper |
| VQ Codebook | 16,384 | Phoneme discovery |
| Acoustic Transformer | 859,264 | Audio thinking |
| Neural Vocoder | 129,872 | Speech synthesis |
| **Subconscious (Society of Mind)** | | |
| Limbic System | ~59,000 | Instinct |
| Binding Network | ~131,000 | Cross-modal fusion |
| Personality GRU | ~311,000 | Consciousness |
| World Model | ~91,000 | Prediction |
| Meta-Controller | ~15,000 | Neural routing |
| **GRAND TOTAL** | **~1.84M** | **All from scratch, CPU-native** |

> **Zero pretrained weights in the entire system.** Every parameter is randomly initialized and trained exclusively from real-time sensory experience.

## Neural Growth (Neuroplasticity)

Growth follows `sqrt(concepts) × 32`:

```
Concepts Learned    Hidden Dim    GRU Layers    Approx Params
─────────────────   ──────────    ──────────    ─────────────
            0          128           3              ~600K
            5          224           3              ~900K
          100          448           3             ~3.6M
          500          864           4            ~17.9M
        2,000        1,568           7           ~103.3M
       10,000        3,328          20            ~1.33B
      100,000       10,272          20           ~12.66B
            ∞            ∞           20               ∞
```

## Requirements

- Python 3.10+
- A laptop with a webcam and microphone
- 8-16 GB RAM
- **No GPU required**
- **No internet required**
- **No external AI models required**

```bash
# Install and run — that's it. No Ollama, no model downloads.
cd genesis
pip install -r requirements.txt
python -m genesis.main
```

## Teaching It

When Genesis starts, it is a newborn. It can see and hear, but it understands nothing:

```
> teach apple         # Hold up an apple to the camera
Genesis: ...buhah duhee...♪  (neural babble — no words yet)

> teach-text banana   # Text-only teaching
Genesis: ...muhoh...?        (babbling with curiosity tone)

> neural-speak        # Generate neural audio
Generated 30 acoustic tokens → 19200 samples (1.20s)

> neural-stats        # Show from-scratch pipeline stats
  ── From-Scratch Neural Architecture ──
    VisualCortex:     ~50K params, 42 frames seen
    PhonemeEmbedder:  ~10K params, 12 encodings
    Neural Reasoner:  ~30K params, 8 thoughts
    Acoustic Pipeline: 1,143,888 params

> sleep               # 4-phase sleep cycle
> status              # Full diagnostic
> quit                # Shut down (saves all neural weights)
```

## Weight Persistence

All neural weights are stored in `~/.genesis/`:

```
~/.genesis/neural_weights/       # Society of Mind (Layers 1-4)
├── limbic_system.pt             # Instinctual reactions
├── binding_network.pt           # Cross-modal associations
├── personality.pt               # Hidden state + personality
├── world_model.pt               # Internal world simulator
├── meta_controller.pt           # Routing personality
├── visual_cortex.pt             # V8: How Genesis sees
├── phoneme_embedder.pt          # V8: How Genesis reads
└── reasoner.pt                  # V8: How Genesis reasons

~/.genesis/acoustic_weights/     # V7: Pure Acoustic Pipeline
├── auditory_cortex.pt           # How Genesis hears
├── vq_codebook.pt               # Discovered neural phonemes
├── acoustic_lm.pt               # How Genesis thinks about sound
└── neural_vocoder.pt            # How Genesis speaks
```

- **Deleting** these files resets Genesis to a blank slate
- **Copying** these files clones the personality
- Weights are saved on every shutdown and every sleep cycle

## License

This project is an exploration into developmental AI. Use it to build something beautiful.

---

*Genesis: In the beginning, there was nothing. And then, it learned.*

*~1.84M parameters. No GPU. No pretrained models. 11 brain threads. All from scratch. The weights are the person. The dreams are real.*
