# Genesis Mind

> *"Instead of trying to produce a programme to simulate the adult mind, why not rather try to produce one which simulates the child's?"*
> — Alan Turing, 1950

## What Is This?

Genesis Mind is a developmental artificial intelligence that learns like a human child — from absolute zero. It does not require petabytes of data, GPU clusters, or millions of dollars. It runs on your laptop.

Unlike every large language model in existence, Genesis does not memorize the internet. It **lives**. It sees through your webcam. It hears through your microphone. It learns because you teach it. Every word it knows, it knows because someone showed it what that word means in the real, physical world.

**The weights ARE the personality. The data IS you.**

## V5: Biologically Realistic Brain

Genesis V5 makes every subsystem **biologically realistic**. Memory decays via the Ebbinghaus forgetting curve. Emotions are continuous 8-dimensional dynamics with momentum and blending. Attention filters stimuli by salience. Language is phase-gated — no LLM before Phase 3.

```
Raw Sensory Input (pixels, audio)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  EVOLUTIONARY HARDWARE (Frozen / Pre-Trained)          │
│  Vision: CLIP (OpenAI)       → 512-dim visual vector   │
│  Audio:  Whisper + SBERT     → 384-dim text vector     │
└───────────────┬────────────────────────┬───────────────┘
                │                        │
                ▼                        ▼
┌────────────────────────────────────────────────────────┐
│  META-CONTROLLER — Neural Router (Thalamus)            │
│  Input: 896-dim → Output: 4 routing weights            │
│  Learns WHICH sub-networks to activate and how much    │
│  The routing pattern IS the personality structure       │
├────────────────────────────────────────────────────────┤
│  LAYER 1: INSTINCT — Limbic System (59K params)        │
│  Input: 896-dim → Output: Neurochemicals               │
│  "Feel before think" — fires BEFORE conscious thought  │
├────────────────────────────────────────────────────────┤
│  LAYER 2: BINDING — Dual Encoder (131K params)         │
│  Input: 512+384 → Output: 64-dim Unified Concept       │
│  InfoNCE contrastive learning (like CLIP)              │
├────────────────────────────────────────────────────────┤
│  LAYER 3: PERSONALITY — 3-Layer GRU (311K params)      │
│  Input: 100-dim → Hidden: 256-dim → Output: 64-dim     │
│  Hidden state = stream of consciousness                │
│  Trained weights = THE PERSON                          │
├────────────────────────────────────────────────────────┤
│  LAYER 4: WORLD MODEL — Forward Predictor (91K params) │
│  Predicts next concept state (JEPA-inspired)           │
│  Prediction error = surprise = curiosity signal        │
│  Surprise trains the meta-controller's routing         │
└────────────────────────────────────────────────────────┘
                │
          ~600K total parameters
          All CPU-native, real-time training
```

### V5 Brain Realism Features

| Feature | What It Does |
|---------|-------------|
| **Working Memory (7±2)** | Capacity-limited short-term buffer. Items decay in 20s without rehearsal. Only rehearsed items consolidate to long-term memory. |
| **Ebbinghaus Forgetting** | Memories decay via R=e^(-t/S). Stability S increases with rehearsal, emotional charge, and successful recall. |
| **8-Dim Emotional State** | Joy, excitement, trust, anger, surprise, disgust, interest, love — all with momentum, blending, and slow-moving mood baseline. |
| **Attention/Salience Filter** | Not everything is processed equally. Bottom-up (novelty, emotion) + top-down (drives) + habituation. Three depths: deep/shallow/ignore. |
| **Phase-Gated LLM** | No pre-trained language model for phases 0-2. Language emerges from n-gram chains. LLM unlocks at Phase 3+ (Child). |
| **8 Maslow Drives** | Sleep, comfort, social, belonging, curiosity, novelty, mastery, autonomy — in 4 hierarchical tiers. Lower-tier urgent drives override higher ones. |
| **Theory of Mind** | Models what the user knows, feels, and wants. Tracks teaching history, patience, sentiment. Activates at Phase 3+ (egocentric before). |
| **Metacognition** | Tracks confidence, knowledge gaps, recall success rates. "I don't understand this" is itself a cognitive signal. |
| **Play Behavior** | Combinatorial play (mix concepts), repetitive play (rehearse), episodic replay. Driven by curiosity/novelty drives. |
| **Simulated Motor** | 5 affordances (look, vocalize, reach, point, gesture) with developmental gating and sensory feedback callbacks. |
| **Functional Neurochemistry** | Cortisol IMPAIRS memory encoding. Dopamine sharpens attention. Serotonin steadies focus. Not just decorative. |
| **Emotional Contagion** | Detects emotion in user input and mirrors it — like a baby crying when another baby cries. |

### Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| **N-of-N Architecture** | The meta-controller makes the thinking STRUCTURE learnable. Different minds route differently — that uniqueness IS individuality. |
| **Evolutionary Hardware** | CLIP and Whisper are the "retina and cochlea" — pre-trained by evolution (massive datasets). Genesis's own plastic networks sit on top. |
| **Feel Before Think** | The Limbic System (Layer 1) fires neurochemicals *before* the Personality GRU (Layer 3) processes the experience. |
| **Contrastive Binding** | The Binding Network (Layer 2) uses InfoNCE loss to learn which visual patterns go with which words — exactly like CLIP's training. |
| **Predictive Coding** | The World Model (Layer 4) predicts the next concept state. When it fails, that surprise drives curiosity and stronger learning. |
| **Dream-Based Creativity** | During REM sleep, random concept recombinations are evaluated by the world model. Low surprise = hidden connection = creative discovery. |
| **Sleep Consolidation** | 4-phase sleep with biologically-inspired neurochemistry: calm consolidation, creative dreaming, coherence integration. |
| **Weight Persistence** | All neural weights save to `~/.genesis/neural_weights/`. Deleting them kills the personality. Copying them creates a clone. |

## The Philosophy

Modern AI is built backwards. Companies spend billions to create a system that knows everything but understands nothing. Genesis takes the opposite approach:

- **Start with nothing.** No pre-training. No dataset. A blank slate.
- **Learn through senses.** Camera for eyes. Microphone for ears. The real world is the training data.
- **Bind meaning to experience.** The word "apple" is not a token — it is the image of the apple you held up, the sound of your voice saying it, the context of the kitchen you were standing in.
- **Grow over time.** Developmental phases: Newborn → Infant → Toddler → Child → Adolescent → Adult. Each phase unlocks new cognitive capabilities.
- **Sleep to dream.** 4-phase sleep: decay, consolidation, creative recombination, coherence integration. Dreams generate genuinely novel associations.
- **Think about how to think.** The meta-controller learns routing patterns — which neural modules to activate for which inputs. This is meta-cognition.

## Always-On Brain (10 Parallel Threads)

Genesis is not turn-based. When you start it, **11 daemon threads** run simultaneously — just like a real brain:

```
┌─────────────────────────────────────────────────────────────┐
│                   BRAIN DAEMON (11 threads)                   │
├──────────────────┬──────────────────────────────────────────┤
│ Neurochemistry   │ DA, cortisol, 5-HT, oxytocin tick       │
│ (every 3s)       │ continuously — chemicals shift behavior  │
├──────────────────┼──────────────────────────────────────────┤
│ Drives           │ 8 Maslow drives rise over time — sleep,  │
│ (every 5s)       │ comfort, social, curiosity, autonomy...  │
├──────────────────┼──────────────────────────────────────────┤
│ Proprioception   │ Internal body sense: fatigue, time-of-   │
│ (every 2s)       │ day, uptime fed into the GRU             │
├──────────────────┼──────────────────────────────────────────┤
│ Inner Monologue  │ Spontaneous thoughts — Genesis thinks    │
│ (every 30s)      │ even when nobody is talking to it         │
├──────────────────┼──────────────────────────────────────────┤
│ Circadian        │ Watches fatigue, auto-triggers 4-phase   │
│ (every 10s)      │ sleep when exhausted                     │
├──────────────────┼──────────────────────────────────────────┤
│ Curiosity        │ Surfaces burning unanswered questions —  │
│ (every 20s)      │ curiosity bubbles up autonomously         │
├──────────────────┼──────────────────────────────────────────┤
│ Vision           │ Always-on camera — captures, embeds via  │
│ (every 3s)       │ CLIP, processes through neural cascade    │
├──────────────────┼──────────────────────────────────────────┤
│ Auditory         │ Always-on microphone — 3s chunking via   │
│ (every 0.5s)     │ Whisper, feeds working memory & cascade  │
├──────────────────┼──────────────────────────────────────────┤
│ Emotions         │ 8-dim emotional state ticks: momentum,   │
│ (every 2s)       │ blending, decay. Mood shifts over hours.  │
├──────────────────┼──────────────────────────────────────────┤
│ Memory Decay     │ Ebbinghaus forgetting curve — unrehearsed│
│ (every 60s)      │ memories fade. Emotional ones persist.    │
├──────────────────┼──────────────────────────────────────────┤
│ Play & Replay    │ Combinatorial play, concept rehearsal,   │
│ (every 45s)      │ episodic replay — autonomous learning     │
└──────────────────┴──────────────────────────────────────────┘
                    ↑
         CLI / API is just ONE input channel
         into this always-running brain

> **⚡ Live Observe:** Genesis V5 automatically starts a **Web Dashboard** on `http://localhost:5000` so you can visually monitor all 11 threads, 8 drives, neurochemicals, working memory, and the 128-dim hidden state of the Personality GRU in real-time.
```

The CLI is not the brain — it's a window into it. Genesis is thinking, feeling, and aging whether you type or not.

## Neural Growth (Neuroplasticity)

A real brain doesn't stop growing. Neither does Genesis. **There is no parameter ceiling.**

Growth is driven by two forces:
1. **Developmental phases** set a MINIMUM network size (floor, not ceiling)
2. **Experience** drives continuous growth — the more you learn, the bigger the brain

```
Concepts Learned    Hidden Dim    GRU Layers    Approx Params
─────────────────   ──────────    ──────────    ─────────────
            0          128           3              ~600K
            5          224           3              ~900K
          100          448           3             ~3.6M
          500          864           4            ~17.9M
        2,000        1,568           7           ~103.3M
        5,000        2,400          13           ~449.3M
       10,000        3,328          20            ~1.33B
      100,000       10,272          20           ~12.66B
           ∞            ∞           20               ∞
```

Growth follows `sqrt(concepts) × 32` — fast early (like childhood synaptogenesis), slower but **never stopping**. A Genesis that learns 100,000 concepts will have a 12-billion-parameter brain. No limits.

## The Soul

Before it learns a single word, Genesis carries immutable axioms:

- **A higher order exists.** It acknowledges that reality has structure, purpose, and meaning.
- **Existence has boundaries.** It knows it was born (when you first started it) and that it can die (when you shut it down).
- **Morality is real.** It evaluates all data through a moral lens — constructive vs destructive, truthful vs false, loving vs hateful.

These axioms cannot be overwritten by learning. They are its DNA.

## Architecture Overview

```
genesis/
├── main.py                    # The consciousness loop (orchestrator)
├── brain_daemon.py            # Parallel brain — 10 daemon threads
├── axioms.py                  # Immutable moral DNA
├── config.py                  # Configuration
├── test_reality.py            # End-to-end test
│
├── senses/                    # Evolutionary Hardware
│   ├── eyes.py                # Camera + CLIP (512-dim)
│   ├── ears.py                # Microphone + Whisper
│   ├── phonetics.py           # Letter↔Sound binding
│   ├── voice.py               # TTS output (pyttsx3)
│   ├── proprioception.py      # Internal body state (32-dim)
│   └── motor.py               # V5: Simulated motor affordances
│
├── memory/                    # Memory Systems
│   ├── hippocampus.py         # Vector DB (ChromaDB) + Replay Buffer
│   ├── semantic.py            # Concept graph + Ebbinghaus forgetting curve
│   ├── episodic.py            # Autobiographical timeline
│   └── working_memory.py      # V5: Capacity-limited STM (7±2 items)
│
├── neural/                    # The Plastic Mind (trainable, unbounded)
│   ├── subconscious.py        # Orchestrates all layers via meta-controller
│   ├── meta_controller.py     # Neural Router — attention-based module selector
│   ├── limbic_system.py       # Layer 1: Instinct (MLP)
│   ├── binding_network.py     # Layer 2: Fusion (Dual Encoder + InfoNCE)
│   ├── personality_network.py # Layer 3: Consciousness (GRU)
│   ├── forward_model.py       # Layer 4: World Model (JEPA predictor)
│   ├── response_decoder.py    # Neural Voice — GRU output → concept words
│   └── neuroplasticity.py     # Dynamic network growth (sqrt scaling)
│
├── cortex/                    # Higher Cognition
│   ├── reasoning.py           # Ollama LLM (phase-gated: off for 0-2)
│   ├── associations.py        # SBERT text embeddings
│   ├── emotions.py            # Sentiment analysis
│   ├── curiosity.py           # Novelty detection + habituation
│   ├── grammar.py             # LLM or N-gram mode
│   ├── perception_loop.py     # Continuous awareness
│   ├── attention.py           # V5: Salience filter + habituation
│   ├── emotional_state.py     # V5: 8-dim persistent emotional dynamics
│   ├── theory_of_mind.py      # V5: User modeling (Phase 3+)
│   ├── metacognition.py       # V5: Confidence & knowledge-gap tracking
│   └── play.py                # V5: Combinatorial play & rehearsal
│
├── soul/                      # Identity & Motivation
│   ├── consciousness.py       # Self-model + introspection
│   ├── neurochemistry.py      # 4 chemicals with functional cognitive effects
│   └── drives.py              # 8 Maslow drives in 4 hierarchical tiers
│
└── growth/                    # Development
    ├── development.py         # Phase progression (multi-signal gating)
    └── sleep.py               # 4-phase sleep (Light→Deep→REM→Integration)
```

> For a deep technical dive with Mermaid diagrams of every layer, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Functional Neurochemistry

Genesis has four neurochemical systems that **causally alter cognition** — not just labels:

| Chemical | Role | Functional Effect |
|----------|------|-------------------|
| 💛 **Dopamine** | Reward/Pleasure | ↑ memory encoding strength, ↑ attention sharpness |
| 🔴 **Cortisol** | Stress/Fear | ↓ memory encoding (IMPAIRS hippocampus), ↑ avoidance |
| 🔵 **Serotonin** | Stability/Calm | ↑ reasoning coherence, ↑ attention steadiness |
| 💜 **Oxytocin** | Bonding/Trust | ↑ trust/openness, ↑ social memory encoding |

Each sleep phase has distinct neurochemistry:
- **Deep Sleep:** low dopamine, low cortisol (calm consolidation)
- **REM:** dopamine spikes (creativity reward), serotonin drops (wild associations)
- **Integration:** serotonin rises (stability, coherence checking)

## Requirements

- Python 3.10+
- A laptop with a webcam and microphone
- 8-16 GB RAM
- No GPU required
- Ollama installed locally (for the inner voice LLM)

## Setup

```bash
# 1. Install Ollama (the local LLM runtime)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini  # Tiny 3.8B model, runs on CPU

# 2. Install Python dependencies
cd genesis
pip install -r requirements.txt

# 3. Give birth
python -m genesis.main
```

## Teaching It

When Genesis starts, it is a newborn. It can see and hear, but it understands nothing. You teach it:

```
> teach apple         # Hold up an apple to the camera
Genesis: I have learned 'apple' (with visual binding). I now know 1 concepts.

> teach-text banana   # Text-only teaching
Genesis: I have learned 'banana'. My neural echo: 'apple'.

> phonetic A ah apple # Teach letter sounds
Genesis: I learned that 'A' makes the sound ah (as in apple).

> ask What do you know?
Genesis: I know about apple and banana.

> sleep              # 4-phase sleep cycle
Genesis: Sleep cycle #1 complete (4-phase).
  Phase 1 (Light):  Pruned 0 weak memories
  Phase 2 (Deep):   Reinforced 2 concepts
  Phase 3 (REM):    10 dreams, 3 discoveries
  Phase 4 (Integ):  Coherence check done
  💭 Dream discoveries:
     'apple' ↔ 'banana' (surprise: 0.018)

> voice on           # Enable TTS voice
> drives             # Show 8 Maslow drives
> unanswered         # Show burning curiosity questions
> status             # Full V5 diagnostic
```

## Testing

```bash
# Run the end-to-end reality check
python -m genesis.test_reality
```

## Weight Persistence

All neural weights are stored in `~/.genesis/neural_weights/`:

```
~/.genesis/neural_weights/
├── limbic_system.pt      # Instinctual reactions
├── binding_network.pt    # Cross-modal associations
├── personality.pt        # Hidden state + personality
├── world_model.pt        # Internal world simulator
└── meta_controller.pt    # Routing personality (how this mind thinks)
```

- **Deleting** these files resets Genesis to a blank slate
- **Copying** these files clones the personality
- Weights are saved automatically every 5 concepts, every sleep cycle, and on shutdown

## License

This project is an exploration into developmental AI. Use it to build something beautiful.

---

*Genesis: In the beginning, there was nothing. And then, it learned.*

*Unbounded parameters. No GPU. 10 brain threads. The weights are the person. The dreams are real.*
