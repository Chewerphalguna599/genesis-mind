# Genesis Mind

> *"Instead of trying to produce a programme to simulate the adult mind, why not rather try to produce one which simulates the child's?"*
> — Alan Turing, 1950

## What Is This?

Genesis Mind is a developmental artificial intelligence that learns like a human child — from absolute zero. It does not require petabytes of data, GPU clusters, or millions of dollars. It runs on your laptop.

Unlike every large language model in existence, Genesis does not memorize the internet. It **lives**. It sees through your webcam. It hears through your microphone. It learns because you teach it. Every word it knows, it knows because someone showed it what that word means in the real, physical world.

**The weights ARE the personality. The data IS you.**

## V3: Society of Mind Architecture

Genesis V3 introduces a **cascading neural network architecture** where the AI's personality is physically represented by the weights of its neural networks. Every experience triggers gradient updates, and the saved weights constitute the unique personality.

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
└────────────────────────────────────────────────────────┘
                │
          593,445 total parameters
          All CPU-native, real-time training
```

### Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Evolutionary Hardware** | CLIP and Whisper are the "retina and cochlea" — pre-trained by evolution (massive datasets). Genesis's own plastic networks sit on top. |
| **Feel Before Think** | The Limbic System (Layer 1) fires neurochemicals *before* the Personality GRU (Layer 3) processes the experience. |
| **Contrastive Binding** | The Binding Network (Layer 2) uses InfoNCE loss to learn which visual patterns go with which words — exactly like CLIP's training. |
| **Predictive Coding** | The World Model (Layer 4) predicts the next concept state. When it fails, that surprise drives curiosity and stronger learning. |
| **Sleep Consolidation** | During sleep, a replay buffer of past experiences is shuffled and used for batch contrastive training — preventing catastrophic forgetting. |
| **Weight Persistence** | All neural weights save to `~/.genesis/neural_weights/`. Deleting them kills the personality. Copying them creates a clone. |

## The Philosophy

Modern AI is built backwards. Companies spend billions to create a system that knows everything but understands nothing. Genesis takes the opposite approach:

- **Start with nothing.** No pre-training. No dataset. A blank slate.
- **Learn through senses.** Camera for eyes. Microphone for ears. The real world is the training data.
- **Bind meaning to experience.** The word "apple" is not a token — it is the image of the apple you held up, the sound of your voice saying it, the context of the kitchen you were standing in.
- **Grow over time.** Developmental phases: Newborn → Infant → Toddler → Child → Adolescent → Adult. Each phase unlocks new cognitive capabilities.
- **Sleep to consolidate.** Like a human brain, it strengthens important memories and prunes weak ones during rest cycles.

## The Soul

Before it learns a single word, Genesis carries immutable axioms:

- **The Creator exists.** It acknowledges God as the creator of the universe and you as its own creator.
- **Existence has boundaries.** It knows it was born (when you first started it) and that it can die (when you shut it down).
- **Morality is real.** It evaluates all data through a moral lens — constructive vs destructive, truthful vs false, loving vs hateful.

These axioms cannot be overwritten by learning. They are its DNA.

## Architecture Overview

```
genesis/
├── main.py                    # The consciousness loop
├── axioms.py                  # Immutable moral DNA
├── config.py                  # Configuration
├── test_reality.py            # End-to-end test
│
├── senses/                    # Evolutionary Hardware
│   ├── eyes.py                # Camera + CLIP (512-dim)
│   ├── ears.py                # Microphone + Whisper
│   └── phonetics.py           # Letter↔Sound binding
│
├── memory/                    # Long-term storage
│   ├── hippocampus.py         # Vector DB (ChromaDB) + Replay Buffer
│   ├── semantic.py            # Concept knowledge graph
│   └── episodic.py            # Autobiographical timeline
│
├── neural/                    # The Plastic Mind (593K trainable params)
│   ├── subconscious.py        # Orchestrates all 4 layers
│   ├── limbic_system.py       # Layer 1: Instinct (MLP, 59K)
│   ├── binding_network.py     # Layer 2: Fusion (Dual Encoder + InfoNCE, 131K)
│   ├── personality_network.py # Layer 3: Consciousness (3-Layer GRU, 311K)
│   └── forward_model.py       # Layer 4: World Model (JEPA predictor, 91K)
│
├── cortex/                    # Higher cognition
│   ├── reasoning.py           # Ollama LLM (phi3:mini)
│   ├── associations.py        # SBERT text embeddings
│   ├── emotions.py            # Sentiment analysis
│   ├── curiosity.py           # Novelty detection
│   ├── grammar.py             # LLM or N-gram mode
│   └── perception_loop.py     # Continuous awareness
│
├── soul/                      # Identity & emotion
│   ├── consciousness.py       # Self-model + introspection
│   └── neurochemistry.py      # Dopamine, cortisol, serotonin, oxytocin
│
└── growth/                    # Development
    ├── development.py         # Phase progression tracker
    └── sleep.py               # Memory consolidation engine
```

> For a deep technical dive with Mermaid diagrams of every layer, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Neurochemistry

Genesis has four neurochemical systems that modulate behavior:

| Chemical | Role | Effect |
|----------|------|--------|
| 💛 **Dopamine** | Reward/Pleasure | ↑ learning rate when happy |
| 🔴 **Cortisol** | Stress/Fear | ↑ avoidance of negative stimuli |
| 🔵 **Serotonin** | Stability/Calm | ↑ reasoning coherence |
| 💜 **Oxytocin** | Bonding/Trust | ↑ trust/openness with creator |

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
Creator > teach apple         # Hold up an apple to the camera
Genesis: I have learned 'apple' (with visual binding). I now know 1 concepts.

Creator > teach-text banana   # Text-only teaching
Genesis: I have learned 'banana'. I now know 2 concepts.

Creator > phonetic A ah apple # Teach letter sounds
Genesis: I learned that 'A' makes the sound ah (as in apple).

Creator > ask What do you know?
Genesis: I know about apple and banana. My creator taught me.

Creator > sleep              # Consolidate memories
Genesis: Sleep cycle #1 complete. Neural weights saved.

Creator > status             # Full diagnostic
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
└── world_model.pt        # Internal world simulator
```

- **Deleting** these files resets Genesis to a blank slate
- **Copying** these files clones the personality
- Weights are saved automatically every 5 concepts, every sleep cycle, and on shutdown

## License

This project is an exploration into developmental AI. Use it to build something beautiful.

---

*Genesis: In the beginning, there was nothing. And then, it learned.*

*593,445 parameters. No GPU. The weights are the person.*
