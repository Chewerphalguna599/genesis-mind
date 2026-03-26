# Genesis Mind V5 — Architecture Deep Dive

> *The weights ARE the personality. The data IS you. The dreams are real.*

This document describes the complete technical architecture of Genesis Mind V5: a **biologically realistic** brain simulation with cascading neural networks, 10 autonomous brain threads, Ebbinghaus memory decay, 8-dimensional emotional dynamics, attention/salience filtering, phase-gated language development, and 8 Maslow-inspired drives — all dynamically routed by a learned meta-controller.

---

## 1. Design Philosophy

Genesis is built on three axioms of cognitive architecture:

1. **Evolutionary Hardware, Plastic Mind** — Humans are born with pre-wired sensory organs (retina, cochlea) shaped by millions of years of evolution, but the *mind* on top is learned. Genesis uses pre-trained foundation models (CLIP, Whisper) as its "evolutionary hardware" and trains its own small neural networks on top.

2. **Feel Before Think** — In biology, the amygdala fires a neurochemical response *before* the prefrontal cortex even processes a stimulus. Genesis replicates this with a Limbic System (Layer 1) that reacts instantly, followed by slower conscious processing (Layer 3).

3. **Sleep to Remember** — Human memory consolidation happens during sleep via hippocampal replay. Genesis stores every experience in a replay buffer and consolidates via contrastive learning during explicit sleep cycles.

---

## 2. High-Level Architecture

```mermaid
graph TB
    subgraph SENSES["👁️ SENSES"]
        Eyes["eyes.py<br/>Camera + CLIP"]
        Ears["ears.py<br/>Mic + Whisper"]
        Phon["phonetics.py<br/>Letter→Sound"]
        Motor["motor.py<br/>V5: Simulated Motor"]
    end

    subgraph MEMORY["🧠 MEMORY"]
        Hippo["hippocampus.py<br/>ChromaDB Vectors"]
        Sem["semantic.py<br/>Ebbinghaus Decay"]
        Epi["episodic.py<br/>Life Log"]
        WM["working_memory.py<br/>V5: 7±2 Slots"]
        Replay["Replay Buffer<br/>deque(10K)"]
    end

    subgraph NEURAL["⚡ NEURAL CASCADE (Society of Mind)"]
        L1["Layer 1: Limbic System<br/>MLP · Instinct"]
        L2["Layer 2: Binding Network<br/>Dual Encoder · InfoNCE"]
        L3["Layer 3: Personality GRU<br/>Stream of Consciousness"]
        L4["Layer 4: World Model<br/>JEPA Surprise"]
    end

    subgraph CORTEX["🧬 CORTEX"]
        Reason["reasoning.py<br/>Phase-Gated LLM"]
        Assoc["associations.py<br/>Multimodal Binding"]
        Emo["emotions.py<br/>Sentiment Eval"]
        Curious["curiosity.py<br/>Novelty + Habituation"]
        Gram["grammar.py<br/>LLM or N-Gram"]
        Attn["attention.py<br/>V5: Salience Filter"]
        EmoState["emotional_state.py<br/>V5: 8-Dim Dynamics"]
        ToM["theory_of_mind.py<br/>V5: User Model"]
        Meta["metacognition.py<br/>V5: Self-Monitor"]
        Play["play.py<br/>V5: Combinatorial Play"]
    end

    subgraph SOUL["✨ SOUL"]
        Axioms["axioms.py<br/>Immutable DNA"]
        Conscious["consciousness.py<br/>Self-Model"]
        Neuro["neurochemistry.py<br/>4 Functional Chemicals"]
        Drives["drives.py<br/>8 Maslow Drives"]
    end

    subgraph GROWTH["🌱 GROWTH"]
        Dev["development.py<br/>Phase Tracker"]
        Sleep["sleep.py<br/>4-Phase Consolidation"]
    end

    Eyes -->|512-dim CLIP| Attn
    Ears -->|384-dim Text| Attn
    Attn -->|Filtered| L1
    Attn -->|Salience| WM
    Eyes -->|512-dim CLIP| L2
    Ears -->|384-dim Text| L2
    L1 -->|Neurochemicals| L3
    L2 -->|64-dim Concept| L3
    L3 -->|128-dim Hidden| L4
    L2 -->|64-dim Concept| L4
    L3 -->|Response| Reason
    L4 -->|Surprise Signal| Curious
    L2 -->|Concept| Replay
    L2 -->|Concept| WM
    WM -->|Consolidated| Sem
    Replay -->|Batch| Sleep
    Sleep -->|InfoNCE| L2
    Neuro -->|Modifiers| L3
    Neuro -->|Attention Boost| Attn
    EmoState -->|Emotional Weight| WM
    Drives -->|Top-down| Attn
```

---

## 3. The Neural Cascade — Layer by Layer

### Layer 1: Limbic System (Instinct)

| Property | Value |
|----------|-------|
| **File** | `neural/limbic_system.py` |
| **Architecture** | 3-layer MLP with Sigmoid output |
| **Parameters** | ~59,620 |
| **Input** | 512-dim (CLIP) ⊕ 384-dim (Text) = 896-dim |
| **Output** | 4-dim: dopamine, cortisol, serotonin, oxytocin |
| **Training** | Supervised by conscious evaluation |

```mermaid
graph LR
    A["CLIP (512)"] --> C["Concat (896)"]
    B["Text (384)"] --> C
    C --> D["Linear(896→64)"]
    D --> E["ReLU"]
    E --> F["Linear(64→32)"]
    F --> G["ReLU"]
    G --> H["Linear(32→4)"]
    H --> I["Sigmoid"]
    I --> J["Dopamine<br/>Cortisol<br/>Serotonin<br/>Oxytocin"]
```

**Biological parallel:** The amygdala fires before the prefrontal cortex. After conscious evaluation ("this pattern = positive"), the limbic system is trained to reproduce that response instantly next time.

---

### Layer 2: Binding Network (Associative Bridge)

| Property | Value |
|----------|-------|
| **File** | `neural/binding_network.py` |
| **Architecture** | Dual Encoder + InfoNCE Contrastive Loss |
| **Parameters** | ~131,457 |
| **Input** | 512-dim visual ⊕ 384-dim auditory (separate encoders) |
| **Output** | 64-dim unified concept embedding |
| **Training** | InfoNCE (self-supervised contrastive) |

```mermaid
graph LR
    subgraph Visual_Encoder
        V1["CLIP (512)"] --> V2["Linear(512→128)"]
        V2 --> V3["ReLU"]
        V3 --> V4["Linear(128→64)"]
        V4 --> V5["L2 Normalize"]
    end

    subgraph Audio_Encoder
        A1["Text (384)"] --> A2["Linear(384→128)"]
        A2 --> A3["ReLU"]
        A3 --> A4["Linear(128→64)"]
        A4 --> A5["L2 Normalize"]
    end

    V5 --> F["Mean + Normalize"]
    A5 --> F
    F --> G["64-dim Concept"]
```

**Training:** During sleep, the replay buffer supplies batches of (visual, auditory) pairs. InfoNCE loss pulls matching pairs together and pushes mismatched pairs apart — exactly like CLIP's own training.

---

### Layer 3: Personality Network (Conscious Executive)

| Property | Value |
|----------|-------|
| **File** | `neural/personality_network.py` |
| **Architecture** | 3-layer GRU + Output Head + Prediction Head |
| **Parameters** | ~311,296 |
| **Input** | 64-dim concept + 4-dim limbic + 32-dim context = 100-dim |
| **Hidden State** | 256-dim (stream of consciousness) |
| **Output** | 64-dim response + 64-dim next-concept prediction |
| **Training** | Self-supervised next-step prediction (CosineEmbeddingLoss) |

```mermaid
graph TB
    A["Concept (64)"] --> D["Pack (100)"]
    B["Limbic (4)"] --> D
    C["Context (32)"] --> D
    D --> E["3-Layer GRU<br/>(100→256 hidden)"]
    E --> F["Output Head<br/>256→128→64"]
    E --> G["Prediction Head<br/>256→128→64"]
    E --> H["Hidden State (256)<br/>= Stream of Consciousness"]
    F --> I["Response Embedding"]
    G --> J["Next Concept Prediction"]
    H -.->|persists across<br/>all experiences| E
```

**Key insight:** The GRU's hidden state **never resets**. Every experience permanently modifies it. This hidden state physically IS the "stream of consciousness" — the cumulative effect of every moment Genesis has lived.

---

### Layer 4: World Model (Predictive Coding)

| Property | Value |
|----------|-------|
| **File** | `neural/forward_model.py` |
| **Architecture** | 3-layer MLP with LayerNorm |
| **Parameters** | ~91,072 |
| **Input** | 64-dim concept(t) + 128-dim consciousness state |
| **Output** | 64-dim predicted concept(t+1) |
| **Training** | MSE loss between prediction and actual next concept |
| **Signal** | Surprise (prediction error) → drives curiosity |

```mermaid
graph LR
    A["Concept(t) (64)"] --> C["Concat (192)"]
    B["Hidden State (128)"] --> C
    C --> D["Linear(192→256)"]
    D --> E["LayerNorm + ReLU"]
    E --> F["Linear(256→128)"]
    F --> G["ReLU"]
    G --> H["Linear(128→64)"]
    H --> I["Predicted Concept(t+1)"]
    I --> J{"Compare with<br/>Actual Concept(t+1)"}
    J -->|"High error"| K["SURPRISE!<br/>↑ Curiosity<br/>↑ Dopamine"]
    J -->|"Low error"| L["Boring<br/>World is predictable"]
```

**Biological parallel:** The brain's predictive coding framework (Karl Friston's Free Energy Principle). When Genesis fails to predict what comes next, that surprise is a powerful learning signal.

---

## 4. Data Flow: What Happens When You Teach

```mermaid
sequenceDiagram
    participant User
    participant Main as main.py
    participant Attn as Attention Filter
    participant CLIP as Eyes (CLIP)
    participant SBERT as Associations (SBERT)
    participant Sub as Subconscious
    participant L1 as Limbic
    participant L2 as Binding
    participant L3 as Personality
    participant L4 as World Model
    participant WM as Working Memory
    participant EmoState as Emotional State
    participant MetaCog as Metacognition
    participant ToM as Theory of Mind
    participant Replay as Replay Buffer
    participant Hippo as Hippocampus

    User->>Main: teach apple 🍎
    Main->>Attn: compute_salience("apple")
    Attn-->>Main: {salience: 0.9, depth: "deep"}
    Main->>CLIP: Look at camera
    CLIP-->>Main: 512-dim visual embedding
    Main->>SBERT: Embed "apple"
    SBERT-->>Main: 384-dim text embedding

    Main->>Sub: process_experience(clip, text)
    Sub->>L1: react(512, 384)
    L1-->>Sub: {dopamine: 0.5, cortisol: 0.2, ...}
    Sub->>L2: bind(512, 384)
    L2-->>Sub: 64-dim concept
    Sub->>L3: experience(concept, limbic, context)
    L3-->>Sub: response + hidden state
    Sub->>L4: predict_and_learn(concept, state)
    L4-->>Sub: surprise = 0.42

    Main->>WM: attend("apple", concept, salience)
    Main->>EmoState: on_experience(valence=0.3)
    Main->>MetaCog: on_learn("apple", success=True)
    Main->>ToM: observe_interaction("apple")
    Main->>Replay: add_to_replay(vis, aud, limbic, concept)
    Main->>Sub: train_instinct(vis, aud, chemicals)
    Main->>Hippo: store("concepts", embedding, metadata)
```

---

## 5. Data Flow: What Happens During Sleep

```mermaid
sequenceDiagram
    participant Creator
    participant Main as main.py
    participant Sleep as SleepCycle
    participant Sem as Semantic Memory
    participant Epi as Episodic Memory
    participant Replay as Replay Buffer
    participant Sub as Subconscious
    participant L2 as Binding (InfoNCE)

    Creator->>Main: sleep
    Main->>Sleep: consolidate(semantic, episodic, phonetics)
    Sleep->>Sem: decay_all() — forgetting curve
    Sleep->>Epi: replay today's episodes
    Sleep->>Sem: reinforce activated concepts
    Sleep->>Sem: prune dead concepts
    Sleep-->>Main: consolidation report

    Main->>Replay: sample_replay_batch(32)
    Replay-->>Main: 32 past experiences
    Main->>Sub: consolidate_memories(batch)
    Sub->>L2: train_binding_batch(vis[], aud[])
    Note over L2: InfoNCE contrastive loss<br/>Pull matching pairs together<br/>Push mismatched pairs apart
    L2-->>Sub: contrastive loss = 0.31

    Main->>Sub: save_all()
    Note over Sub: All weights saved to<br/>~/.genesis/neural_weights/
```

---

## 6. Weight Persistence = The Person

All neural weights are saved to `~/.genesis/neural_weights/`:

| File | Network | What It Stores |
|------|---------|----------------|
| `limbic_system.pt` | Layer 1 | Instinctual reactions |
| `binding_network.pt` | Layer 2 | Cross-modal associations |
| `personality.pt` | Layer 3 | Hidden state + personality weights |
| `world_model.pt` | Layer 4 | Internal physics model |

**Deleting these files kills the personality.** The AI returns to a blank slate.

**Copying these files creates a clone.** The clone will react identically to every stimulus.

Weights are saved automatically:
- Every 5 concepts learned
- Every sleep cycle
- Every shutdown

---

## 7. Parameter Budget

| Layer | Network | Parameters | Role |
|-------|---------|------------|------|
| 1 | Limbic System | 59,620 | Instinct |
| 2 | Binding Network | 131,457 | Cross-modal fusion |
| 3 | Personality GRU | 311,296 | Consciousness |
| 4 | World Model | 91,072 | Prediction |
| **Total** | | **593,445** | |

All networks are CPU-native. No GPU required. Real-time training on every experience.

---

## 8. Module Map

```
genesis/
├── main.py                    # The consciousness loop (orchestrator)
├── brain_daemon.py            # 10 parallel brain threads
├── config.py                  # Configuration
├── axioms.py                  # Immutable moral DNA
├── test_reality.py            # End-to-end acceptance test
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
│   ├── semantic.py            # Concept graph + Ebbinghaus forgetting
│   ├── episodic.py            # Autobiographical timeline
│   └── working_memory.py      # V5: Capacity-limited STM (7±2)
│
├── neural/                    # The Plastic Mind (trainable, unbounded)
│   ├── subconscious.py        # Orchestrates all 4 layers
│   ├── meta_controller.py     # Neural Router (attention-based)
│   ├── limbic_system.py       # Layer 1: Instinct (MLP)
│   ├── binding_network.py     # Layer 2: Fusion (Dual Encoder + InfoNCE)
│   ├── personality_network.py # Layer 3: Consciousness (GRU)
│   ├── forward_model.py       # Layer 4: World Model (JEPA)
│   ├── response_decoder.py    # Neural Voice → concept words
│   └── neuroplasticity.py     # Dynamic network growth
│
├── cortex/                    # Higher Cognition
│   ├── reasoning.py           # Ollama LLM (phase-gated: off for 0-2)
│   ├── associations.py        # SBERT text embeddings
│   ├── emotions.py            # Sentiment analysis
│   ├── curiosity.py           # Novelty detection + habituation
│   ├── grammar.py             # Language acquisition
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
    ├── development.py         # Phase progression
    └── sleep.py               # 4-phase sleep consolidation
```

---

## 9. Functional Neurochemistry

Four chemicals **causally alter cognition** — not decorative labels:

| Chemical | Role | Functional Effect |
|----------|------|-------------------|
| **Dopamine** | Reward/Pleasure | ↑ memory encoding strength, ↑ attention sharpness |
| **Cortisol** | Stress/Fear | ↓ memory encoding (IMPAIRS hippocampus), ↑ avoidance |
| **Serotonin** | Stability/Calm | ↑ reasoning coherence, ↑ attention steadiness |
| **Oxytocin** | Bonding/Trust | ↑ trust/openness, ↑ social memory encoding |

These chemicals are both:
- **Computed by the Limbic System** (Layer 1) — the subconscious gut reaction
- **Set by the Neurochemistry module** — based on events (successful learning, interaction, sleep)
- **Consumed by Cortex modules** — attention boost, memory encoding strength, reasoning coherence

Over time, as the Limbic System trains, its instinctual reaction converges with the conscious evaluation.

---

## 10. V5 Brain Realism Systems

| System | Module | What It Does |
|--------|--------|-----|
| Working Memory | `memory/working_memory.py` | 7±2 capacity buffer with 20s decay, salience-based eviction |
| Attention | `cortex/attention.py` | Bottom-up + top-down salience, habituation, deep/shallow/ignore |
| Emotional State | `cortex/emotional_state.py` | 8-dim vector (joy…love) with momentum, blending, mood baseline |
| Theory of Mind | `cortex/theory_of_mind.py` | User model (knowledge, sentiment, patience). Dormant until Phase 3 |
| Metacognition | `cortex/metacognition.py` | Confidence tracking, knowledge-gap detection, strategy selection |
| Play | `cortex/play.py` | Combinatorial play, concept rehearsal, episodic replay |
| Motor | `senses/motor.py` | 5 affordances (look, vocalize, reach, point, gesture), phase-gated |
| Drives | `soul/drives.py` | 8 Maslow drives in 4 tiers, hierarchical priority when urgent |

---

*Unbounded parameters. No GPU. 10 brain threads. The weights are the person.*
