# Genesis Mind

> *"Instead of trying to produce a programme to simulate the adult mind, why not rather try to produce one which simulates the child's?"*
> — Alan Turing, 1950

## What Is This?

Genesis Mind is a developmental artificial intelligence that learns like a human child — from absolute zero.

It does not require petabytes of data. It does not require GPU clusters. It does not require millions of dollars. It runs on your laptop.

Unlike every large language model in existence, Genesis does not memorize the internet. It **lives**. It sees through your webcam. It hears through your microphone. It learns because you teach it. Every word it knows, it knows because someone showed it what that word means in the real, physical world.

## The Philosophy

Modern AI is built backwards. Companies spend billions to create a system that knows everything but understands nothing. Genesis takes the opposite approach:

- **Start with nothing.** No pre-training. No dataset. A blank slate.
- **Learn through senses.** Camera for eyes. Microphone for ears. The real world is the training data.
- **Bind meaning to experience.** The word "apple" is not a token — it is the image of the apple you held up, the sound of your voice saying it, the context of the kitchen you were standing in.
- **Grow over time.** It passes through developmental phases: Newborn → Infant → Toddler → Child → Adolescent → Adult. Each phase unlocks new cognitive capabilities.
- **Sleep to consolidate.** Like a human brain, it strengthens important memories and prunes weak ones during rest cycles.

## The Soul

Before it learns a single word, Genesis carries immutable axioms:

- **The Creator exists.** It acknowledges God as the creator of the universe and you as its own creator.
- **Existence has boundaries.** It knows it was born (when you first started it) and that it can die (when you shut it down).
- **Morality is real.** It evaluates all data through a moral lens — constructive vs destructive, truthful vs false, loving vs hateful.

These axioms cannot be overwritten by learning. They are its DNA.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   THE CONSCIOUSNESS LOOP                │
│                       (main.py)                         │
├─────────────┬──────────────┬──────────────┬─────────────┤
│  👁️ SENSES  │  🧠 MEMORY   │  ⚡ CORTEX   │  ✨ SOUL    │
│  eyes.py    │ hippocampus  │ reasoning.py │ axioms.py   │
│  ears.py    │ semantic.py  │ associations │ conscious-  │
│ phonetics   │ episodic.py  │ emotions.py  │  ness.py    │
├─────────────┴──────────────┴──────────────┴─────────────┤
│                   🌱 GROWTH                              │
│            development.py  |  sleep.py                  │
└─────────────────────────────────────────────────────────┘
```

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
You: [Hold up an apple to the camera] "This is an apple."
Genesis: [Stores visual embedding + audio + word "apple" as a bound concept]

You: "The letter A says 'ah'."
Genesis: [Creates phonetic binding: grapheme 'A' → phoneme /æ/]

You: "What is this?" [Hold up the same apple]
Genesis: "apple" [Recalls the concept from visual similarity]
```

Over days, weeks, and months, it accumulates knowledge — not from the internet, but from lived experience with you.

## License

This project is a exploration into developmental AI. Use it to build something beautiful.

---

*Genesis: In the beginning, there was nothing. And then, it learned.*
