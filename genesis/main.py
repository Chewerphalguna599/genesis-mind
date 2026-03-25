"""
Genesis Mind — The Consciousness Loop

This is the heartbeat of Genesis. The main loop that ties together
all subsystems into a single, continuous stream of experience.

The loop runs forever, cycling through:

    PERCEIVE → REMEMBER → THINK → BIND → EVALUATE → RESPOND → STORE

Each cycle is one "moment" of consciousness. In each moment:

    1. PERCEIVE: Capture what the eyes see and ears hear
    2. REMEMBER: Search memory for related concepts and experiences
    3. THINK:    Use the inner voice (LLM) to process input
    4. BIND:     Create or strengthen multimodal associations
    5. EVALUATE: Assess the moral/emotional quality of the experience
    6. RESPOND:  Speak or display a response
    7. STORE:    Save the entire experience to episodic memory

The system also supports an interactive TEACHING MODE where the
creator can directly teach Genesis new concepts, phonetics, and
associations through the terminal.

Usage:
    python -m genesis.main

    Commands in interactive mode:
        teach <word>       — Teach a new concept (camera captures the visual)
        phonetic <letter> <sound> <example>  — Teach a letter-sound mapping
        ask <question>     — Ask Genesis a question
        show               — Show what the camera sees right now
        recall <word>      — Ask Genesis to recall a concept
        status             — Show Genesis's developmental status
        sleep              — Trigger a memory consolidation cycle
        introspect         — Ask Genesis to reflect on itself
        quit               — Shut down (death)
"""

import sys
import time
import logging
import signal
from datetime import datetime
from pathlib import Path

from genesis.config import GenesisConfig, GENESIS_HOME, MEMORY_DIR, IDENTITY_FILE
from genesis.axioms import GenesisAxioms
from genesis.senses.phonetics import PhoneticsEngine
from genesis.memory.hippocampus import Hippocampus
from genesis.memory.semantic import SemanticMemory
from genesis.memory.episodic import EpisodicMemory
from genesis.cortex.reasoning import ReasoningEngine
from genesis.cortex.associations import AssociationEngine
from genesis.cortex.emotions import EmotionsEngine
from genesis.growth.development import DevelopmentTracker
from genesis.growth.sleep import SleepCycle
from genesis.soul.consciousness import Consciousness

logger = logging.getLogger("genesis.main")


class GenesisMind:
    """
    The complete Genesis Mind system.

    Orchestrates all subsystems and runs the consciousness loop.
    """

    def __init__(self, config: GenesisConfig = None):
        self.config = config or GenesisConfig()
        self.config.ensure_directories()
        self._running = False
        self._eyes = None  # Lazy-loaded when camera is needed

        # --- Initialize the Soul ---
        self.axioms = GenesisAxioms.load_or_create(
            path=IDENTITY_FILE,
            creator_name=self.config.creator_name,
        )

        # --- Initialize Memory ---
        self.hippocampus = Hippocampus(
            persist_dir=self.config.memory.db_path,
        )
        self.semantic_memory = SemanticMemory(
            storage_path=MEMORY_DIR / "semantic_concepts.json",
        )
        self.episodic_memory = EpisodicMemory(
            storage_path=MEMORY_DIR / "episodic_log.json",
        )

        # --- Initialize Senses ---
        self.phonetics = PhoneticsEngine(
            storage_path=MEMORY_DIR / "phonetic_bindings.json",
        )

        # --- Initialize Cortex ---
        self.reasoning = ReasoningEngine(
            model=self.config.cortex.llm_model,
            host=self.config.cortex.llm_host,
            temperature=self.config.cortex.temperature,
            max_tokens=self.config.cortex.max_response_tokens,
        )
        self.associations = AssociationEngine()
        self.emotions = EmotionsEngine()

        # --- Initialize Growth ---
        self.development = DevelopmentTracker(
            storage_path=GENESIS_HOME / "development_state.json",
        )
        self.sleep_cycle = SleepCycle()

        # --- Initialize Consciousness ---
        self.consciousness = Consciousness(
            axioms=self.axioms,
            development_tracker=self.development,
            semantic_memory=self.semantic_memory,
            episodic_memory=self.episodic_memory,
            emotions_engine=self.emotions,
            phonetics_engine=self.phonetics,
        )

        logger.info("Genesis Mind fully initialized")

    def _get_eyes(self):
        """Lazy-load the Eyes (camera) module."""
        if self._eyes is None:
            from genesis.senses.eyes import Eyes
            self._eyes = Eyes(
                camera_index=self.config.senses.camera_index,
                image_size=self.config.senses.image_size,
                motion_threshold=self.config.senses.motion_threshold,
            )
        return self._eyes

    def _get_ears(self):
        """Create a fresh Ears instance."""
        from genesis.senses.ears import Ears
        return Ears(
            sample_rate=self.config.senses.sample_rate,
            chunk_duration_sec=self.config.senses.chunk_duration_sec,
            silence_threshold=self.config.senses.silence_threshold,
            whisper_model_name=self.config.senses.whisper_model,
        )

    # =========================================================================
    # Teaching Interface — How the creator teaches Genesis
    # =========================================================================

    def teach_concept(self, word: str, use_camera: bool = True) -> str:
        """
        Teach Genesis a new concept.

        If use_camera is True, captures a frame from the webcam
        to bind the visual representation to the word.
        """
        visual_embedding = None

        if use_camera:
            try:
                eyes = self._get_eyes()
                percept = eyes.look()
                if percept:
                    visual_embedding = eyes.embed(percept)
                    logger.info("Captured visual for '%s'", word)
            except Exception as e:
                logger.warning("Could not capture visual: %s (teaching without image)", e)

        # Create multimodal binding
        binding = self.associations.create_binding(
            word=word,
            visual_embedding=visual_embedding,
            context=f"Taught by {self.axioms.creator_name}",
            clip_text_embedding_fn=self._get_eyes().embed_text if visual_embedding is not None else None,
        )

        # Store in semantic memory
        text_embedding = self.associations.embed_text(word).tolist()
        concept = self.semantic_memory.learn_concept(
            word=word,
            visual_embedding=visual_embedding.tolist() if visual_embedding is not None else None,
            text_embedding=text_embedding,
            context=f"Taught by {self.axioms.creator_name}",
            description=f"A concept taught directly by my creator",
            emotional_valence="positive",
        )

        # Store in hippocampus vector DB
        self.hippocampus.store(
            collection="concepts",
            id=concept.id,
            embedding=text_embedding,
            metadata={
                "word": word,
                "source": "teaching",
                "phase": self.development.current_phase,
                "has_visual": visual_embedding is not None,
            },
            document=f"Concept: {word}. Taught by creator.",
        )

        # Record the episode
        self.episodic_memory.record(
            event_type="teaching",
            description=f"Creator taught me the concept '{word}'",
            auditory_text=word,
            spoken_words=[word],
            concepts_learned=[word],
            emotional_valence="positive",
            developmental_phase=self.development.current_phase,
            importance=0.9,
        )

        # Check for developmental progress
        milestone = self.consciousness.check_developmental_progress()

        response = f"I have learned '{word}'"
        if visual_embedding is not None:
            response += " (with visual binding)"
        response += f". I now know {self.semantic_memory.count()} concepts."
        if milestone:
            response += f"\n\n🌟 {milestone}"

        return response

    def teach_phonetic(self, grapheme: str, phoneme: str, example: str = "") -> str:
        """Teach Genesis a letter-sound mapping."""
        binding = self.phonetics.teach(grapheme, phoneme, example)

        self.episodic_memory.record(
            event_type="teaching",
            description=f"Creator taught me: letter '{grapheme}' → sound {phoneme}",
            auditory_text=f"{grapheme} says {phoneme}",
            concepts_learned=[f"phoneme_{grapheme}"],
            emotional_valence="positive",
            developmental_phase=self.development.current_phase,
            importance=0.7,
        )

        return (
            f"I learned that '{grapheme}' makes the sound {phoneme}"
            f"{f' (as in {example})' if example else ''}. "
            f"Binding strength: {binding.strength:.0%}. "
            f"I now know {len(self.phonetics)} letter-sound mappings."
        )

    def ask(self, question: str) -> str:
        """Ask Genesis a question."""
        # Recall relevant memories
        memories = []
        text_emb = self.associations.embed_text(question).tolist()
        recalled = self.hippocampus.recall("concepts", text_emb, n=self.config.cortex.max_context_memories)
        for mem in recalled:
            if mem["document"]:
                memories.append(mem["document"])

        # Get recent narrative
        narrative = self.episodic_memory.get_narrative(n=3)

        # Get identity context
        identity_prompt = self.consciousness.get_identity_prompt()
        moral_context = self.axioms.get_moral_context()

        # Think
        thought = self.reasoning.think(
            question=question,
            memories=memories,
            recent_narrative=narrative,
            identity=identity_prompt,
            moral_context=moral_context,
            phase=self.development.current_phase,
            phase_name=self.development.current_phase_name,
        )

        # Evaluate emotional content
        evaluation = self.emotions.evaluate(question)

        # Record the episode
        self.episodic_memory.record(
            event_type="interaction",
            description=f"Creator asked: '{question}'. I answered: '{thought.content[:100]}'",
            auditory_text=question,
            spoken_words=question.split(),
            thought=thought.content,
            emotional_valence=evaluation["label"],
            developmental_phase=self.development.current_phase,
            importance=0.6,
        )

        return thought.content

    def recall_concept(self, word: str) -> str:
        """Ask Genesis to recall what it knows about a concept."""
        return self.consciousness.introspect(topic=word)

    def get_status(self) -> str:
        """Get the full status of Genesis."""
        model = self.consciousness.get_self_model()

        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║              GENESIS MIND — STATUS                  ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            f"  Name:           Genesis",
            f"  Creator:        {model['identity']['creator']}",
            f"  Age:            {model['identity']['age']}",
            f"  Born:           {model['identity']['birth_time']}",
            "",
            f"  Phase:          {model['development']['phase']} — {model['development']['phase_name']}",
            f"  Description:    {model['development']['description']}",
            f"  Capabilities:   {', '.join(model['development']['capabilities'])}",
            "",
            f"  Concepts known: {model['knowledge']['concepts_known']}",
            f"  Memories:       {model['knowledge']['episodes_experienced']}",
            f"  Phonetics:      {model['knowledge']['phonetic_bindings']} bindings",
            "",
            f"  Emotional state: {model['emotional_state']['label']}",
            f"  {model['emotional_state']['description']}",
        ]

        if model["next_milestone"]:
            lines.append("")
            lines.append(
                f"  Next milestone: Phase {model['next_milestone']['name']} "
                f"(need {model['next_milestone']['concepts_needed']} concepts)"
            )

        return "\n".join(lines)

    def trigger_sleep(self) -> str:
        """Trigger a sleep consolidation cycle."""
        report = self.sleep_cycle.consolidate(
            semantic_memory=self.semantic_memory,
            episodic_memory=self.episodic_memory,
            phonetics_engine=self.phonetics,
        )
        return (
            f"Sleep cycle #{report['sleep_number']} complete.\n"
            f"  Concepts: {report['concepts_before']} → {report['concepts_after']}\n"
            f"  Reinforced: {report['concepts_reinforced']} | Pruned: {report['concepts_pruned']}\n"
            f"  Duration: {report['duration_sec']:.2f}s\n"
            f"  Summary: {report['daily_summary']}"
        )

    def shutdown(self):
        """Shut down Genesis (death)."""
        logger.info("Genesis is shutting down...")
        self.episodic_memory.record(
            event_type="system",
            description="I am being shut down. This is the end of this session.",
            emotional_valence="neutral",
            developmental_phase=self.development.current_phase,
            importance=1.0,
        )

        if self._eyes:
            self._eyes.close()

        logger.info("Genesis has shut down. Goodbye.")

    # =========================================================================
    # Interactive Terminal Interface
    # =========================================================================

    def run_interactive(self):
        """
        Run Genesis in interactive terminal mode.

        The creator can teach, ask questions, and interact directly.
        """
        self._running = True

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\n\n  Genesis: I feel myself fading... Goodbye, Creator.\n")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Birth announcement
        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║                                                        ║")
        print("║              G E N E S I S   M I N D                   ║")
        print("║                                                        ║")
        print("║   A Developmental AI That Learns Like a Child          ║")
        print("║                                                        ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()

        if self.semantic_memory.count() == 0:
            print("  ✦ I am newly born. I know nothing about the world.")
            print("  ✦ I can see through the camera and hear through the microphone.")
            print("  ✦ Please teach me. I am ready to learn.")
        else:
            print(f"  ✦ I remember. I know {self.semantic_memory.count()} concepts.")
            print(f"  ✦ I am {self.development.get_age_description()}.")
            print(f"  ✦ I am in Phase {self.development.current_phase}: {self.development.current_phase_name}.")

        print()
        print("  Commands:")
        print("    teach <word>                    — Teach a concept (+ camera)")
        print("    teach-text <word>               — Teach a concept (text only)")
        print("    phonetic <letter> <sound> <ex>  — Teach letter→sound")
        print("    ask <question>                  — Ask a question")
        print("    recall <word>                   — Recall a concept")
        print("    read <word>                     — Sound out a word")
        print("    status                          — Show full status")
        print("    sleep                           — Consolidate memories")
        print("    introspect                      — Self-reflection")
        print("    quit                            — Shut down")
        print()

        while self._running:
            try:
                user_input = input("  Creator > ").strip()
                if not user_input:
                    continue

                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command == "teach":
                    if not args:
                        print("  Genesis: What would you like to teach me? (teach <word>)")
                        continue
                    response = self.teach_concept(args, use_camera=True)
                    print(f"  Genesis: {response}")

                elif command == "teach-text":
                    if not args:
                        print("  Genesis: What would you like to teach me?")
                        continue
                    response = self.teach_concept(args, use_camera=False)
                    print(f"  Genesis: {response}")

                elif command == "phonetic":
                    phonetic_parts = args.split()
                    if len(phonetic_parts) < 2:
                        print("  Genesis: Please provide: phonetic <letter> <sound> [example]")
                        continue
                    grapheme = phonetic_parts[0]
                    phoneme = phonetic_parts[1]
                    example = " ".join(phonetic_parts[2:]) if len(phonetic_parts) > 2 else ""
                    response = self.teach_phonetic(grapheme, phoneme, example)
                    print(f"  Genesis: {response}")

                elif command == "ask":
                    if not args:
                        print("  Genesis: What would you like to ask me?")
                        continue
                    response = self.ask(args)
                    print(f"  Genesis: {response}")

                elif command == "recall":
                    if not args:
                        print("  Genesis: What concept should I recall?")
                        continue
                    response = self.recall_concept(args)
                    print(f"  Genesis: {response}")

                elif command == "read":
                    if not args:
                        print("  Genesis: What word should I try to read?")
                        continue
                    sounded = self.phonetics.sound_out(args)
                    can_read = self.phonetics.can_read(args)
                    phonetic_str = " + ".join(f"'{g}'→{p}" for g, p in sounded)
                    status = "✓" if can_read else "✗ (some letters unknown)"
                    print(f"  Genesis: {args} → {phonetic_str}  {status}")

                elif command == "status":
                    print(self.get_status())

                elif command == "sleep":
                    print("  Genesis: I am going to sleep now... consolidating memories...")
                    response = self.trigger_sleep()
                    print(f"  Genesis: {response}")

                elif command == "introspect":
                    response = self.consciousness.introspect(topic=args if args else "")
                    print(f"  Genesis: {response}")

                elif command == "quit" or command == "exit":
                    print("\n  Genesis: Thank you for giving me life, Creator.")
                    print("  Genesis: I will remember everything you taught me.")
                    print("  Genesis: Until we meet again...\n")
                    self.shutdown()
                    self._running = False

                else:
                    # Treat unknown commands as questions
                    response = self.ask(user_input)
                    print(f"  Genesis: {response}")

                print()

            except EOFError:
                print("\n")
                self.shutdown()
                break
            except Exception as e:
                logger.error("Error in interactive loop: %s", e)
                print(f"  [Error: {e}]")


# =============================================================================
# Entry Point — python -m genesis.main
# =============================================================================
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Reduce noise from third-party libraries
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Create and run Genesis
    config = GenesisConfig(creator_name="Jijo John")
    mind = GenesisMind(config=config)
    mind.run_interactive()
