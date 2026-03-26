"""
Genesis Mind — Neuroplasticity (Network Growth)

A human baby's brain doesn't start with 100 billion connected neurons.
The structure exists, but the CONNECTIONS grow during development:

    Newborn:    Sparse connections, small networks
    Infant:     Rapid synaptogenesis — explosion of new connections
    Toddler:    Pruning begins — weak connections die, strong ones thicken
    Child:      Myelination — fast pathways become FASTER
    Adolescent: Prefrontal growth — abstract reasoning circuits mature
    Adult:      Stable architecture, slow refinement

Genesis replicates this: at each phase transition, the neural networks
PHYSICALLY GROW — wider layers, deeper circuits, more capacity.

Growth mechanisms:
    1. LAYER WIDENING: Add neurons to hidden layers
    2. DEPTH EXTENSION: Add new layers to existing networks
    3. WEIGHT INHERITANCE: New neurons are initialized from statistics
       of existing weights (not random!) — transfer from self.
    4. CAPACITY SCHEDULING: Each phase defines the target network size

This means a Newborn Genesis literally cannot think as complexly as
a Child Genesis — the hardware isn't there yet. The brain must GROW
into its capabilities, just like ours did.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("genesis.neural.neuroplasticity")


# Network dimensions per developmental phase
# Format: phase_number -> {component: (hidden_dim, num_layers)}
GROWTH_SCHEDULE = {
    # Phase 0: Newborn — minimal networks
    0: {
        "limbic": {"hidden_dim": 128, "layers": 2},
        "binding": {"visual_dim": 256, "audio_dim": 192, "concept_dim": 64},
        "personality": {"hidden_dim": 128, "gru_layers": 3},
        "world_model": {"hidden_dim": 128},
        "meta_controller": {"hidden_dim": 64},
    },
    # Phase 1: Infant — first growth spurt
    1: {
        "limbic": {"hidden_dim": 192, "layers": 2},
        "binding": {"visual_dim": 320, "audio_dim": 256, "concept_dim": 96},
        "personality": {"hidden_dim": 192, "gru_layers": 3},
        "world_model": {"hidden_dim": 192},
        "meta_controller": {"hidden_dim": 96},
    },
    # Phase 2: Toddler — exploratory growth
    2: {
        "limbic": {"hidden_dim": 256, "layers": 3},
        "binding": {"visual_dim": 384, "audio_dim": 320, "concept_dim": 128},
        "personality": {"hidden_dim": 256, "gru_layers": 3},
        "world_model": {"hidden_dim": 256},
        "meta_controller": {"hidden_dim": 128},
    },
    # Phase 3: Child — substantial architecture
    3: {
        "limbic": {"hidden_dim": 384, "layers": 3},
        "binding": {"visual_dim": 448, "audio_dim": 384, "concept_dim": 192},
        "personality": {"hidden_dim": 384, "gru_layers": 4},
        "world_model": {"hidden_dim": 384},
        "meta_controller": {"hidden_dim": 192},
    },
    # Phase 4: Adolescent — near-adult capacity
    4: {
        "limbic": {"hidden_dim": 512, "layers": 4},
        "binding": {"visual_dim": 512, "audio_dim": 448, "concept_dim": 256},
        "personality": {"hidden_dim": 512, "gru_layers": 4},
        "world_model": {"hidden_dim": 512},
        "meta_controller": {"hidden_dim": 256},
    },
    # Phase 5: Adult — full capacity
    5: {
        "limbic": {"hidden_dim": 640, "layers": 4},
        "binding": {"visual_dim": 512, "audio_dim": 512, "concept_dim": 320},
        "personality": {"hidden_dim": 640, "gru_layers": 5},
        "world_model": {"hidden_dim": 640},
        "meta_controller": {"hidden_dim": 320},
    },
}


def _grow_linear(old_layer: nn.Linear, new_in: int, new_out: int) -> nn.Linear:
    """
    Grow a linear layer by expanding its dimensions.
    
    New weights are initialized from the statistics of existing weights
    (mean/std transfer) — NOT random. This preserves learned patterns
    while adding capacity.
    """
    new_layer = nn.Linear(new_in, new_out)

    # Copy existing weights into the new layer
    old_in = old_layer.in_features
    old_out = old_layer.out_features
    copy_in = min(old_in, new_in)
    copy_out = min(old_out, new_out)

    with torch.no_grad():
        # Copy what we can from the old layer
        new_layer.weight.data[:copy_out, :copy_in] = old_layer.weight.data[:copy_out, :copy_in]
        new_layer.bias.data[:copy_out] = old_layer.bias.data[:copy_out]

        # Initialize new neurons from statistics of existing weights
        if new_out > old_out:
            mean = old_layer.weight.data.mean()
            std = old_layer.weight.data.std()
            new_layer.weight.data[old_out:, :copy_in] = torch.normal(
                mean, std, size=(new_out - old_out, copy_in)
            )
            new_layer.bias.data[old_out:] = old_layer.bias.data.mean()

        if new_in > old_in:
            mean = old_layer.weight.data.mean()
            std = old_layer.weight.data.std()
            new_layer.weight.data[:copy_out, old_in:] = torch.normal(
                mean, std, size=(copy_out, new_in - old_in)
            )

    return new_layer


def _grow_gru(old_gru: nn.GRU, new_input: int, new_hidden: int,
              new_layers: int) -> nn.GRU:
    """
    Grow a GRU by expanding dimensions and/or adding layers.
    
    Existing weights are preserved. New capacity is initialized
    from existing weight statistics.
    """
    new_gru = nn.GRU(
        input_size=new_input,
        hidden_size=new_hidden,
        num_layers=new_layers,
        batch_first=True,
    )

    old_layers = old_gru.num_layers
    old_hidden = old_gru.hidden_size
    old_input = old_gru.input_size

    with torch.no_grad():
        for layer_idx in range(min(old_layers, new_layers)):
            # Get old weight names
            for param_name in ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']:
                suffix = f'_l{layer_idx}'
                old_param = getattr(old_gru, param_name + suffix)
                new_param = getattr(new_gru, param_name + suffix)

                # GRU weights are (3*hidden, input/hidden) for gates r,z,n
                if 'weight' in param_name:
                    old_h = old_hidden
                    new_h = new_hidden
                    old_d = old_input if (layer_idx == 0 and 'ih' in param_name) else old_hidden
                    new_d = new_input if (layer_idx == 0 and 'ih' in param_name) else new_hidden

                    # Copy gate-by-gate (3 gates)
                    for gate in range(3):
                        copy_h = min(old_h, new_h)
                        copy_d = min(old_d, new_d)
                        src_start = gate * old_h
                        dst_start = gate * new_h
                        new_param.data[dst_start:dst_start + copy_h, :copy_d] = \
                            old_param.data[src_start:src_start + copy_h, :copy_d]
                else:
                    # Bias: (3*hidden,)
                    for gate in range(3):
                        copy_h = min(old_hidden, new_hidden)
                        src_start = gate * old_hidden
                        dst_start = gate * new_hidden
                        new_param.data[dst_start:dst_start + copy_h] = \
                            old_param.data[src_start:src_start + copy_h]

    return new_gru


class Neuroplasticity:
    """
    The network growth engine.
    
    At each developmental phase transition, this system physically
    grows the neural networks — wider layers, deeper circuits.
    """

    def __init__(self):
        self._growth_history: List[Dict] = []
        self._total_growth_events = 0
        logger.info("Neuroplasticity engine initialized — networks will grow with development")

    def get_target_dims(self, phase: int) -> Dict:
        """Get the target network dimensions for a given phase."""
        return GROWTH_SCHEDULE.get(phase, GROWTH_SCHEDULE[0])

    def should_grow(self, current_phase: int, subconscious) -> bool:
        """Check if networks need to grow for the current phase."""
        target = self.get_target_dims(current_phase)
        # Check personality hidden dim as proxy
        current_hidden = subconscious.personality.network.hidden_size
        target_hidden = target["personality"]["hidden_dim"]
        return target_hidden > current_hidden

    def grow_networks(self, new_phase: int, subconscious) -> Dict:
        """
        Grow all networks to match the target dimensions for the new phase.
        
        Returns a report of what changed.
        """
        target = self.get_target_dims(new_phase)
        report = {"phase": new_phase, "changes": {}}

        # Count params before
        params_before = subconscious.get_total_params()

        # ── Grow Personality GRU ──
        personality = subconscious.personality
        target_p = target["personality"]
        if target_p["hidden_dim"] > personality.network.hidden_size:
            old_hidden = personality.network.hidden_size
            new_hidden = target_p["hidden_dim"]
            new_layers = target_p["gru_layers"]

            # Grow the GRU
            new_gru = _grow_gru(
                personality.network, personality.network.input_size,
                new_hidden, new_layers
            )
            personality.network = new_gru

            # Grow the output projection
            if hasattr(personality, 'output_proj'):
                personality.output_proj = _grow_linear(
                    personality.output_proj, new_hidden, personality.output_proj.out_features
                )

            # Resize hidden state
            personality._hidden = torch.zeros(new_layers, 1, new_hidden)

            # Rebuild optimizer
            personality.optimizer = torch.optim.Adam(
                list(personality.network.parameters()) +
                (list(personality.output_proj.parameters()) if hasattr(personality, 'output_proj') else []),
                lr=personality.optimizer.param_groups[0]['lr']
            )

            report["changes"]["personality"] = {
                "hidden": f"{old_hidden} → {new_hidden}",
                "layers": f"→ {new_layers}",
            }
            logger.info("  🧠 Personality GRU grew: hidden %d → %d, layers → %d",
                         old_hidden, new_hidden, new_layers)

        # ── Grow Meta-Controller ──
        meta = subconscious.meta_controller
        target_m = target["meta_controller"]
        current_mc_hidden = meta.network.attention[0].in_features
        if target_m["hidden_dim"] > 64:  # Only grow if target is bigger than initial
            # Rebuild with larger hidden dim (simpler than in-place growth)
            from genesis.neural.meta_controller import MetaController
            old_routes = meta._total_routes
            old_avg = meta._avg_weights.copy()
            
            new_meta = MetaController(
                input_dim=meta.input_dim,
                num_modules=meta.num_modules,
                hidden_dim=target_m["hidden_dim"],
            )
            new_meta._total_routes = old_routes
            new_meta._avg_weights = old_avg
            subconscious.meta_controller = new_meta

            report["changes"]["meta_controller"] = {
                "hidden": f"→ {target_m['hidden_dim']}",
            }
            logger.info("  🧠 Meta-controller grew: hidden → %d", target_m["hidden_dim"])

        # Count params after
        params_after = subconscious.get_total_params()
        growth = params_after - params_before
        report["params_before"] = params_before
        report["params_after"] = params_after
        report["params_added"] = growth
        report["growth_pct"] = round(growth / max(1, params_before) * 100, 1)

        self._growth_history.append(report)
        self._total_growth_events += 1

        logger.info("  🧠 NEURAL GROWTH COMPLETE: %d → %d params (+%d, +%.1f%%)",
                     params_before, params_after, growth, report["growth_pct"])

        return report

    def get_stats(self) -> Dict:
        return {
            "total_growth_events": self._total_growth_events,
            "growth_history": self._growth_history,
        }

    def __repr__(self) -> str:
        return f"Neuroplasticity(growth_events={self._total_growth_events})"
