# Shitty_SNN/engine/runners.py
from __future__ import annotations
import numpy as np

def run_stream_counts_with_competition_and_homeostasis(
    sim, T: int, input_group_ix: int, sparse_iter,
    bin_ms: int,
    bias_hidden: float = 0.02,
    bias_output_init: float = 0.0,
    soft_wta_gain: float = 0.9,
    target_rate_hz: float = 5.0,
    homeo_tau_s: float = 0.5,
    homeo_k: float = 0.05,
    progress_every: int = 1000
):
    """
    Counts-only streamed simulation with:
      - tiny hidden bias (to avoid silence),
      - homeostatic output bias toward target rate,
      - soft WTA on output.
    Returns: [counts_in, counts_hidden, counts_out]
    """
    counts = [np.zeros(g.size, dtype=np.int32) for g in sim.groups]

    it = iter(sparse_iter)
    try:
        next_bin, next_idx = next(it)
    except StopIteration:
        next_bin, next_idx = None, None

    def gid(group):
        for i, g in enumerate(sim.groups):
            if g is group: return i
        raise ValueError("Group not in sim.groups")

    out_ix = len(sim.groups) - 1
    dt_s   = bin_ms / 1000.0

    # Homeostasis state
    n_out = sim.groups[out_ix].size
    out_bias = np.full(n_out, bias_output_init, dtype=np.float32)
    ema_rate = np.zeros(n_out, dtype=np.float32)
    beta = dt_s / (homeo_tau_s + dt_s)  # EMA coefficient

    for t in range(T):
        if progress_every and (t % progress_every == 0):
            print(f"[sim] step {t}/{T}", flush=True)

        # deliver delayed syn currents
        post_curr = [np.zeros(g.size, dtype=np.float32) for g in sim.groups]
        for s in sim.syns:
            post_curr[gid(s.post)] += s.deliver_and_collect()

        # inject sparse inputs
        if next_bin is not None and next_bin == t and next_idx is not None and len(next_idx) > 0:
            post_curr[input_group_ix][next_idx] += 1.0
            while True:
                try:
                    peek_bin, peek_idx = next(it)
                except StopIteration:
                    next_bin, next_idx = None, None
                    break
                if peek_bin != t:
                    next_bin, next_idx = peek_bin, peek_idx
                    break
                if len(peek_idx) > 0:
                    post_curr[input_group_ix][peek_idx] += 1.0

        # hidden bias
        if bias_hidden > 0:
            post_curr[1] += bias_hidden

        # output bias (homeostatic)
        if n_out > 0:
            post_curr[out_ix] += out_bias

        # soft WTA on outputs (sharpen differences)
        if soft_wta_gain > 0 and n_out > 0:
            oc = post_curr[out_ix]
            mu = oc.mean() if oc.size else 0.0
            post_curr[out_ix] = oc - soft_wta_gain * (oc - mu)

        # step neurons + accumulate counts
        for i, g in enumerate(sim.groups):
            g.step(post_curr[i], sim.dt)
            counts[i] += g.spikes.astype(np.int32)

        # schedule synaptic current + STDP
        for s in sim.syns:
            s.accumulate_from_pre()
        for s in sim.syns:
            s.update_plasticity()

        # homeostasis: EMA rate & bias update
        if n_out > 0:
            inst_rate = sim.groups[out_ix].spikes.astype(np.float32) / dt_s
            ema_rate = (1.0 - beta) * ema_rate + beta * inst_rate
            err = ema_rate - target_rate_hz
            out_bias -= homeo_k * err * dt_s
            np.clip(out_bias, -0.5, 0.5, out=out_bias)

    return counts
