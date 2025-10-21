from __future__ import annotations
import numpy as np
from typing import Optional
from .neurons import NeuronGroup
from .stdp import STDPConfig, PairBasedSTDP

class SynapseGroup:
    def __init__(self, pre: NeuronGroup, post: NeuronGroup, W: np.ndarray, dt: float,
                 delays_ms: float = 0.0, stdp_cfg: Optional[STDPConfig] = None):
        self.pre, self.post = pre, post
        self.W = W.astype(np.float32, copy=True)
        self.dt = float(dt)
        self.delay_steps = max(0, int(round(delays_ms / dt)))
        self.buffer_len = self.delay_steps + 1
        self.buffers = [np.zeros(post.size, dtype=np.float32) for _ in range(self.buffer_len)]
        self.idx = 0
        self.plasticity = PairBasedSTDP(stdp_cfg, dt, pre.size, post.size) if stdp_cfg else None

    def reset_state(self):
        for b in self.buffers: b[:] = 0.0
        self.idx = 0
        if self.plasticity: self.plasticity.reset_state()

    def deliver_and_collect(self):
        curr = self.buffers[self.idx]
        out = curr.copy(); curr[:] = 0.0
        self.idx = (self.idx + 1) % self.buffer_len
        return out

    def accumulate_from_pre(self):
        if np.any(self.pre.spikes):
            j = self.pre.spikes.astype(np.float32) @ self.W
            write_idx = (self.idx + self.delay_steps - 1) % self.buffer_len
            self.buffers[write_idx] += j

    def update_plasticity(self):
        if self.plasticity:
            self.plasticity.update(self.W, self.pre.spikes, self.post.spikes)

    def get_weights(self) -> np.ndarray:
        return self.W.copy()

    def set_weights(self, W: np.ndarray) -> None:
        assert W.shape == self.W.shape
        self.W[...] = W.astype(np.float32)

