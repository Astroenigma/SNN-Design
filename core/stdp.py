from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class STDPConfig:
    A_plus: float = 0.01
    A_minus: float = 0.012
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    w_min: float = 0.0
    w_max: float = 1.0

class PairBasedSTDP:
    def __init__(self, cfg: STDPConfig, dt: float, pre_size: int, post_size: int):
        self.cfg = cfg
        self.dt = float(dt)
        self.pre_trace = np.zeros(pre_size, dtype=np.float32)
        self.post_trace = np.zeros(post_size, dtype=np.float32)
        self.decay_pre = np.exp(-self.dt / self.cfg.tau_plus)
        self.decay_post = np.exp(-self.dt / self.cfg.tau_minus)

    def reset_state(self):
        self.pre_trace[:] = 0.0
        self.post_trace[:] = 0.0

    def update(self, W: np.ndarray, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> None:
        assert W.ndim == 2
        assert pre_spikes.shape[0] == W.shape[0]
        assert post_spikes.shape[0] == W.shape[1]

        # decay traces
        self.pre_trace *= self.decay_pre
        self.post_trace *= self.decay_post

        # increment on spikes
        if np.any(pre_spikes):
            self.pre_trace[pre_spikes] += 1.0
        if np.any(post_spikes):
            self.post_trace[post_spikes] += 1.0

        # LTD on pre spikes (use integer indices, not boolean masks in np.ix_)
        if np.any(pre_spikes):
            pre_idx = np.where(pre_spikes)[0]
            # subtract post_trace from all outgoing weights of spiking pres
            W[pre_idx, :] -= self.cfg.A_minus * self.post_trace[None, :]

        # LTP on post spikes
        if np.any(post_spikes):
            post_idx = np.where(post_spikes)[0]
            # add pre_trace to all incoming weights of spiking posts
            W[:, post_idx] += self.cfg.A_plus * self.pre_trace[:, None]

        # clip
        np.clip(W, self.cfg.w_min, self.cfg.w_max, out=W)

