from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Protocol

@dataclass
class LIFConfig:
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_thresh: float = 1.0
    tau_mem: float = 20.0
    refractory_period: float = 2.0

class NeuronGroup(Protocol):
    def step(self, input_current: np.ndarray, dt: float) -> np.ndarray: ...
    def reset_state(self) -> None: ...
    @property
    def size(self) -> int: ...

class LIFNeuronGroup:
    def __init__(self, n: int, cfg: LIFConfig):
        self._n = int(n)
        self.cfg = cfg
        self.v = np.full(self._n, cfg.v_rest, dtype=np.float32)
        self.ref_remaining = np.zeros(self._n, dtype=np.float32)
        self.spikes = np.zeros(self._n, dtype=np.bool_)

    @property
    def size(self) -> int:
        return self._n

    def reset_state(self) -> None:
        self.v[:] = self.cfg.v_rest
        self.ref_remaining[:] = 0.0
        self.spikes[:] = False

    def step(self, input_current: np.ndarray, dt: float) -> np.ndarray:
        self.spikes[:] = False
        active = self.ref_remaining <= 0.0
        dv = (-(self.v - self.cfg.v_rest) + input_current) * (dt / self.cfg.tau_mem)
        self.v[active] += dv[active]
        fired = self.v >= self.cfg.v_thresh
        if np.any(fired):
            self.spikes[fired] = True
            self.v[fired] = self.cfg.v_reset
            self.ref_remaining[fired] = self.cfg.refractory_period
        self.ref_remaining[~active] -= dt
        return self.spikes
