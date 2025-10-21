from __future__ import annotations
import numpy as np
from typing import Dict, List
from core.neurons import NeuronGroup
from core.synapse import SynapseGroup

class Simulator:
    def __init__(self, dt: float, groups: List[NeuronGroup], syns: List[SynapseGroup]):
        self.dt, self.groups, self.syns = dt, groups, syns

    def reset_state(self):
        for g in self.groups: 
            if hasattr(g, "reset_state"): g.reset_state()
        for s in self.syns: s.reset_state()

    def run(self, T: int, external_inputs: Dict[int, np.ndarray], record=True):
        rasters = [np.zeros((T, g.size), dtype=np.bool_) for g in self.groups] if record else None
        for t in range(T):
            post_curr = [np.zeros(g.size, dtype=np.float32) for g in self.groups]
            for s in self.syns: post_curr[self._gid(s.post)] += s.deliver_and_collect()
            for gi, train in external_inputs.items():
                post_curr[gi] += train[t].astype(np.float32)
            for i,g in enumerate(self.groups):
                g.step(post_curr[i], self.dt)
                if record: rasters[i][t]=g.spikes
            for s in self.syns: s.accumulate_from_pre(); s.update_plasticity()
        return rasters

    def _gid(self, group: NeuronGroup)->int:
        for i,g in enumerate(self.groups):
            if g is group: return i
        raise ValueError
    
    # engine/simulator.py  (append inside class Simulator)
    def run_stream(
        self,
        T: int,
        input_group_ix: int,
        sparse_iter,
        record: bool = True,
        progress_every: int = 1000,
    ):
        """
        Streamed simulation: consumes an iterator that yields (t_bin, idx_array)
        for spikes into 'input_group_ix' per step. No [T,N] allocation.
        """
        rasters = [np.zeros((T, g.size), dtype=np.bool_) for g in self.groups] if record else None

        # Prepare a pointer over sparse inputs
        sparse_iter = iter(sparse_iter)
        next_bin, next_idx = None, None
        try:
            next_bin, next_idx = next(sparse_iter)
        except StopIteration:
            next_bin, next_idx = None, None

        for t in range(T):
            # progress print (non-blocking hint)
            if progress_every and (t % progress_every == 0):
                print(f"[sim] step {t}/{T}", flush=True)

            # 1) deliver delayed syn currents
            post_curr = [np.zeros(g.size, dtype=np.float32) for g in self.groups]
            for s in self.syns:
                post_curr[self._gid(s.post)] += s.deliver_and_collect()

            # 2) external input for this step: sparse indices -> unit current
            if next_bin is not None and next_bin == t and next_idx is not None and len(next_idx) > 0:
                vec = post_curr[input_group_ix]
                vec[next_idx] += 1.0
                # advance sparse stream if multiple chunks share same t
                while True:
                    try:
                        peek_bin, peek_idx = next(sparse_iter)
                    except StopIteration:
                        next_bin, next_idx = None, None
                        break
                    if peek_bin != t:
                        next_bin, next_idx = peek_bin, peek_idx
                        break
                    if len(peek_idx) > 0:
                        vec[peek_idx] += 1.0

            # 3) step neurons
            for i, g in enumerate(self.groups):
                g.step(post_curr[i], self.dt)
                if record:
                    rasters[i][t] = g.spikes

            # 4) schedule synaptic current from this step's pre spikes
            for s in self.syns:
                s.accumulate_from_pre()

            # 5) plasticity update
            for s in self.syns:
                s.update_plasticity()

        return rasters

