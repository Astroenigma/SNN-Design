# encoding/binned_stream.py
from __future__ import annotations
import numpy as np
from typing import Iterator, Tuple

class BinnedEventStream:
    """Streams sparse input indices per time bin (bin_ms) to avoid huge [T,N] rasters."""
    def __init__(self, events: np.ndarray, in_shape: Tuple[int,int],
                 use_polarity: bool = True, bin_ms: int = 10):
        assert events.ndim == 2 and events.shape[1] == 4
        self.events = events
        self.H, self.W = in_shape
        self.use_polarity = bool(use_polarity)
        self.bin_ms = int(bin_ms)
        self.N = self.H * self.W * (2 if self.use_polarity else 1)
        t = events[:, 0]
        self.t_bin = (t // self.bin_ms).astype(np.int64)
        self.T_bins = int(self.t_bin.max(initial=0)) + 1

    def _flat_index(self, x: int, y: int, p: int) -> int:
        base = y * self.W + x
        if self.use_polarity:
            ch = 0 if p > 0 else 1
            return base * 2 + ch
        return base

    def iter_sparse(self) -> Iterator[Tuple[int, np.ndarray]]:
        order = np.argsort(self.t_bin, kind="stable")
        tb = self.t_bin[order]
        x = self.events[order, 1]
        y = self.events[order, 2]
        p = self.events[order, 3]

        start, n = 0, len(order)
        while start < n:
            b = tb[start]
            end = start
            while end < n and tb[end] == b:
                end += 1
            if end > start:
                idxs = [self._flat_index(int(x[i]), int(y[i]), int(p[i])) for i in range(start, end)]
                yield int(b), np.unique(np.array(idxs, dtype=np.int64))
            else:
                yield int(b), np.empty((0,), dtype=np.int64)
            start = end

