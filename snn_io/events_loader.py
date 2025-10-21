from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Tuple, Dict, Any, Optional

def _infer_time_steps(t_raw: np.ndarray, time_unit: Optional[str]) -> np.ndarray:
    if time_unit is None:
        time_unit = "us" if t_raw.max(initial=0) > 10_000 else "ms"
    u = time_unit.lower()
    if u in ("us", "µs", "micro", "microseconds"):  return (t_raw // 1000).astype(np.int64)
    if u in ("ms", "milliseconds"):                  return t_raw.astype(np.int64)
    if u in ("s", "sec", "seconds"):                 return (t_raw * 1000.0).astype(np.int64)
    return (t_raw // 1000).astype(np.int64) if t_raw.max(initial=0) > 10_000 else t_raw.astype(np.int64)

def _polarity_to_pm1(p_raw: np.ndarray) -> np.ndarray:
    if p_raw.dtype == np.bool_: return np.where(p_raw, 1, -1).astype(np.int64)
    p = p_raw.astype(np.int64)
    if np.any(p == 0): return np.where(p > 0, 1, -1).astype(np.int64)
    if np.all((p == 1) | (p == -1)): return p
    return np.where(p > 0, 1, -1).astype(np.int64)

def load_events_generic(npz_path: Path, format_cfg: Optional[Dict[str, Any]] = None
                        ) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """Normalize various NPZ schemas to (K,4) = [t(ms), x, y, p(+1/-1)], plus (H,W), T."""
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.keys()); cfg = format_cfg or {}

    def finish(x, y, p, t_raw):
        t_step = _infer_time_steps(t_raw, cfg.get("time_unit"))
        p_pm1  = _polarity_to_pm1(p)
        x = x.astype(np.int64); y = y.astype(np.int64)
        ev = np.stack([t_step, x, y, p_pm1], axis=1)
        H = int(y.max(initial=0)) + 1
        W = int(x.max(initial=0)) + 1
        T = int(t_step.max(initial=0)) + 5 if ev.size else 100
        return ev, (H, W), T

    # explicit mapping
    if "keys" in cfg:
        k = cfg["keys"]
        x, y, t = data[k["x"]], data[k["y"]], data[k["t"]]
        p = data[k["p"]] if "p" in k and k["p"] in data else np.ones_like(x, dtype=np.int64)
        return finish(x, y, p, t)

    # common layouts
    if "events" in keys:
        ev = data["events"]
        # structured array?
        if getattr(ev, "dtype", None) is not None and ev.dtype.fields:
            fields = set(ev.dtype.fields.keys())
            def pick(*c):
                for name in c:
                    if name in fields: return name
                    if name.lower() in fields: return name.lower()
                    if name.upper() in fields: return name.upper()
                return None
            fx = pick("x"); fy = pick("y"); fp = pick("p"); ft = pick("t","ts","time")
            if fx and fy and ft:
                x, y = ev[fx], ev[fy]
                p = ev[fp] if fp else np.ones_like(x, dtype=np.int64)
                t = ev[ft]
                return finish(x, y, p, t)
        # matrix?
        ev = np.asarray(ev)
        if ev.ndim == 2 and ev.shape[1] >= 4:
            t, x, y, p = ev[:,0], ev[:,1], ev[:,2], ev[:,3]
            if t.max(initial=0) < max(x.max(initial=0), y.max(initial=0)):  # maybe [x,y,p,t]
                x, y, p, t = ev[:,0], ev[:,1], ev[:,2], ev[:,3]
            return finish(x, y, p, t)

    # separate arrays
    if {"x","y","t"}.issubset(keys):
        x, y, t = data["x"], data["y"], data["t"]
        p = data["p"] if "p" in keys else np.ones_like(x, dtype=np.int64)
        return finish(x, y, p, t)

    raise ValueError(f"Unrecognized NPZ structure. Keys={sorted(keys)}")
