from __future__ import annotations
import numpy as np
from typing import Tuple

class EventEncoder:
    def __init__(self, in_shape: Tuple[int,int], T:int, use_polarity=True):
        self.H,self.W=in_shape; self.T=int(T); self.use_polarity=use_polarity
        self.N=self.H*self.W*(2 if use_polarity else 1)
    def _index(self,x:int,y:int,p:int)->int:
        base=y*self.W+x
        if self.use_polarity: ch=0 if p>0 else 1; return base*2+ch
        return base
    def from_events(self,events:np.ndarray)->np.ndarray:
        spikes=np.zeros((self.T,self.N),dtype=np.bool_)
        if events is None or len(events)==0: return spikes
        for t,x,y,p in events.astype(np.int64):
            if 0<=t<self.T and 0<=x<self.W and 0<=y<self.H:
                spikes[t,self._index(x,y,p)]=True
        return spikes
