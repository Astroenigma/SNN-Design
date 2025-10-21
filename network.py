from __future__ import annotations
import numpy as np
from typing import Dict
from core.neurons import LIFNeuronGroup, LIFConfig
from core.synapse import SynapseGroup
from core.stdp import STDPConfig
from engine.simulator import Simulator

class Network:
    def __init__(self,in_size:int,n_hidden:int,n_out:int,dt:float,delay_ms=1.0,
                 lif_cfg:LIFConfig|None=None,stdp_cfg:STDPConfig|None=None,seed=0):
        rng=np.random.default_rng(seed); lif=lif_cfg or LIFConfig()
        self.input=LIFNeuronGroup(in_size,lif)
        self.hidden=LIFNeuronGroup(n_hidden,lif)
        self.output=LIFNeuronGroup(n_out,lif)
        W1=np.maximum(rng.normal(0,0.2,(in_size,n_hidden)),0).astype(np.float32)
        W2=np.maximum(rng.normal(0,0.2,(n_hidden,n_out)),0).astype(np.float32)
        self.s_in_h=SynapseGroup(self.input,self.hidden,W1,dt,delay_ms,stdp_cfg)
        self.s_h_o=SynapseGroup(self.hidden,self.output,W2,dt,delay_ms,stdp_cfg)
        self.groups=[self.input,self.hidden,self.output]
        self.syns=[self.s_in_h,self.s_h_o]
        self.sim=Simulator(dt,self.groups,self.syns)
    def reset(self): self.sim.reset_state()
    def run(self,T:int,input_train:np.ndarray)->Dict[str,np.ndarray]:
        r=self.sim.run(T,{0:input_train},True)
        return {'spikes_input':r[0],'spikes_hidden':r[1],'spikes_output':r[2],
                'W_in_h':self.s_in_h.W.copy(),'W_h_o':self.s_h_o.W.copy()}
