from __future__ import annotations


import os, sys
from pathlib import Path
import numpy as np


sys.path.insert(0, os.path.dirname(__file__))

# importing all packages
from core.neurons import LIFConfig
from core.stdp import STDPConfig
from network import Network
from visualize.diagram import draw_fully_connected

# our modules
## loaders: for now it integrates .npz to unified forrmat. {t, x, y, z}
from snn_io.events_loader import load_events_generic

## stream: binned, sparse per timestamp indices
from encoding.binned_stream import BinnedEventStream

## runner: simulation loop and handling homeostasis + soft-WTA
from engine.runners import run_stream_counts_with_competition_and_homeostasis

## analysis: summarize the output spikes, for analysis and post processing
from analysis.output_analysis import analyze_output


def main():
    # === set your event file here ===
    ## have to give a folder next: deisgning a pipeline and a dataloader for this
    event_path = Path("test_data/0000.npz")
    outdir = Path("runs/last_run")
    # ================================

    # Tuned defaults
    BIN_MS   = 10       ## time bin size for binned stream
    dt       = 1.0      ## simulation timestep (ms), simulator use dt in LIF update
    delay_ms = 1.0      ## synaptic delay (ms) used by syanptic ring buffers for updates.
    use_pol  = True     ## whether to use polarity channels in input encoding
    n_hidden = 256      ## hidden layer size
    n_out    = 10
    seed     = 0        ## random seed for weight init: for reproducibility

    # Load events 
    if event_path.exists():
        events, in_shape, T_raw = load_events_generic(event_path)
        print(f"Loaded events from {event_path}")
        print(f"Inferred in_shape={in_shape}, T_raw(ms)={T_raw}, events={len(events)}")
    else:
        print(f"WARNING: {event_path} not found. please check the path")
        # in_shape = (32, 32)
        # H, W = in_shape
        # events = np.array([[10, W//2, H//2, 1]], dtype=np.int64)
        # T_raw = 100

    # Stream + bin (sparse)
    stream = BinnedEventStream(events, in_shape, use_polarity=use_pol, bin_ms=BIN_MS)
    T = stream.T_bins
    N_in = stream.N
    print(f"Streaming with bin={BIN_MS} ms -> T_bins={T}, N_in={N_in}")

    # Build network (excitable LIF + STDP)
    lif  = LIFConfig(v_thresh=0.4, tau_mem=10.0)
    stdp = STDPConfig(A_plus=0.01, A_minus=0.012, tau_plus=20.0, tau_minus=20.0, w_min=0.0, w_max=2.0)
    net  = Network(N_in, n_hidden, n_out, dt=dt, delay_ms=delay_ms, lif_cfg=lif, stdp_cfg=stdp, seed=seed)

    # Stronger weights + small jitter to break symmetry
    net.s_in_h.W *= 2.5
    net.s_h_o.W  *= 2.0
    rng = np.random.default_rng(0)
    net.s_h_o.W *= (0.9 + 0.2 * rng.random(net.s_h_o.W.shape)).astype(np.float32)

    # Run counts-only with soft-WTA + homeostasis
    net.sim.reset_state()
    counts_in, counts_h, counts_out = run_stream_counts_with_competition_and_homeostasis(
        net.sim, T=T, input_group_ix=0, sparse_iter=stream.iter_sparse(),
        bin_ms=BIN_MS,
        bias_hidden=0.02,
        bias_output_init=0.0,
        soft_wta_gain=0.9,
        target_rate_hz=5.0,
        homeo_tau_s=0.5,
        homeo_k=0.05,
        progress_every=1000
    )

    print("Output spike counts:", counts_out)

    # Save minimal artifacts (counts + weights)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "run_data_counts_only.npz",
             spike_counts_input=counts_in,
             spike_counts_hidden=counts_h,
             spike_counts_output=counts_out,
             W_in_h=net.s_in_h.W,
             W_h_o=net.s_h_o.W)

    # Output analysis (prints + saves txt/csv)
    analyze_output(counts_out, T_bins=T, bin_ms=BIN_MS, path_dir=outdir)

    # Diagram only if small input (avoid 80k-node figure)
    if N_in <= 1000:
        draw_fully_connected(N_in, n_hidden, n_out, str(outdir / "architecture.png"))

    print(f"Saved lightweight results + analysis to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
