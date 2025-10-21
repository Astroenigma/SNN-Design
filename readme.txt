──────────────────────────────────────────────
                SHITTY_SNN
──────────────────────────────────────────────
Event-based Spiking Neural Network Framework
──────────────────────────────────────────────

1. OVERVIEW
------------
This repository implements a modular, biologically inspired
Spiking Neural Network (SNN) designed for event-based data
(e.g., DVS or neuromorphic camera streams).  
The system supports:
  • LIF (Leaky Integrate-and-Fire) neurons
  • STDP (Spike Timing-Dependent Plasticity)
  • Delayed synaptic transmission
  • Homeostatic output regulation
  • Soft-Winner-Take-All competition
  • Sparse event-based simulation with 10–50 ms bins

The framework is modular and production-ready, intended to
serve as a baseline for SNN research, event-data learning,
and hardware-software co-design.

──────────────────────────────────────────────
2. DIRECTORY STRUCTURE
──────────────────────────────────────────────

core/
 ├─ neurons.py     → LIF neuron model + configuration
 ├─ synapse.py     → Synapse group + delay buffers
 ├─ stdp.py        → STDP learning rule implementation

encoding/
 ├─ events_loader.py   → Universal NPZ event loader
 ├─ binned_stream.py   → Sparse event binning + iterator

engine/
 ├─ runners.py     → Simulation loop (soft-WTA + homeostasis)

analysis/
 ├─ output_analysis.py → Post-run analysis and logging

visualize/
 ├─ diagram.py     → Creates simple fully-connected layer diagram

test_data/
 ├─ 0000.npz       → Example DVS event file

runs/
 ├─ last_run/      → Output of most recent simulation

main.py            → Entry script. Adjust event_path & hyperparams here.
README.txt         → Overview (this file)
developer_logs.txt → Design notes and conceptual explanations

──────────────────────────────────────────────
3. QUICKSTART
──────────────────────────────────────────────

• Requirements: Python ≥3.9, NumPy, Matplotlib (for diagrams)
• To run:

      > python main.py

• Change the input event file path inside main.py:

      event_path = Path("test_data/0000.npz")

• The simulation will:
      1. Load event data
      2. Convert to sparse 10 ms bins
      3. Run an SNN for each bin
      4. Apply STDP + homeostasis + WTA
      5. Save counts and weight matrices

Output:
  • `runs/last_run/run_data_counts_only.npz`
  • `runs/last_run/output_analysis.txt`
  • `runs/last_run/output_analysis.csv`
  • (optional) `runs/last_run/architecture.png`

──────────────────────────────────────────────
4. DESIGN GOALS
──────────────────────────────────────────────

1. Modular structure:
   Each logical block (neurons, synapses, STDP, runner) is in a separate module.

2. Biologically Plausible:
   Uses LIF neurons with decay, refractory behavior,
   STDP with pre/post traces, and homeostasis for rate stability.

3. Efficient for Event Data:
   Operates on sparse indices (not dense tensors). Complexity ≈ O(#events).

4. Extensible:
   Easy to add new learning rules, neuron models, or encoders.

──────────────────────────────────────────────
5. DATA FLOW SUMMARY
──────────────────────────────────────────────

  NPZ Event Data
       ↓
  events_loader → standardizes [t, x, y, p]
       ↓
  binned_stream → yields (time_bin, active_input_neurons)
       ↓
  engine.runners → updates LIF neurons + STDP + homeostasis
       ↓
  analysis.output_analysis → produces summary and plots

──────────────────────────────────────────────
6. EXAMPLE OUTPUT
──────────────────────────────────────────────
Loaded events from test_data/0000.npz
Inferred in_shape=(200,200), T_raw(ms)=47367, events=6303278
Streaming with bin=10 ms -> T_bins=4737, N_in=80000
[sim] step 0/4737 ... [sim] step 9000/9473
Output spike counts: [213 210 211 209 213 212 212 213 212 213]
Winner index: 0 (count=213)
Avg firing rate: 44.7 Hz
Entropy (bits): 3.32
Top-5 units: [9, 7, 0, 4, 8]

──────────────────────────────────────────────
7. FUTURE EXTENSIONS
──────────────────────────────────────────────
• Add adaptive threshold neurons (ALIF)
• Add lateral inhibition in hidden layer
• Integrate Poisson spike encoder for static images
• GPU acceleration with Numba or Torch backend
──────────────────────────────────────────────
