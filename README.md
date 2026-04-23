# Project Overview
This project evaluates the performance and accuracy of hybrid Bayesian Neural Network (BNN) branch predictors within the ChampSim environment. 

# Repository Structure

├── cc_predictors
│   ├── hybrid_bnn_2bit
│   │   ├── hybrid_bnn_2bit.cc
│   │   └── hybrid_bnn_2bit.h
│   ├── hybrid_bnn_2bit_simple
│   │   ├── hybrid_bnn_2bit_simple.cc
│   │   └── hybrid_bnn_2bit_simple.h
│   ├── hybrid_bnn_bimodal
│   │   ├── hybrid_bnn_bimodal.cc
│   │   └── hybrid_bnn_bimodal.h
│   ├── hybrid_bnn_bimodal_simple
│   │   ├── hybrid_bnn_bimodal_simple.cc
│   │   └── hybrid_bnn_bimodal_simple.h
│   ├── hybrid_bnn_common.h
│   ├── hybrid_bnn_gshare
│   │   ├── hybrid_bnn_gshare.cc
│   │   └── hybrid_bnn_gshare.h
│   └── hybrid_bnn_gshare_simple
│       ├── hybrid_bnn_gshare_simple.cc
│       └── hybrid_bnn_gshare_simple.h
├── README.md
└── Research_Paper_Group6.pdf



# Setup and Trace Acquisition
The evaluation utilizes SPEC CPU 2006 traces. Use the following commands to download the necessary files into a traces/ directory:

## Download GCC Trace

curl -L -f https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/602.gcc_s-1850B.champsimtrace.xz -o traces/ 403.gcc-16B.champsimtrace.xz


## Download PERLBENCH Trace 

curl -L -f https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/600.perlbench_s-210B.champsimtrace.xz -o traces/600.perlbench_s-210B.champsimtrace.xz


## Configuration and Build
The champsim_config.json is configured for a single-core out-of-order processor. To switch between predictors, update the "branch_predictor" field with the folder name of the desired model.

champsim_config.json:

{
        "executable_name": "champsim",
        "block_size": 64,
        "page_size": 4096,
        "heartbeat_frequency": 1000000,
        "num_cores": 1,
        "branch_predictor": "<predictor's name>",
        "replacement": "lru",
        "L1I": { "sets": 64, "ways": 8, "prefetcher": "no" },
        "L1D": { "sets": 64, "ways": 12, "prefetcher": "no" },
        "L2C": { "sets": 1024, "ways": 8, "prefetcher": "no" },
        "LLC": { "sets": 2048, "ways": 16, "prefetcher": "no" }
}


## To compile the simulator:

rm -rf .csconfig
make clean
./config.sh champsim_config.json
make


# Execution Instructions
Run the following commands to simulate the branch predictors. Each run uses a 2-million instruction warmup followed by a 5-million instruction detailed simulation.

## RUN GCC:
./bin/champsim --warmup-instructions 2000000 --simulation-instructions 5000000 traces/602.gcc_s-1850B.champsimtrace.xz


./bin/champsim --warmup-instructions 2000000 --simulation-instructions 5000000 traces/403.gcc-16B.champsimtrace.xz


## RUN PERL:
./bin/champsim --warmup-instructions 2000000 --simulation-instructions 5000000 traces/600.perlbench_s-210B.champsimtrace.xz


## Evaluation Metrics
The primary metric for this assignment is Prediction Accuracy (%).
* Simple Policy: Invokes the BNN on all "weak" branch predictions.
* Complex Policy: Leverages BNN-MCD uncertainty gating (variance threshold) to only override the baseline when the neural model is highly confident.
