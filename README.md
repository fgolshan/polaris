# Polaris – Thesis Experiment Repository

This repository accompanies the master thesis:

**“A Stepping Stone Toward Inter-Domain Traffic Engineering in a Path-Aware Data Plane.”**

It ties together (i) the SCION codebase and applications extended for Polaris + traffic engineering (TE), and (ii) the scripts and data used to run and reproduce the thesis experiments.

## Repositories (submodules)

This repo uses Git submodules:

- **`scionproto/`** – modified SCION code base with added support for **Polaris** and **traffic engineering (TE)**.
- **`scion-apps/`** – SCION applications with an added **Polaris end-host**, based on the SCION bandwidth tester, with added congestion control. The end host also contains **TE** support.
- **`seed-emulator/`** – the upstream SEED Emulator code base used by the experiment scripts (included for convenience; no thesis-specific changes were made here).

> All thesis-relevant commits were performed by **Fariborz Golshani**.

## Experiments (`networks/`)

The `networks/` folder contains:

- **Topology definitions / setup scripts** for running SCION networks under emulation (e.g., `networks/polaris-topo/`).
- **Automated experiment runners** that orchestrate end-to-end runs (traffic generation, collection of outputs, and plotting).
- **Traffic generation** via **`traffic_generator_v3.py`**, which automates SCION bandwidth-tester sessions and supports timed “events” (e.g., bandwidth changes or TE rule pushes during a run).
- **Plotting and analysis scripts**, including a script for plotting **network snapshots**.
- **Results** from the performed experiments (as produced by the scripts).

## How to use (high level)

1. **Set up the dependencies** by following the instructions in each sub-repository:
   - `scionproto/`
   - `scion-apps/`
   - `seed-emulator/`

2. **Run the experiment tooling** from `networks/` as needed.
   - Most scripts support `--help` for usage and arguments.
   - Typical workflow: generate/launch a topology → replace relevant binaries → run automated experiments → plot results.

## Getting started (quickstart)

This repo is a meta-repository that ties together multiple components via Git submodules (`scionproto/`, `scion-apps/`, `seed-emulator/`) and provides experiment tooling under `networks/`. The steps below get you from a fresh clone to a first end-to-end experiment run.

---

### 1) Clone this repository (including submodules)

Clone **with all submodules** in one command:

```bash
git clone --recurse-submodules https://github.com/fgolshan/polaris.git
cd polaris
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

---

### 2) Build SCION (router code with Polaris + basic TE support)

Build the SCION code from the `scionproto/` submodule by following the build instructions in that code base:

```bash
cd scionproto
# follow scionproto's build instructions (README / docs in that submodule)
```

This produces the SCION router binaries with Polaris support and basic traffic engineering (TE) functionality.

---

### 3) Compile the Polaris/TE-capable bandwidth tester end host (static binary)

Compile the SCION bandwidth tester from the `scion-apps/` submodule. The easiest way is to run the helper script in the **submodule root**:

```bash
cd ../scion-apps
./build_my_bwtester.sh
```

This builds a **static** binary of the Polaris + TE-capable bandwidth tester that can be copied into the emulated environment and used for experiments.

---

### 4) Set up and launch SEED Emulator (SCION topology)

Set up the emulator by following the instructions in the `seed-emulator/` submodule:

```bash
cd ../seed-emulator
# follow seed-emulator setup instructions (README / docs in that submodule)
```

Create your own SCION topology or use an existing one from `./networks/`. For example:

- `./networks/polaris-topo-full`

Detailed instructions for writing, compiling, and launching SCION topologies are provided in the `seed-emulator` submodule documentation.

---

### 5) Replace binaries inside the running network and restart routers

Once your SCION network is running, replace the binaries inside the emulated environment with the ones you built above and restart the routers.

Use `replace_binaries.py`, for example:

```bash
cd ../networks
python replace_binaries.py --debug --prometheus :9000
```

Use Prometheus port `:9000` to stay consistent with the other scripts.

---

### 6) Sanity-check connectivity and available paths

Open the emulator Internet Map:

- http://127.0.0.1:8080/map.html

Launch a console on a node and verify:

- connectivity with `scion ping`
- path availability with `scion showpaths`

---

### 7) Run your first experiment 🥳

You’re now ready to run an end-to-end experiment using `run_single_experiment.py`.

The script will:
- generate traffic flows according to a JSON specification,
- optionally apply traffic engineering (TE) events and/or bandwidth capping,
- collect client/server logs and router metrics (Prometheus),
- produce summary files and plots.

#### Find or create experiment specs

Example JSON specifications are available in:
- `./networks/experiments/` (e.g., `testspec.json`, `pcaspec.json`, `tcspec.json`)
- `./networks/tests/`

A convenient starting point is to create a temporary test folder and copy templates into it:

```bash
mkdir -p networks/tests/tmp
cp -r networks/experiments/traffic_engineering/pca_vs_bwcap_templates/pca_template/* networks/tests/tmp/
```

#### Run a single experiment

From the `networks/` folder, run:

```bash
cd networks
python run_single_experiment.py tests/tmp/testspec.json tests/tmp/ --capacity 2.0 --pcaspec tests/tmp/pcaspec.json
```

Notes:
- `--capacity 2.0` assumes you run a topology with **2 Mbps links** (e.g., `polaris-topo-full`). Adjust or omit this flag if your topology uses different capacities.
- Capacity is used to normalize w.r.t. link capacity in some plots.

#### Inspect outputs

After the experiment finishes, check `networks/tests/tmp/`. You should find:
- client and server output logs,
- a `metrics/` folder with Prometheus metrics of all routers (plus aggregates and visualizations), including router topology files,
- `loss_summary.txt` and `loss_stats.csv`,
- `loss_stats.csv`,
- a path switch timeline plot.

Optional: generate network flow distribution snapshot plots for each path switch:

```bash
python draw_snapshots.py tests/tmp/
```

## Notes

- This repo is intended as a companion artifact to the thesis.
- For SCION build and runtime details, defer to the documentation within the respective sub-repositories.
