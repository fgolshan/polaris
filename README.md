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

## Notes

- This repo is intended as a companion artifact to the thesis.
- For SCION build and runtime details, defer to the documentation within the respective sub-repositories.
