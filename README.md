# Midterm Coursework — Evolutionary Mountain-Climbing Creatures (PyBullet + Genetic Algorithm)

This repository contains my university **midterm coursework** implementing an evolutionary simulation where randomly generated “creatures” (URDF robots built from genomes) attempt to move/climb in a **PyBullet** physics environment. A **Genetic Algorithm (GA)** evolves better creatures across generations based on fitness (distance travelled).

It includes:
- A custom **genome encoding** for creature morphology + motor control
- A **Population** class for GA evolution (selection, crossover/mutation)
- A **PyBullet simulation** runner (single-thread & multi-process variants)
- A **Gymnasium environment** version (`MountainClimbEnv`) for RL-style interaction
- Utility scripts to save/load genomes from CSV and replay them
- Unit tests to demonstrate core functionality

A written report is also included: `Entropy_Regularization_Rewritten_FINAL_TOC.pdf`.

---

## Tech Stack

- Python 3.10+ (recommended)
- `pybullet`
- `numpy`
- `gymnasium` (for the environment wrapper)
- `noise` (used for terrain / procedural generation utilities)

---

## Project Structure

```
MIdterm Coursework/
├─ creature.py                  # Creature definition + motors
├─ genome.py                    # Genome representation + CSV save/load
├─ population.py                # GA population + breeding
├─ simulation.py                # PyBullet simulation runner (DIRECT)
├─ mountain_climb_env.py        # Gymnasium Env: MountainClimbEnv
├─ mountain_climb_env_updated.py
├─ cw-envt.py                   # Environment/arena helper script (GUI)
├─ prepare_shapes.py            # Shape/mesh preparation utilities
├─ offline_from_csv.py          # Replay a saved genome from a CSV (DIRECT)
├─ realtime_from_csv.py         # Replay a saved genome in real-time (GUI)
├─ shapes/                      # OBJ meshes used in the simulation
├─ test_*.py                    # Unit tests / runnable experiments
└─ Entropy_Regularization_Rewritten_FINAL_TOC.pdf
```

---

## Quick Start (Windows)

The original setup notes for Windows are:

1) Open **PowerShell as Administrator**  
2) Navigate to the project folder  
3) Create + activate a virtual environment:

```powershell
python -m venv pybullet_venv
pybullet_venv\Scripts\activate
```

4) Install dependencies (see below), then run:

```powershell
py mountain_climb_env.py
```

---

## Setup (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

---

## Install Dependencies

If you don’t have a `requirements.txt`, you can install the main libraries manually:

```bash
pip install pybullet numpy gymnasium noise
```

> If your system has issues building `noise`, try upgrading pip/wheel first:
> `pip install -U pip wheel setuptools`

---

## Running the Project

### 1) Run the Gymnasium environment (PyBullet + Gym wrapper)
```bash
python mountain_climb_env.py
```

This runs the `MountainClimbEnv` class which defines observations/actions and steps the simulation.

### 2) Run the Genetic Algorithm (recommended: no-threads version for Windows)
**Single-thread (more compatible):**
```bash
python test_ga_no_threads.py
```

**Multi-process (may not work on some setups):**
```bash
python test_ga.py
```

These scripts evolve a population over many iterations and print fitness statistics.  
They also save elite genomes as CSV files like `elite_<iteration>.csv`.

### 3) Replay a saved genome from CSV
**Offline replay (DIRECT):**
```bash
python offline_from_csv.py elite_10.csv
```

**Realtime replay (GUI):**
```bash
python realtime_from_csv.py elite_10.csv
```

---

## Fitness / Evaluation (High Level)

A creature’s performance is evaluated by running it in the simulator for a fixed number of steps (e.g., 2400).  
Fitness is measured using the creature’s **distance travelled** (see GA test scripts and creature distance tracking).

---

## Tests

Run all tests:

```bash
python -m unittest
```

Or run a specific test file:

```bash
python test_population.py
```
