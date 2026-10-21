# mobility-affects-disease
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23407173.svg)](https://doi.org/10.5281/zenodo.23407173)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/heykuldip/mobility-affects-disease)](https://github.com/heykuldip/mobility-affects-disease/releases)
[![Container](https://img.shields.io/badge/Container-Zenodo-blue)](https://doi.org/10.5281/zenodo.23406362)

An MPI-parallel agent-based simulation, built on [Repast4Py](https://github.com/Repast/repast4py), of how
mobility assumptions shape contact networks and epidemic outcomes. Five mobility models move a synthetic
population of 1,000,000 agents through 347,120 places in Fairfax County, Virginia, in hourly steps over
50 simulated days, with identical disease parameters.

This is the code for:

> Atwal KS, Von Hoene E, Amiri H, Pfoser D, Züfle A, Anderson T. *Mobility and Contact Networks Shape
> Epidemic Outcomes: A Large-Scale Agent-Based Modeling Study.* arXiv:2607.26217 (2026).

The code for the revised paper (arXiv v2) is tagged `v2.0-paper`. It is archived on Zenodo as [10.5281/zenodo.23407174](https://doi.org/10.5281/zenodo.23407174). It corrects three issues in the original version, tagged `v1.0-paper`, whose results it supersedes: the random-number generators are now seeded independently on each process, so runs are reproducible; contacts are chosen from a random ordering of the agents present, fixed for the run; and the R0 counter now counts only new infections.

------------------------------------------------------------------------

## Mobility models

| Mobility model (paper) | Script | Daily activities | Destination choice |
|---|---|---|---|
| NHTS Advan | `nhts_advan.py` | NHTS trip chains | Advan foot-traffic weights; workplaces from LODES commuting flows |
| NHTS Distance | `nhts_distance.py` | NHTS trip chains | Distance decay from home |
| NHTS Random | `nhts_random.py` | NHTS trip chains | Uniform within the activity's place type |
| Random Advan | `random_advan.py` | Random activities, uniform dwell times | Advan foot-traffic weights for public places and schools; homes and workplaces uniform |
| Random Places | `random_places.py` | No activity structure, uniform dwell times | Uniform over all places |

Each script has a matching `config/*.yaml`.

## Project structure

``` text
README.md
LICENSE
nhts_advan.py
nhts_distance.py
nhts_random.py
random_advan.py
random_places.py
places.py
human_agent/
  ├─ human_agent.py
  └─ disease.py
config/
  ├─ nhts_advan.yaml
  ├─ nhts_distance.yaml
  ├─ nhts_random.yaml
  ├─ random_advan.yaml
  └─ random_places.yaml
scripts/
  └─ run_sweep.sh             the paper's runs on Slurm
analysis/
  ├─ submit_networks.sh       contact networks and mobility measures, one Slurm job per run
  ├─ networks.py
  ├─ second_waves.py          whether each run's epidemic recurs after its first wave
  ├─ postprocessing.ipynb     Tables 1-4 and Figs. 1 and 3
  └─ destination_weights.py   Fig. 2 (needs the licensed Advan inputs)
input/            (created by you)
output/           (created after runs)
```

------------------------------------------------------------------------

## 1. Requirements

- **Container (used for the paper):** Singularity or Apptainer. The MPICH inside the image starts the
  processes, so no host MPI is needed.
- **Native install (alternative):** Python 3.11, an MPI implementation (MPICH or Open MPI) with its
  compiler wrappers (`mpicc`, `mpicxx`), and pipenv.
- **Analysis:** Python 3 with numpy, pandas, matplotlib, pyyaml and python-igraph (the paper's analysis
  used Python 3.12.8).

Tested versions (inside the container):

- Python 3.11.4
- mpi4py 4.0.0
- Repast4Py 1.1.6
- MPICH 3.4.3

A full-scale run needs one node with about 200 GB of memory (the paper’s runs requested 488 GB for Random Advan)
and 66–104 GB of disk for its log. See [Resource requirements](#8-resource-requirements).

------------------------------------------------------------------------

## 2. Get the code

``` bash
git clone https://github.com/heykuldip/mobility-affects-disease.git
cd mobility-affects-disease
git checkout v2.0-paper     # the version used for the revised paper
mkdir -p input output
```

------------------------------------------------------------------------

## 3. Input data

The input files are not included in this repository. Each config in `config/` names the files it reads
from `input/`:

| File | Contents |
|---|---|
| `ffx_agents_trips_*.csv` | Agents with census tract, type (worker, student or homemaker) and daily trip chain, including home, work and school locations |
| `input_buildings_ffx.csv` | The 347,120 places (building footprints and points of interest), with type and coordinates |
| `dwell_time_*.json` | Activity start times and durations for the NHTS-based models |
| `restaurants_dict*.json`, `recreation_dict*.json`, `errands_dict*.json`, `hospitals_dict*.json`, `schools_dict.json` | Candidate destinations and selection probabilities by census tract |

For `scripts/run_sweep.sh`, put each model's files in its own folder, `<INPUTS>/<folder>/input/`, where
the folders are `nhts_advan`, `nhts_distance`, `nhts_random`, `random_advan` and `random_all` (Random
Places). The distance-decay variants' files (Fig. 3) go in `<INPUTS>/nhts_distance/distance_decay_input/`.

The inputs are built from:

- 2017 National Household Travel Survey (Federal Highway Administration), public
- 2015 LEHD Origin-Destination Employment Statistics (U.S. Census Bureau), public
- Fairfax County building footprints (Fairfax County Open Geospatial Data), public
- Census tract totals used to synthesize the population, public
- 2019 Advan Weekly Patterns and points of interest, licensed through Dewey

Because the derived inputs include licensed Advan data, they can't be redistributed here. For access,
contact the corresponding author listed in the paper or open an issue in this repository; access is
subject to the data license.

------------------------------------------------------------------------

## 4. Run with the container (used for the paper)

The software environment is a public image:

- Image: `ghcr.io/heykuldip/mobility-affects-disease:1.1.0`
- Digest: `sha256:6b2c7a00372cd8800b7c435b7d736150c95886e1ce3bca9165c739a8be8b079d`
- File the runs used: `abm-1.1.0.sif`, archived as [10.5281/zenodo.23406362](https://doi.org/10.5281/zenodo.23406362)

Build a Singularity image once (Apptainer accepts the same command):

``` bash
singularity build abm-1.1.0.sif docker://ghcr.io/heykuldip/mobility-affects-disease@sha256:6b2c7a00372cd8800b7c435b7d736150c95886e1ce3bca9165c739a8be8b079d
```

The MPICH inside the image starts the processes. Set `random.seed` in the config first: the model stops
without it.

``` bash
singularity exec --cleanenv abm-1.1.0.sif mpiexec -n 64 python nhts_advan.py config/nhts_advan.yaml
```

`--cleanenv` keeps your shell's environment, such as a `PYTHONPATH`, out of the container. Check MPI
before a long run; each process should print a different rank and the same size:

``` bash
singularity exec --cleanenv abm-1.1.0.sif mpiexec -n 2 \
    python -c "from mpi4py import MPI; c = MPI.COMM_WORLD; print(c.Get_rank(), c.Get_size())"
```

If every process prints `0 1`, the processes didn't form one MPI job.

------------------------------------------------------------------------

## 5. Run with Slurm (as for the paper)

The paper's runs, 30 per mobility model with base seeds 1–30, and the four distance-decay variants of
Fig. 3, were made with `scripts/run_sweep.sh`. Each run uses 64 MPI processes on one node. Run it from the
repository root with the model files committed, so each run records the commit it ran:

``` bash
export INPUTS=/path/to/inputs          # the folders described in section 3
export IMG=/path/to/abm-1.1.0.sif
bash scripts/run_sweep.sh sweep         # builds the run folders and shows the sbatch commands
bash scripts/run_sweep.sh sweep --submit
bash scripts/run_sweep.sh decay --submit
```

Results go to `$OUT/<sweep|decay>/<model>/runs/run_NN/` (by default `OUT=/scratch/$USER/abm`). The script
sets each model's partition and memory; override them with `PARTITION` and `MEMORY`, and see its header
for the other settings.

------------------------------------------------------------------------

## 6. Native install with pipenv (alternative)

If your cluster uses environment modules, load Python, MPI and compiler modules **before** creating the
environment, so the builds bind to the MPI you will run with.

``` bash
python -m pip install --user pipenv
pipenv --python 3.11
pipenv run pip install mpi4py
env CC=mpicxx pipenv run pip install repast4py==1.1.6
pipenv run pip freeze > requirements-freeze.txt   # optional: record exact versions
```

`CC=mpicxx` is required: Repast4Py compiles native MPI C++ code, which must be built with your MPI's C++
compiler wrapper.

Quick checks:

``` bash
pipenv run python -c "import mpi4py, repast4py; print('ok')"
pipenv run mpirun -n 2 python -c "from mpi4py import MPI; c = MPI.COMM_WORLD; print(c.Get_rank(), c.Get_size())"
```

Run interactively, replacing `64` with the number of cores you can use (`mpiexec` works too). As in
section 4, set `random.seed` in the config first.

``` bash
pipenv run mpirun -n 64 python nhts_advan.py config/nhts_advan.yaml
pipenv run mpirun -n 64 python nhts_distance.py config/nhts_distance.yaml
pipenv run mpirun -n 64 python nhts_random.py config/nhts_random.yaml
pipenv run mpirun -n 64 python random_advan.py config/random_advan.yaml
pipenv run mpirun -n 64 python random_places.py config/random_places.yaml
```

------------------------------------------------------------------------

## 7. Outputs and post-processing

Each run writes, next to its log file:

- `agent_log0.csv`: a row per agent per simulated hour, with `tick`, `agent_id`, `disease_state`, `x`,
  `y`, `n_infected` and `agents_contacted` (that hour's contacts). Set `write_agent_log: false` in the
  config to skip it.
- `summary_ticks.csv`: the agents in each disease state, and the new infections, per hour.
- `initial_cases.csv`: the infections caused by each initial case, from which R0 is computed.
- `run_info.json`: the run's seeds, R0 and resource use.

Then:

``` bash
bash analysis/submit_networks.sh $OUT/sweep --submit    # network.json and mobility.json in each run folder
bash analysis/submit_networks.sh $OUT/decay --submit
python analysis/second_waves.py $OUT/sweep
SWEEP_DIR=$OUT/sweep DECAY_DIR=$OUT/decay jupyter nbconvert --to notebook --execute analysis/postprocessing.ipynb
```

The notebook produces Tables 1–4 and Figs. 1 and 3, and is saved with the paper's results.
`analysis/destination_weights.py` produces Fig. 2 from the licensed Advan inputs.

------------------------------------------------------------------------

## 8. Resource requirements

A run takes a few hours on one node with 64 MPI ranks and writes a 66–104 GB log. All 150 runs (30 per
mobility model) produce about 12 TB, so point `OUT` at scratch storage. Building a run's network reads its
whole log, taking about 1.5 hours and up to about 100 GB of memory; Random Advan's much denser networks
need several times more, which is why the paper's network measures for it come from five runs.

------------------------------------------------------------------------

## 9. Troubleshooting

- **`undefined symbol` when importing `repast4py`** (for example `_ZN3MPI8Datatype4FreeEv`): the compiled
  extension no longer matches the MPI that is loaded, typically after a cluster module update. Load the
  MPI module Repast4Py was built against, or rebuild it against the current one with
  `env CC=mpicxx pipenv run pip install --force-reinstall --no-deps --no-cache-dir repast4py==1.1.6`.
  The container avoids this.
- **Every process reports rank 0:** the MPI that launches the job doesn't match the one mpi4py was built
  against.
- **`mpirun` uses a different Python than pipenv:** check with `pipenv run mpirun -n 2 which python`.
- **Missing shared libraries:** usually module load order; load MPI and compiler modules before building.
- **Job fails at start-up:** you requested more MPI ranks than the allocation provides.

Useful checks:

``` bash
pipenv run which python
pipenv run python -V
which mpirun
pipenv run mpirun -n 2 which python
pipenv run python -c "import mpi4py, repast4py; print('imports ok')"
```

------------------------------------------------------------------------

## 10. Reproducibility notes

- The paper reports results from 30 runs per mobility model (base seeds 1–30), each with 64 MPI processes.
- A run is reproducible from its base seed (`random.seed`): every process seeds its generators from it, so
  rerunning with the same seed and number of processes gives the same results.
- The released code reproduces the paper's runs: the first two simulated days of each model's first run
  match the original runs exactly.

------------------------------------------------------------------------

## Citation

If you use this code or methodology in your research, please cite:

``` bibtex
@article{atwal2026mobility,
  title   = {Mobility and Contact Networks Shape Epidemic Outcomes: A Large-Scale Agent-Based Modeling Study},
  author  = {Atwal, Kuldip Singh and Von Hoene, Emma and Amiri, Hossein and Pfoser, Dieter and Z{\"u}fle, Andreas and Anderson, Taylor},
  journal = {arXiv preprint arXiv:2607.26217},
  year    = {2026}
}
```

For reproducibility, also cite the Zenodo DOI of the specific software version used in your analysis.

## License and acknowledgments

Released under the GNU General Public License v3.0; see `LICENSE`.

Built on Repast4Py (Collier N, Ozik J. Distributed agent-based simulation with Repast4Py. *Winter
Simulation Conference*, 2022). Computing resources were provided by the Office of Research Computing at
George Mason University (https://orc.gmu.edu).
