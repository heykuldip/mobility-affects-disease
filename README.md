# mobility-affects-disease

An MPI-parallel agent-based simulation, built on [Repast4Py](https://github.com/Repast/repast4py), of how
mobility assumptions shape contact networks and epidemic outcomes. Five mobility scenarios move a synthetic
population of 1,000,000 agents through 347,120 places in Fairfax County, Virginia, in hourly steps over
50 simulated days, with identical disease parameters.

This is the code for:

> Atwal KS, Von Hoene E, Amiri H, Pfoser D, Züfle A, Anderson T. *Mobility and Contact Networks Shape
> Epidemic Outcomes: A Large-Scale Agent-Based Modeling Study.* arXiv:2607.26217 (2026).

The version used for the paper is tagged `v1.0-paper`.

------------------------------------------------------------------------

## Scenarios

| Scenario (paper) | Script | Daily activities | Destination choice |
|---|---|---|---|
| NHTS Advan | `nhts_advan.py` | NHTS trip chains | Advan foot-traffic weights |
| NHTS Distance | `nhts_distance.py` | NHTS trip chains | Distance decay from home |
| NHTS Random | `nhts_random.py` | NHTS trip chains | Uniform within the activity's place type |
| Random Advan | `random_advan.py` | Random activities, uniform dwell times | Advan foot-traffic weights |
| Random Places | `random_places.py` | No activity structure, uniform dwell times | Uniform over all places |

Each script has a matching `config/*.yaml` and `*.slurm` file.

## Project structure

``` text
README.md
LICENSE
nhts_advan.py
nhts_advan.slurm
nhts_random.py
nhts_random.slurm
nhts_distance.py
nhts_distance.slurm
random_advan.py
random_advan.slurm
random_places.py
random_places.slurm
process_output.ipynb
human_agent/
  ├─ human_agent.py
  └─ disease.py
config/
  ├─ nhts_advan.yaml
  ├─ nhts_random.yaml
  ├─ nhts_distance.yaml
  ├─ random_advan.yaml
  └─ random_places.yaml
input/            (created by you)
output/           (created after runs)
```

------------------------------------------------------------------------

## 1. Requirements

- **Container (used for the paper):** Singularity or Apptainer, plus a host MPI that is ABI-compatible
  with the MPICH inside the image. The paper's runs used `mpich/3.3.2-ofi` on the host.
- **Native install (alternative):** Python 3.11, an MPI implementation (MPICH or Open MPI) with its
  compiler wrappers (`mpicc`, `mpicxx`), and pipenv.

Tested versions:

- Python 3.11.4
- Repast4Py 1.1.6
- MPICH 3.3.2

A full-scale run needs one node with about 200 GB of memory and 66–104 GB of disk for its log. See
[Resource requirements](#8-resource-requirements).

------------------------------------------------------------------------

## 2. Get the code

``` bash
git clone https://github.com/heykuldip/mobility-affects-disease.git
cd mobility-affects-disease
git checkout v1.0-paper     # the version used for the paper
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
| `dwell_time_*.json` | Activity start times and durations for the NHTS-based scenarios |
| `restaurants_dict*.json`, `recreation_dict*.json`, `errands_dict*.json`, `hospitals_dict*.json`, `schools_dict.json` | Candidate destinations and selection probabilities by census tract |

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

The software environment is a public image built from
[GMU-GeoSciences/repast4py-container](https://github.com/GMU-GeoSciences/repast4py-container):

- Image: `ghcr.io/gmu-geosciences/repast4py-container:v1.0.4`
- Digest: `sha256:cdc412e4b0969108414055b15cb94cf040c566903184b882d207c8a3d5cbed44`

The paper's runs used an image built from the `latest` tag, which points to this same image.

Build a Singularity image once (Apptainer accepts the same command):

``` bash
singularity build repast4py_v1.0.4.sif docker://ghcr.io/gmu-geosciences/repast4py-container:v1.0.4
```

The host's MPI then starts the processes inside the container:

``` bash
mpirun -n 64 singularity run repast4py_v1.0.4.sif python nhts_advan.py config/nhts_advan.yaml
```

This works because the host MPI and the MPICH in the image share an ABI. Check it before a long run;
each process should print a different rank and the same size:

``` bash
mpirun -n 2 singularity run repast4py_v1.0.4.sif \
    python -c "from mpi4py import MPI; c = MPI.COMM_WORLD; print(c.Get_rank(), c.Get_size())"
```

If every process prints `0 1`, the host MPI and the image's MPI don't match.

------------------------------------------------------------------------

## 5. Run with Slurm (as for the paper)

The paper's 30 runs per scenario were submitted as a Slurm job array. Each task was one independent
50-day simulation using 64 MPI ranks on one node, and passed Repast4Py a JSON parameter override so
that it wrote its own log. The provided `*.slurm` scripts launch a single run; to run the full set,
add the array and override lines:

``` bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=200GB
#SBATCH --time=2-12:00:00
#SBATCH --array=0-29

# load your cluster's MPI and Singularity modules here
SIF=/path/to/repast4py_v1.0.4.sif
OVERRIDE=$(printf '{"agent_log_file": "output/agent_log0_%s"}' "$SLURM_ARRAY_TASK_ID")
mpirun -n 64 singularity run "$SIF" python ./nhts_advan.py ./config/nhts_advan.yaml "$OVERRIDE"
```

Edit the partition, account, memory, time limit and image path for your cluster, then submit:

``` bash
sbatch nhts_advan.slurm
sbatch nhts_distance.slurm
sbatch nhts_random.slurm
sbatch random_advan.slurm
sbatch random_places.slurm
```

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

Run interactively, replacing `64` with the number of cores you can use (`mpiexec` works too):

``` bash
pipenv run mpirun -n 64 python nhts_advan.py config/nhts_advan.yaml
pipenv run mpirun -n 64 python nhts_distance.py config/nhts_distance.yaml
pipenv run mpirun -n 64 python nhts_random.py config/nhts_random.yaml
pipenv run mpirun -n 64 python random_advan.py config/random_advan.yaml
pipenv run mpirun -n 64 python random_places.py config/random_places.yaml
```

To use this environment in the Slurm scripts, activate it (`source "$(pipenv --venv)/bin/activate"`) and
replace `singularity run <image> python` with `python`.

------------------------------------------------------------------------

## 7. Outputs and post-processing

Each run writes one CSV to `output/`, with a row per agent per simulated hour: `tick`, `agent_id`,
`disease_state`, `x`, `y`, `n_infected` and `agents_contacted` (that hour's contacts).

To regenerate the disease curves, install Jupyter and the packages the notebook imports (for example
`pipenv install jupyter pandas`), then open `process_output.ipynb` and run all cells. The notebook is
saved with reference results for comparison.

------------------------------------------------------------------------

## 8. Resource requirements

A run takes 2.3–3.3 hours on one node with 64 MPI ranks and writes 66–104 GB of output. All 150 runs
(30 per scenario) produce about 12 TB, so point `output/` at scratch storage.

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

- The paper reports results from 30 runs per scenario.
- Runs are stochastic, so results vary between runs; compare averages across runs rather than individual
  outputs.
- All scenarios were re-tested after the code was cleaned for release.

------------------------------------------------------------------------

## Citation

If you use this code, please cite:

``` bibtex
@article{atwal2026mobility,
  title   = {Mobility and Contact Networks Shape Epidemic Outcomes: A Large-Scale Agent-Based Modeling Study},
  author  = {Atwal, Kuldip Singh and Von Hoene, Emma and Amiri, Hossein and Pfoser, Dieter and Z{\"u}fle, Andreas and Anderson, Taylor},
  journal = {arXiv preprint arXiv:2607.26217},
  year    = {2026}
}
```

## License and acknowledgments

Released under the GNU General Public License v3.0; see `LICENSE`.

Built on Repast4Py (Collier N, Ozik J. Distributed agent-based simulation with Repast4Py. *Winter
Simulation Conference*, 2022). Computing resources were provided by the Office of Research Computing at
George Mason University (https://orc.gmu.edu).
