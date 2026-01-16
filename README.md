
#     mobility_affects_disease

This repository contains an MPI-parallel, Repast4Py-based simulation
framework for running multiple mobility scenarios and reproducing
disease outcome.

------------------------------------------------------------------------

## Project Structure

``` text
README.md
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

## 1. Prerequisites

Tested versions: - Python 3.11.4 - Repast4Py 1.1.6 - MPICH 3.3.2

You must ensure compatible versions on your machine or cluster. Use an
isolated environment. We use **pipenv** in this guide.

You need: - Python 3.11.x - An MPI implementation (MPICH or OpenMPI) -
Compiler toolchain (GCC/Clang with MPI wrappers)

------------------------------------------------------------------------

## 2. Environment Setup with Pipenv

Install pipenv if needed:

``` bash
python -m pip install --user pipenv
```

From the repository root, create the environment using Python 3.11:

``` bash
pipenv --python 3.11
```

Enter the environment:

``` bash
pipenv shell
```

All remaining commands should be executed inside this shell.

If your cluster uses environment modules, load Python, MPI, and compiler
modules **before** creating the pipenv so that builds bind to the
correct MPI stack.

------------------------------------------------------------------------

## 3. Install Repast4Py with MPI

Confirm MPI availability:

``` bash
mpirun --version
which mpicc
```

Install MPI Python bindings so they compile against your MPI:

``` bash
pipenv run pip install mpi4py
```

Install Repast4Py:

``` bash
pipenv run pip install repast4py==1.1.6
```

Optional: freeze dependencies for reproducibility:

``` bash
pipenv lock
```

Quick checks:

``` bash
pipenv run python -c "import mpi4py, repast4py; print('ok')"
pipenv run mpirun -n 2 python -c "from mpi4py import MPI; print('rank', MPI.COMM_WORLD.Get_rank())"
```

------------------------------------------------------------------------

## 4. Get the Code and Inputs

Clone the repository and create the `input` directory:

``` bash
git clone <REPO_URL>
cd <REPO_DIR>
mkdir -p input
```

Copy the provided input files into:

``` text
<repo_root>/input/
```

------------------------------------------------------------------------

## 5. Run Simulations (Interactive)

Replace `64` with the number of cores you can use.

``` bash
pipenv run mpirun -n 64 python nhts_advan.py config/nhts_advan.yaml
pipenv run mpirun -n 64 python nhts_random.py config/nhts_random.yaml
pipenv run mpirun -n 64 python nhts_distance.py config/nhts_distance.yaml
pipenv run mpirun -n 64 python random_advan.py config/random_advan.yaml
pipenv run mpirun -n 64 python random_places.py config/random_places.yaml
```

Use `mpiexec` if your system prefers it.

------------------------------------------------------------------------

## 6. Run with Slurm

Slurm scripts are provided for each scenario. Edit them to match your
cluster resources:

-   partition or queue
-   account
-   walltime
-   nodes and tasks
-   memory

Ensure each script activates the pipenv environment or uses `pipenv run`
before launching MPI.

Submit jobs:

``` bash
sbatch nhts_advan.slurm
sbatch nhts_random.slurm
sbatch nhts_distance.slurm
sbatch random_advan.slurm
sbatch random_places.slurm
```

------------------------------------------------------------------------

## 7. Outputs and Post-Processing

After all runs complete, an `output/` directory appears at the repo root
containing logs for each scenario.

Use the Jupyter notebook to regenerate disease outcomes:

``` bash
pipenv run jupyter notebook
```

Open `process_output.ipynb` and run all cells. The notebook is saved
with reference results for comparison.

------------------------------------------------------------------------

## 8. Troubleshooting

Common issues: - MPI mismatch between build and runtime - `mpirun` using
a different Python than pipenv - Missing shared libraries due to module
order - Requesting more ranks than available

Useful checks:

``` bash
pipenv run which python
pipenv run python -V
which mpirun
pipenv run mpirun -n 2 which python
pipenv run python -c "import mpi4py, repast4py; print('imports ok')"
```

------------------------------------------------------------------------

## 9. Reproducibility via Containers

Recommendation: - Use Singularity or Apptainer for HPC. - Optionally
provide Docker for local use.

``` bash
mpirun -n <N> python nhts_advan.py config/nhts_advan.yaml
```

------------------------------------------------------------------------

## 10. Validation

All mobility scenarios were tested after cleaning the code. You should
be able to reproduce the disease outcomes in `process_output.ipynb` once
simulation outputs are generated.

If something fails or looks inconsistent, inspect the logs in `output/`
