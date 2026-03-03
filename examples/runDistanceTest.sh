#!/bin/bash
#requesting one node
#SBATCH -N1
#requesting 40 cpus
#SBATCH -n40
#SBATCH --mail-user=ucapmwe@ucl.ac.uk
#SBATCH --mail-type=ALL

# Request wallclock time (format days-hours:minutes:seconds or hours:minutes:seconds)
#SBATCH --time=2:00:00

# Set the name of the job
#SBATCH --job-name=distanceTest

# Request memory per node: 0 means all memory on node
#SBATCH --mem=0

# Set the working directory and output files
#SBATCH --chdir=/home/ucapmwe/dist-benchmark

# ==========================================================================

# --- MODULES & VENV --------------------------------------------------------
cd /home/ucapmwe/dist-benchmark

# activate venv
conda activate distTest

# --- SET ENVIRONMENT VARIABLES ---------------------------------------------
# Set number of OpenMP threads being used per MPI process: 1 = hyperthreading, 2 = single thread
# export OMP_NUM_THREADS=1


# --- LAUNCH MPI JOB --------------------------------------------------------

# # Generate Slurm hostfile (optional, for debugging)
# scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.slurm

# # Get node list as array
# nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# # Run on first node with desired params
# srun --nodes=1 --nodelist=${nodes[0]} --ntasks=1 --cpu-bind=cores python3 runTest.py --method dist_m4ri_RW --dataset bivariate_bicycle &
srun -n1 python3 runTest.py --method QDistRndMW --dataset lifted_product  &
srun -n1 python3 runTest.py --method QDistRndMW --dataset bivariate_bicycle  &
srun -n1 python3 runTest.py --method QDistRndMW --dataset hyperbolic_surface_code --datafile RG-3-8.txt &
srun -n1 python3 runTest.py --method QDistRndMW --dataset hyperbolic_colour_code --datafile RG-3-8.txt &
srun -n1 python3 runTest.py --method QDistRndMW --dataset codeTables --datafile QECC_k=1.txt &
srun -n1 python3 runTest.py --method QDistRndMW --dataset codeTables --datafile QECC_4k=n.txt &
srun -n1 python3 runTest.py --method QDistRndMW --dataset stim_circuit --datafile 01_surfacecodes &
srun -n1 python3 runTest.py --method QDistRndMW --dataset stim_circuit --datafile 02_surface_code_trans_cx_circuits &
srun -n1 python3 runTest.py --method QDistRndMW --dataset stim_circuit --datafile 03_colorcodes_midout &
srun -n1 python3 runTest.py --method QDistRndMW --dataset stim_circuit --datafile 04_colorcodes_superdense &
srun -n1 python3 runTest.py --method QDistRndMW --dataset stim_circuit --datafile 05_bivariatebicyclecodes &
srun -n1 python3 runTest.py --method UndetectableErrorStim --dataset lifted_product  &
srun -n1 python3 runTest.py --method UndetectableErrorStim --dataset bivariate_bicycle  &
srun -n1 python3 runTest.py --method UndetectableErrorStim --dataset hyperbolic_surface_code --datafile RG-3-8.txt &
srun -n1 python3 runTest.py --method UndetectableErrorStim --dataset hyperbolic_colour_code --datafile RG-3-8.txt &
srun -n1 python3 runTest.py --method UndetectableErrorStim --dataset codeTables --datafile QECC_k=1.txt &
srun -n1 python3 runTest.py --method UndetectableErrorStim --dataset codeTables --datafile QECC_4k=n.txt &
srun -n1 python3 runTest.py --method UndetectableErrorStim --dataset stim_circuit --datafile 01_surfacecodes &
srun -n1 python3 runTest.py --method UndetectableErrorStim --dataset stim_circuit --datafile 02_surface_code_trans_cx_circuits &
srun -n1 python3 runTest.py --method UndetectableErrorStim --dataset stim_circuit --datafile 03_colorcodes_midout &
srun -n1 python3 runTest.py --method UndetectableErrorStim --dataset stim_circuit --datafile 04_colorcodes_superdense &
srun -n1 python3 runTest.py --method UndetectableErrorStim --dataset stim_circuit --datafile 05_bivariatebicyclecodes &
srun -n1 python3 runTest.py --method decoderDist --dataset lifted_product  &
srun -n1 python3 runTest.py --method decoderDist --dataset bivariate_bicycle  &
srun -n1 python3 runTest.py --method decoderDist --dataset hyperbolic_surface_code --datafile RG-3-8.txt &
srun -n1 python3 runTest.py --method decoderDist --dataset hyperbolic_colour_code --datafile RG-3-8.txt &
srun -n1 python3 runTest.py --method decoderDist --dataset codeTables --datafile QECC_k=1.txt &
srun -n1 python3 runTest.py --method decoderDist --dataset codeTables --datafile QECC_4k=n.txt &
srun -n1 python3 runTest.py --method decoderDist --dataset stim_circuit --datafile 01_surfacecodes &
srun -n1 python3 runTest.py --method decoderDist --dataset stim_circuit --datafile 02_surface_code_trans_cx_circuits &
srun -n1 python3 runTest.py --method decoderDist --dataset stim_circuit --datafile 03_colorcodes_midout &
srun -n1 python3 runTest.py --method decoderDist --dataset stim_circuit --datafile 04_colorcodes_superdense &
srun -n1 python3 runTest.py --method decoderDist --dataset stim_circuit --datafile 05_bivariatebicyclecodes &
srun -n1 python3 runTest.py --method GurobiDist --dataset lifted_product  &
srun -n1 python3 runTest.py --method GurobiDist --dataset bivariate_bicycle  &
srun -n1 python3 runTest.py --method GurobiDist --dataset hyperbolic_surface_code --datafile RG-3-8.txt &
srun -n1 python3 runTest.py --method GurobiDist --dataset hyperbolic_colour_code --datafile RG-3-8.txt &
srun -n1 python3 runTest.py --method GurobiDist --dataset codeTables --datafile QECC_k=1.txt &
srun -n1 python3 runTest.py --method GurobiDist --dataset codeTables --datafile QECC_4k=n.txt &
srun -n1 python3 runTest.py --method GurobiDist --dataset stim_circuit --datafile 01_surfacecodes &
srun -n1 python3 runTest.py --method GurobiDist --dataset stim_circuit --datafile 02_surface_code_trans_cx_circuits &
srun -n1 python3 runTest.py --method GurobiDist --dataset stim_circuit --datafile 03_colorcodes_midout &
srun -n1 python3 runTest.py --method GurobiDist --dataset stim_circuit --datafile 04_colorcodes_superdense &
srun -n1 python3 runTest.py --method GurobiDist --dataset stim_circuit --datafile 05_bivariatebicyclecodes &
srun -n1 python3 runTest.py --method CLISATDist --dataset lifted_product  &
srun -n1 python3 runTest.py --method CLISATDist --dataset bivariate_bicycle  &
srun -n1 python3 runTest.py --method CLISATDist --dataset hyperbolic_surface_code --datafile RG-3-8.txt &
srun -n1 python3 runTest.py --method CLISATDist --dataset hyperbolic_colour_code --datafile RG-3-8.txt &
srun -n1 python3 runTest.py --method CLISATDist --dataset codeTables --datafile QECC_k=1.txt &
srun -n1 python3 runTest.py --method CLISATDist --dataset codeTables --datafile QECC_4k=n.txt &
srun -n1 python3 runTest.py --method CLISATDist --dataset stim_circuit --datafile 01_surfacecodes &
srun -n1 python3 runTest.py --method CLISATDist --dataset stim_circuit --datafile 02_surface_code_trans_cx_circuits &
srun -n1 python3 runTest.py --method CLISATDist --dataset stim_circuit --datafile 03_colorcodes_midout &
srun -n1 python3 runTest.py --method CLISATDist --dataset stim_circuit --datafile 04_colorcodes_superdense &
srun -n1 python3 runTest.py --method CLISATDist --dataset stim_circuit --datafile 05_bivariatebicyclecodes &
wait