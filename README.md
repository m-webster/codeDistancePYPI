# codeDistance
Distance algorithms for classical codes, quantum codes and quantum circuits

# Authors
Mark Webster, Abe Jacob and Oscar Higgott

# Installation
To install the python library, use 

    pip install codedistance

If you wish to make changes to the source code or run the examples from the package, you can also download the codebase from the [github page](https://github.com/m-webster/codeDistancePYPI), create a virtual environment, navigate to the local copy of the codebase and run:

    pip install -e .

Certain distance-finding algorithms require installation of external packages:

1. dist_m4_RW and dist_m4_CC: require compilation of the C library from https://github.com/QEC-pages/dist-m4ri/ and 
2. magmaMinWeight, magmaMinWord and magmaWEDist: requires a Magma licence https://magma.maths.usyd.edu.au/magma/ and magma binary on system path
3. DistRndGAP: requires GAP https://www.gap-system.org/ with Guava https://gap-packages.github.io/guava/ and QDistRnd package https://qec-pages.github.io/QDistRnd/ and gap binary on system path
4. CLISATDist: requies a SAT solver binary on the system path - by default this is cashwmaxsatcoreplus which runs only on Unix systems and is available from https://maxsat-evaluations.github.io/2023
5. qubitserfBZ and qubitserfMM: requires compiliation of the C library from https://github.com/qubitserfed/Qubitserf. By default, this results in an executable called "interface". This binary should be renamed qubitserf and be accessible on the system path

To compile on a Windows machine we recommend using Cygwin with the packages listed  [here](https://www.codefull.net/2015/12/essential-cygwin-development-packages/).


# Main Distance-Finding Functions
The package finds distances of classical codes, CSS quantum codes, non-CSS quantum codes and quantum circuits. 
There are three main functions for these:
1. codeDistance(H,L,tB,method,params,seed): this is the main worker function
2. CSScodeDistance(Hx,Hz,method,params,component,seed)
3. circuitDistance(qc,method,seed)

The main inputs to these functions are:
- H,L,Hx,Hz,qc: A representation of the code or circuit
- method: the name of the distance-finding method to use
- params: a dictionary of optional settings for the distance-finding method
- seed: optional seed for random number generation 

# Input Formats for Codes and Circuits
The form of the input code/circuit depends on the function called. 

For codeDistance, H is a binary check matrix and L is binary matrix representing a logical basis - these are usually best provided as numpy int8 arrays.
To find distance for non-CSS codes, H should be given in symplectic (two block) form [HX|HZ].
To indicate that H is in two block form, set tB=2. The tB notation is used throughout the code base to indicate the number of blocks in the matrix.
L can optionally be excluded or set to None. This would be the case for classical codes or where we have not yet calculated a logical basis. For instance, where H is a two-block matrix representing a non-CSS code, a logical basis is calculated if required by codeDistance.

For CSScodeDistance binary matrices Hx and Hz are used to represent the X and Z checks respectively. These do not need to be full rank, and can be an over-complete set.
The component parameter can be set to 'Z' to find Z-distance (the default setting) or 'X' for X-distance.

For circuitDistance, the circuit qc should be provided in Stim format.

# Output Format
The three main functions return a python dictionary with the following fields:
- n: length of the code/circuit
- k: number of logicals/observables
- d: minimum distance estimate
- L: example of a logical or codeword with weight d in binary form
- T: the number of trials conducted (eg for QDistRnd or syndrome decoder methods)
- R: the number of trials which gave the distance estimate d
- progress: status information output by the method during processing

# Distance-Finding Methods
The distance-finding methods available in the package are as follows:

## Random Information Set Algorithms
- dist_m4ri_RW: python wrapper for C implementation of random window algorithm  https://github.com/QEC-pages/dist-m4ri/
- DistRndGAP: wrapper to GAP QDistRnd method https://qec-pages.github.io/QDistRnd/ 
- QDistRndMW: python implementation of QDistRnd
- QDistEvol: similar to QDistRnd but permutations are selected via an evolutionary algorithm rather than randomly

## Connected Component Algorithms
- dist_m4ri_CC: python wrapper for C implementation of connected cluster algorithm  https://github.com/QEC-pages/dist-m4ri/ 
- UndetectableErrorStim: distance finding via the Stim search_for_undetectable_logical_errors circuit function. By default, this method uses increasing values of dont_explore_detection_event_sets_with_size_above until the same minimum distance is found twice
- GraphLikeErrorStim: wrapper to shortest_graphlike_error - similar to setting dont_explore_detection_event_sets_with_size_above=2 and dont_explore_edges_with_degree_above=2 then calling search_for_undetectable_logical_errors
- ColourCodeDistStim: method optimised for colour code distance-finding. Wrapper to search_for_undetectable_logical_errors with dont_explore_detection_event_sets_with_size_above=3 and dont_explore_edges_with_degree_above=3. 
- connectedClusterMW: python version of connected cluster algorithm
- UndetectableErrorMW: python version of UndetectableErrorStim which calls connectedClusterMW
- GraphLikeErrorMW: python version of GraphLikeErrorStim which calls connectedClusterMW
- ColourCodeDistMW: python version of ColourCodeDistStim which calls connectedClusterMW

## Syndrome Decoder Algorithm
- decoderDist: distance-finding via syndrome decoder. By default, uses the BP-OSD from the ldpc package https://pypi.org/project/ldpc/. By changing parameter settings, can also use BP-LSD or tesseract.

## Solver-Based Methods
- GurobiDist: mixed integer programming using Gurobi - requires licence details which can be provided in the gurobiDistModel function in distance.py: https://support.gurobi.com/hc/en-us
- MIPDist: mixed integer programming using OR-tools library. Various solver methods are supported, but the default method is SCIP for scipy.
- CLISATDist: SAT solver method using a command line interface binary (eg cashwmaxsatcoreplus)
- pySATDist: SAT solver method using the pySAT library: https://pysathq.github.io/

## Brouwer-Zimmermann/Codeword Enumeration Algorithms
- magmaMinWeight: Magma Brouwer-Zimmermann algorithm. Only returns minimum distance, not a minimum-weight codeword. Requires a Magma licence https://magma.maths.usyd.edu.au/magma/ 
- magmaMinWord: as for magmaMinWeight but also returns a minimum-weight codeword and slower processing speed.
- magmaWEDist: distance via Magma's weight enumeration function
- BZDistMW: Python implementation of Brouwer-Zimmermann natively supporting CSS codes and multiple block representations of non-CSS quantum codes.
- qubitserfBZ: wrapper to C library implementation of https://github.com/qubitserfed/Qubitserf


## Matching Bipartition/Meet in the Middle
- qubitserfMM: wrapper to C library implementation of https://github.com/qubitserfed/Qubitserf
- MeetMiddleMW: python implementation of algorithm set out in https://github.com/qiskit-community/qiskit-qec/blob/main/docs/tutorials/QEC_Framework_IEEE_2022.ipynb


# Running Datasets from the Paper
The datasets used in the paper are saved in the examples subfolder https://github.com/m-webster/codeDistancePYPI/tree/main/examples. 
As each dataset may have many codes/circuits, the recommended way to run these is via the batch script runTest.py (also within the examples subfolder).
The distance-finding results for the data set are saved in a text file within the examples/results folder.
For example, to find distances of the non-CSS codetables dataset using the QDistEvol function, open a terminal window, navigate/change directory to the examples folder then run the following command via a terminal:

    python3 runTest.py --dataset codeTables --datafile QECC_sample.txt --method QDistEvol

# Parameters for Distance-Finding Methods
Here we lay out the main parameter settings used for various distance-finding methods:

## General Parameters
- method: Distance finding method to use (see above). Options are DistRndGAP, QDistRndMW, dist_m4ri_RW, dist_m4ri_CC, GurobiDist, MIPDist, CLISATDist, pySATDist, decoderDist, UndetectableErrorStim, GraphLikeErrorStim, ColourCodeDistStim, connectedClusterMW, UndetectableErrorMW, GraphLikeErrorMW, ColourCodeDistMW, magmaMinWeight, magmaMinWord, magmaWEDist, qubitserfBZ, qubitserfMM, BZDistMW, MeetMiddleMW. Default method is dist_m4ri_RW.
- iterCount: Number of Iterations to run for methods which require a number of trials to improve accuracy. Default is 10000.
- maxTime: Maximum run time for command-line interface methods. Thereafter, the process is terminated and any partial results are returned. Default is 8 hours = 3600*8.
- nThreads: Number of threads to run where parallel processing available - not fully implemented for all methods. Default is 1.
- verbose: Display verbose debugging info where supported by the method.
- maxErr: Future - not currently implemented. Track codewords and terminate when pErr < maxErr - for probabilistic methods only. Default is -1.
- GF4blockRep: Number of blocks to use when representing non-CSS quantum codes - 2, 3 or 4. Default is 2 for symplectic 2-block representation.  
- LOCheck: Set to 1 to check that the logical/codeword returned satisfies the checks and, for quantum codes/circuits, flips at least one logical.

## Batch Processing Commands via runTest.py
- runMode: Mode to use when running batch script. Possible values: slurm - generate slurm script; parallel - run in parallel using concurrent.futures; serial - run in serial. Default setting is serial.
- maxCount: Max number of codes/circuits to run from each dataset/datafile combination - use --maxCount 1 for debugging for instance. Default setting is 10000.
- repeat: Number of times to repeat for each code/circuit. Useful for doing sensitivity analysis. Default setting is 1. 
- dataset: Data set to run. Options are codeTables, lifted_product, bivariate_bicycle, all.
- datafile: File containing examples - used for codeTables, stim_circuit, hyperbolic codes.
- ignoreTrivialCodes: Set to 1 to ignore quantum codes with trivial codespaces. Default is 1.
- resume: Set to 1 to resume distance finding from previous run for the data set saved in examples/results. Will skip over codes where distance has already been calculated. Default is 0.
- nMin: Only run codes with n >= nMin. Default is 1.
- nMax: Only run codes with n <= nMax. Default is 100000.

## Quantum Circuits 
- filterCircuit: Circuit distance only: filter detector error models to reduce number of checks and error mechanisms to increase processing time. Defauls is 1. 

## QDistEvol/QDistRndMW only
- regroupPerm: For random information set algorithms on non-CSS codes, reorder permutations so that columns corresponding to the same qubit are next to each other. Default is 1.
- genCount: QDistEvol only: population size. Default is 100. 
- offspring: QDistEvol only: offspring per parent. Default is 10.
- pMut: QDistEvol only: average number of transpositions per mutation. Default is 2.0, but is ignored if pMutScale is > 0.
- pMutScale: QDistEvol only: scale pMut (number of transpositions) by code length - set to 0 for to use a fixed pMut value. Default is 50.
- sMut: QDistEvol only: mutation variance as proportion of pMut. Default is 0.2.
- swapPivot: QDistEvol only: set to 1 to choose only transpositions which swap between pivots and non-pivots. Default is 1.
- swapBlockorder: QDistEvol and non-CSS quantum codes only: set to 1 to swap blocks for same qubit. For instance in a length n code in 3-block representation columns 1, n+1 and 2n+1 all correspond to the same qubit. Setting swapBlockorder=1 means that transpositions can also swap colums 1, n+1 and 2n+1 prior to RREF calculation
- HL: QDistEvol/QDistRndMW only: set to 1 to use stacked HL rather than Pryadko kernel method. Default is 0.   
- tabuLength: QDistEvol only: length of tabu list. By default, set to 0 which indicates that the tabu list is not used (which speeds up processing times). Set to 1 to have a tabu list of length n (code length). Otherwise, the length of the tabu list will be a fixed tabuLength.

## Connected Cluster Algorithms
- CCstart: Set CCstart to a value >= 0 to restrict distance-finding to logicals with support on qubit with that index. For dist_m4ri_CC, connectedClusterMW, UndetectableErrorMW, GraphLikeErrorMW, ColourCodeDistMW only. Default is -1 

## MIPDist
- MIP_solver: Type of MIP solver. Default is SCIP. Ooptions are:
    - CBC
    - CP_SAT
    - SCIP
    - GUROBI - requires full installation of Gurobi - https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer
    - CPLEX - requires installation of CPLEX https://www.ibm.com/docs/en/icos
    - XPRESS - requires installation of Xpress https://www.fico.com/fico-xpress-optimization/docs/latest/getting_started/dhtml/chap16.html
    - GLPK - requires installation of GLPK: https://www.gnu.org/software/glpk/

## pySATDist
- pySATsolver: SAT solver to use - options are cd, cd15, cd19, cms, gc3, gc4, g3, g4, g42, lgl, mcb, mcm, mpl, mg3, mc, m22, mgh. Default is g4.
- pySATbinary: pySAT binary  to use - options are rc2, fm, lsu. Default is lsu.

## decoderDist
- decoder: Type of decoder. Options are bposd, bplsd, tesseract, tesseractLongbeam. Default is bposd.
- addStabs: Set to 1 to add random linear combination of stabilisers to randomly generated error. Default is 0. 
- permuteDEM: Set to 1 to apply random permutation to columns of DEM. Default is 1.
- bp_method: For bposd or bplsd only - Belief Propagation method. Default is product_sum.
- bp_schedule: For bposd or bplsd only - Belief Propagation schedule type - options parallel or serial. Default is parallel.
- bp_max_iter: For bposd or bplsd only - Belief Propagation max iterations. Default is 100.
- osd_method: For bposd only - OSD method. Default is OSD_CS.
- osd_order:F or bposd only - OSD order. Default is 1