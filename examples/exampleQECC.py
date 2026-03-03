import numpy as np
from codedistance import *

## [[17,1,7]]
S = '''1000000000000000111011101010111011
0100000000000000100110011111100110
0010000000000000111000100101001000
0001000000000000110111111000011111
0000100000000000100000010110110100
0000010000000000111011100001100001
0000001000000000100110011010001011
0000000100000000101000100111111110
0000000010000000111111111001000100
0000000001000000110100010110011001
0000000000100000100001100001110111
0000000000010000101011011010000000
0000000000001000111110000111111011
0000000000000100100100101001000110
0000000000000010111001111110011000
0000000000000001110111010101110111'''

## 5,1,3
# S = '''1001110101
# 0100110011
# 0011010011
# 0000001111'''

S = bin2ZMat(S)
tB = 2

# n,k,T = Stab2Tableau(S)
# print(np.sum(matMul(XZhad(S),S.T,2)))


params = {
    'iterCount': 100,     ## number of iterations - eg QDistRnd, decoderDist
    'maxTime': 3600*8,      ## max runtime for CLI methods (eg SAT, m4ri, )
    'nThreads': 1,          ## cater for multi-threaded methods
}

## Magma distance finding: requires Magma licence and magma executable in PATH
# method = 'magmaMinWeight'
# method = 'magmaMinWord'
# method = 'magmaWEDist'

## Serban Cercelescu C Library https://github.com/qubitserfed/Qubitserf
## requires compiled  qubitserf executable in PATH
## multi-threaded option available
## Brouwer-Zimmermann
# method = 'qubitSerfBZ'
## Meet in the middle
# method = 'qubitSerfMM'

## Stim methods - truncated Tanner graph traversal
## requires stim pypi package
# method='UndetectableErrorStim'
# method = 'GraphLikeErrorStim'
# method='ColourCodeDistStim'

## MW implementations of Stim methods
# method='UndetectableErrorMW'
# method='GraphLikeErrorMW'
# method='ColourCodeDistMW'

## MW implementation of QDistRnd
# method = 'QDistRndMW'

## Pryadko dist m4ri C package - requires dist_m4ri binary in $PATH
# method = 'dist_m4ri_RW'
# method = 'dist_m4ri_CC'

## SAT solvers:
## requires pysat python library
# method = 'pySATDist'
## Command line interface - requires a SAT solver binary to be in the $PATH - default is cashwmaxsatcoreplus
# method = 'CLISATDist'

## Gurobi MIP solver - requires gurobipy python package
# method = 'GurobiDist'

## OR-tools MIP solver - requires ortools python package
# method = 'MIPDist' 
## various MIP solvers can be used by OR-tools
params['MIP_solver'] = 'SCIP' 
##     CBC_MIXED_INTEGER_PROGRAMMING or CBC
##     SAT_INTEGER_PROGRAMMING or SAT or CP_SAT
##     SCIP_MIXED_INTEGER_PROGRAMMING or SCIP
##     GUROBI_MIXED_INTEGER_PROGRAMMING or GUROBI or GUROBI_MIP - requires full installation of Gurobi - https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer
##     CPLEX_MIXED_INTEGER_PROGRAMMING or CPLEX or CPLEX_MIP - requires installation of CPLEX https://www.ibm.com/docs/en/icos
##     XPRESS_MIXED_INTEGER_PROGRAMMING or XPRESS or XPRESS_MIP - requires installation of Xpress https://www.fico.com/fico-xpress-optimization/docs/latest/getting_started/dhtml/chap16.html
##     GLPK_MIXED_INTEGER_PROGRAMMING or GLPK or GLPK_MIP - requires installation of GLPK: https://www.gnu.org/software/glpk/''',type=str, default='SCIP')

## distance-finding via syndrom decoding
method = 'decoderDist'
## Type of decoder - for decoderDist only. Options are bposd, bplsd, sebplsd, tesseract, tesseractLongbeam",type=str, default='bposd')
params['decoder'] = 'bposd' 
## method=='decoderDist' and decoder in {'bposd','bplsd','sebplsd'} only
params["bp_method"] = 'product_sum' ## Belief Propagation method
params["bp_schedule"] = 'parallel' ## Belief Propagation schedule type - options parallel or serial
params["bp_max_iter"] = 100  ## Belief Propagation max iterations
params["osd_method"] = 'OSD_CS' 
params["osd_order"] = 1 


allMethods = ['magmaMinWeight',
              'magmaMinWord',
              'magmaWEDist',
              'qubitSerfBZ',
              'qubitSerfMM',
              'dist_m4ri_RW',
              'dist_m4ri_CC',
              'QDistRndMW',
              'UndetectableErrorStim',
              'GraphLikeErrorStim',
              'ColourCodeDistStim',
              'UndetectableErrorMW',
              'GraphLikeErrorMW',
              'ColourCodeDistMW',
              'decoderDist',
              'GurobiDist',
              'MIPDist',
              'pySATDist']
methods = [method]
methods = allMethods

startTimer()
for method in methods:
    res = codeDistance(S,L=None,tB=tB,method=method,params=params,seed=0)
    print(method)
    print(f'Calculated distance: {res['d']}')
    print(f'Time Taken: {elapsedTime()}')
    if (res['T'] > 1):
        print(f'Total Trials: {res['T']}')
        print(f'Trials at d={res['d']}: {res['R']}')
    lo = res['L']
    if (np.sum(lo)) > 0:
        r1,V = HowRes(S,lo,2)
        print(f'lo: {bin2Set(lo)}')
        lo = ZMat2D(lo)
        if tB == 2:
            lo = XZhad(lo)   
        r2 = matMul(lo,S.T,2)
        print(f'Logical Operator Check: commutes with stabilisers {np.sum(r2)==0}; non-trivial {np.sum(r1)!=0}')
    print("###########################################\n")