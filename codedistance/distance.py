import os
import re
import subprocess
import datetime
import concurrent.futures
import numba as nb
import stim
import ldpc
from .dem_detector_filtering import filter_by_det_basis_using_chromobius_coords
import gurobipy as gp
from gurobipy import GRB
import ortools
import ortools.linear_solver
import ortools.linear_solver.pywraplp
import scipy
import tesseract_decoder as ts

## MW Library Files
from .common import *
from .NHow import *
from .code_library import *

#########################################################
## Main Distance-Finding Functions
#########################################################

def codeDistance(H,L=None,tB=1,method='QDistRndMW',params={},seed=None):
    '''Main function for code distance finding:
    non-CSS codes  (eg codetables.de): input check matrix S in symplectic form, set tB=2
    CSS codes: input H=HX and L=LX for Z-distance or H=HZ and L=LZ for X-distance, set tB=1
    Classical binary linear codes: set S to be the generator matrix, tB=1 '''
    H = Z2Mat(H)
    if L is not None:
        L = Z2Mat(L)
    r,n = H.shape
    res = {'n':n//tB,'k':0,'d': 0, 'L': Z2MatZeros(n), 'progress': '', 'R': 1,'T': 1}
    wDict = None
    if method in {'magmaMinWeight', 'magmaMinWord','magmaWEDist'}:
        res['d'],res['L'],res['progress'] = magmaDist(H,L,tB,params)
    elif method == 'DistRndGAP':
        res['d'],res['L'],res['progress'] = DistRndGAP(H,L,tB,params) 
    elif method[:9] == 'qubitserf':
        res['d'],res['progress'] = dist_qubitserf(H,L,tB,params) 
    elif method == 'dist_m4ri_RW':
        res['d'],res['L'],res['progress'] = dist_m4ri_RW(H,L,tB,params)
    elif method == 'dist_m4ri_CC':
        res['d'],res['L'],res['progress'] = dist_m4ri_CC(H,L,tB,params)
    elif method in ('QDistRndMW','QDistRndHL','QDistEvol'):
        res['d'],res['L'],wDict,res['progress'] = QDistEvol(H,L,tB,params,seed=seed)
    elif method == 'BZDistMW':
        res['d'],res['L'],res['progress'] = BZDistMW(H,L,tB,params)
    elif method == 'GurobiDist':
        res['d'],res['L'],res['progress'] = gurobiDist(H,L,tB=tB,params=params)
    elif method =='MeetMiddleMW':
        res['d']= MeetMiddleMW(H,L,tB,params)
    else:
        ## methods in this section operate on DEM with tB=1
        if L is None:
            L = defaultLogicals(H,tB)
        ## convert stabs to DEM and simplify
        DEM, ix, params['priors'], n3 = HL2DEM(H,L,tB, True)
        DEMH,DEML = DEM[:r],DEM[r:]
        ## call the relevant method
        if method == 'pySATDist':
            res['d'],res['L'],res['progress']= pySATdist(DEMH,DEML,params)
        elif  method == 'CLISATDist':
            res['d'],res['L'],res['progress']= CLISATDist(DEMH,DEML,params)
        elif method == 'decoderDist':
            res['d'],res['L'],wDict,res['progress'] = decoderDist(DEMH,DEML,params)
        elif method == 'MIPDist':
            res['d'],res['L'],res['progress']= MIPDist(DEMH,DEML,params=params)
        elif method in {'UndetectableErrorStim','GraphLikeErrorStim','ColourCodeDistStim'}:
            res['d'],res['L'],wDict,res['progress'] = StimCodeDist(DEMH,DEML,params) 
        elif method in {'connectedClusterMW','UndetectableErrorMW','ColourCodeDistMW','GraphLikeErrorMW'}:
            res['d'],res['L'],wDict,res['progress'] = connectedClusterMW(DEMH,DEML,method,params)
        else:
            print(f'could not find distance finding method {method}')
            return res
        ## convert error back to original basis
        res['L'] = DEM2L(res['L'],ix,n3,tB)
    if wDict is not None and res['d'] < len(wDict):
        res['R'] = wDict[res['d']]
        res['T'] = np.sum(wDict)
    if params['LOCheck']:
        LOcheck(res['L'],H,L,tB)
    return res

def LOcheck(lo,H,L=None,tB=1):
    '''check if string lo is a logical operator - commutes with all rows of H and anti-commutes with at least one row of L'''
    if np.sum(lo) > 0:
        lo2D = ZMat2D(lo)
        w = weightTB2(lo2D)[0] if tB==2 else np.sum(lo)
        print(f'{func_name()} w:{w} lo: {" ".join(map(str,bin2Set(lo)))}')
        if tB == 2:
            ## Quantum codes - check if lo is in stabiliser group
            R, U = HowRes(H,lo2D,2)
            print("Non-stabiliser",np.sum(R) > 0)
            lo2D = XZhad(lo2D)
        M = matMulZ2(lo2D,Z2Mat(H.T) )
        print("Satisfies Checks",0 == np.sum(M))
        if L is not None and np.sum(L) > 0:
            ## CSS Codes or stabiliser codes where L is supplied
            M = matMulZ2(lo2D,Z2Mat(L.T) )
            print("Anticommutes with a logical", np.sum(M) > 0)
    
def circuitDistance(qc,method,params={},seed=None):
    '''Main function for circuit distance
    input qc is a Stim quantum circuit object'''
    paramDefaults = {
        'isDEM':1,
        'filterCircuit': True, ## filter DEM to only include errors of a particular Pauli type
    }
    params = setDefaultParams(params,paramDefaults)
    DEM = qc.detector_error_model().flattened()
    if params['filterCircuit']:
        ## Oscar Filter
        DEM = filter_by_det_basis_using_chromobius_coords(DEM)
        ## Mark Implementation
        # DEM = ChromobiusDEMFilter(DEM,'Z')
    H,L,params['priors'] = StimDEM2HL(DEM)
    print('DEM dimensions',H.shape)
    ## circuit methods act on single block matrices
    return codeDistance(H,L,tB=1,method=method,params=params,seed=seed)

def CSScodeDistance(Hx,Hz,method='QDistRndMW',params={},component='Z',seed=None):
    '''Calculate distance of CSS code with X/Z checks Hx/Hz'''
    SX,LX,SZ,LZ = CSSCode(SX=Hx,SZ=Hz)
    k,n = LX.shape
    if component=='Z': 
        res = codeDistance(SX,LX,tB=1,method=method,params=params,seed=seed)
    if component=='X':
        res = codeDistance(SZ,LZ,tB=1,method=method,params=params,seed=seed)
    res['k'],res['n'] = k,n
    return res

#########################################################
## Helper Functions
#########################################################

def dirCheck(myDir):
    if not os.path.exists(myDir):
        os.makedirs(myDir)

def randomFilename(n=8,seed=0):
    '''Generate unique random filename based on current time for temporary storage'''
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
    rnd = np.random.default_rng(seed)
    rand = rnd.integers(10 ** n)
    return f'{timestamp}_{str(rand).zfill(n)}'

def CLIRun(process,commands="",maxTime=3600 * 8,captureErrors=False,seed=None):
    '''Run process via CLI - kill after maxTime reached and return stdout'''
    ## save stdout to temporary file just in case of timeout
    fileName = randomFilename(seed=seed)
    myDir = f'{os.getcwd()}/tmp'
    dirCheck(myDir)
    stdOutFile = f'{myDir}/{fileName}.stdout'
    sOut = open(stdOutFile,'w')
    ## redirect errors to file - slight performance penalty
    if captureErrors:
        stdErrFile = f'{myDir}/{fileName}.stderr'
        sErr = open(stdErrFile,'w')
    else:
        sErr =  stderr=subprocess.PIPE
    proc = subprocess.Popen(process,stdin=subprocess.PIPE, stdout=sOut, stderr=sErr, text=True,bufsize=1,universal_newlines=True)
    try:
        if len(commands) > 0:
            proc.communicate(commands,timeout=maxTime)
        else:
            proc.communicate(timeout=maxTime)
        proc.wait()
    except subprocess.TimeoutExpired:
        proc.kill()
    except subprocess.CalledProcessError:
        proc.kill()
    finally:
        sOut.close()
        with open(stdOutFile,'r') as sOut:
            outs = sOut.read()
        os.remove(stdOutFile) 
        ## clean up error file
        if captureErrors:
            sErr.close()
            with open(stdErrFile,'r') as sErr:
                errs = sErr.read()
            os.remove(stdErrFile)
        else:
            errs = ""
    return outs,errs

def setDefaultParams(params,paramDefaults):
    '''update params dict with defaults for values where these are not set'''
    for k in paramDefaults.keys():
        if k not in params:
            params[k] = paramDefaults[k]
    return params

def pauli2str(S,tB=2):
    '''Convert stabilisers S into Pauli Strings
    For tB=1 - classical code, just Pauli X
    For tB=2 - quantum code in 2-block symplectic form, Pauli X, Y, Z'''
    m,n2 = S.shape
    if tB ==2:
        n = n2//2
        S = S[:,:n] * 2 + S[:,n:]
    PauliStrings = "IXZY"
    return "\n".join(["".join([PauliStrings[a] for a in myrow]) for myrow in S])

def defaultLogicals(S,tB):
    '''Return defaul logical operators for distance-finding
    For tB=1 - classical code - return basis of complementary space
    For tB=2 - logical Paulis for k>0 else destabilisers'''
    if tB==2:
        n,k,T = Stab2Tableau(S)
        r = n-k
        ## L are Logical X and Z if k>0  else the destabilisers
        L = ZMatVstack([T[r:n],T[n+r:]]) if k > 0 else T[n:n+r]
    else:
        H,pivots = getH(S,N=2,retPivots=True)
        r,n = H.shape
        ## L are vectors not in <S> which generate the whole vector space
        L = Z2MatZeros((n-r,n))
        for i,j in enumerate(invRange(n,pivots)):
            L[i,j] = 1
    return L

def CSSDual(SX,LX):
    '''Input SX,LX for a CSS code and convert to SZ,LZ'''
    ## convert to int8
    SX = Z2Mat(SX)
    LX = Z2Mat(LX)
    k, n = LX.shape
    ix = np.arange(n)
    ## calculate SZ
    SXLX = Z2Mat(ZMatVstack([SX,LX]))
    SZ = KerZ2(SXLX,ix,0)
    if k==0:
        return SZ, np.zeros((0,n),dtype=np.int8)
    LX,LZ = CSSLXLZ(SX,SZ)
    return SZ, LZ

def isCSS(S):
    '''Check if QECC stabilisers S are CSS or not'''
    m,n2 = S.shape
    n = n2//2
    wX = np.sum(S[:,:n], axis=-1)
    wZ = np.sum(S[:,n:], axis=-1)
    w = wZ[wX>0]
    return np.sum(w) == 0

def CSSSplit(S):
    '''Split 2-block rep into HX and HZ or False if not possible'''
    S = RemoveZeroRows(S)
    m,n2 = S.shape
    n = n2//2
    HX = RemoveZeroRows(S[:,:n])
    HZ = RemoveZeroRows(S[:,n:])
    if len(HX) + len(HZ) == m:
        return HX,HZ
    return False

def CSS2twoBlock(SX,SZ):
    '''Convert SX and SZ to two-bock check matrix form'''
    return ZMatVstack([ZMatHstack([SX,Z2MatZeros(SX.shape)]),ZMatHstack([Z2MatZeros(SZ.shape),SZ])])

def regroupPerm(ix,tB):
    '''take a permutation ix on n * tB columns - eg X, Z, X+Z blocks
    convert it so that the columns corresponding to the same qubit are next to each other'''
    if tB == 1:
        return ix
    n = len(ix)
    nB = n//tB
    ## qubits are equivalence classes of columns mod nB
    ixMod = np.mod(ix,nB)
    ## ordering of columns for single qubit class
    InnerOrder = [[] for i in range(nB)] 
    ## global ordering of qubits
    OuterOrder = []
    for i, c in enumerate(ixMod):
        if len(InnerOrder[c]) == 0:
            OuterOrder.append(c)
        InnerOrder[c].append(ix[i])
    temp = []
    for c in OuterOrder:
        temp += InnerOrder[c]
    temp = ZMat(temp)
    return temp

def NonZeroVecRand(k,seed=None):
    '''Generate random non-zero vector of length k'''
    if k == 1:
        return Z2Mat([1])
    else:
        rng = np.random.default_rng(seed)
        x = rng.integers(2,size=k-1)
        x = Z2Mat([1] + list(x))
        ## rotate vector by random i in [0..k)
        i = rng.integers(k)
        return np.roll(x,i)

def partitionTrials(t,g):
    '''parition t trials into g groups'''
    ## p: trials divided by number of groups
    p = t//g
    ## d: number of groups which need one extra trial
    d = t - p * g
    ## gCount: number of trials per group
    gCount = [p+1] * d + [p] * (g - d)
    return ZMat(gCount)

def setXOR(SetList,ix):
    '''Calculate XOR of list of sets SetList indexed by ix'''
    temp = None
    for e in ix:
        if temp is None:
            temp = SetList[e].copy()
        else:
            temp ^= SetList[e]
    return temp

def wDist2str(wDict):
    '''Printable version of weight dictionary for output'''
    temp = []
    for i, w in enumerate(wDict):
        if w > 0:
            temp.append(f'{i}:{w}')
    return ",".join(temp)

#########################################################
## Various block representations of Quantum Codes
#########################################################

def encode2block(S,addI=False,tB=3):
    '''Encode 2-Block Check or Logical Matrix S for Quantum code into 2, 3, 4 block representation'''
    if tB not in {2,3,4}:
        return S
    if tB == 2:
        return XZhad(S)
    elif tB==4:
        return Two2FourBlock(S,addI) 
    else:
        return Two2ThreeBlock(S,addI) 

def decodeBlock(S,tB=3):
    '''Decode 2, 3, 4 block representation back into 2-block representation'''
    if tB <= 2:
        return S
    ## check if vector or multi dim matrix
    v = len(S.shape) == 1
    S = ZMat2D(S)
    if tB ==3:
        H = Three2TwoBlock(S) 
    elif tB == 4:
        H = Four2TwoBlock(S) 
    else:
        H = S
    if v:
        return H[0]
    return H

def Two2FourBlock(S,addI=False):
    '''Map [[n,k,d]] non-CSS code using [[4n,2k,2d]] mapping of https://arxiv.org/abs/1004.3791'''
    r,n = S.shape
    nB = n//2
    Sx,Sz = S[:,:nB], S[:,nB:]
    Sy = Sx * Sz
    H = ZMatHstack([Sx ^ Sy, Sy, Sz ^ Sy, Sx | Sz])
    if addI:
        H = appendI(H,nB)
    C = matMul(H,H.T,2)
    return H

def Four2TwoBlock(S):
    '''Convert binary vector in three block X|Z|X+Z form to X|Z
    used to convert correction back to two block Pauli form'''
    r,n = S.shape
    nB = n//4
    ## need to enforce Sx + Sz + Sy = 0
    ## so flip Sx if Sy is set - if both Sx and Sy then Sx' = 0 and Sz' = 1
    Sx,Sy,Sz = [S[:,i*nB:i*nB+nB] for i in range(3)]
    H = ZMatHstack([Sx^Sy,Sz^Sy])
    return H

def Two2ThreeBlock(S,addI=False):
    '''Convert two-block stabiliser generators into DEM check matrix of form X|Z|X+Z
    the X block indicates which stabilisers are flipped by an X error on the qubit corresponding to the column'''
    v = len(S.shape) == 1
    S = ZMat2D(S)
    r,n = S.shape
    nB = n//2
    Sx,Sz = S[:,:nB], S[:,nB:]
    Sy = Sx ^ Sz
    H = ZMatHstack([Sz,Sx,Sy]) 
    if addI:
        H = appendI(H,nB)
    C = matMul(H,H.T,2)
    if v:
        return H[0]
    return H

def Three2TwoBlock(S):
    '''Convert binary vector in three block X|Z|X+Z form to X|Z
    used to convert correction back to two block Pauli form'''
    v = len(S.shape) == 1
    S = ZMat2D(S)
    r,n = S.shape
    nB = n//3
    ## need to enforce Sx + Sz + Sy = 0
    ## so flip Sx if (Sx + Sz + Sy == 1) i.e. Sx' = Sx + (Sx + Sz + Sy) = Sz + Sy
    Sx,Sz,Sy = [S[:,i*nB:i*nB+nB] for i in range(3)]
    H = ZMatHstack([Sx^Sy,Sz^Sy])
    if v:
        return H[0]
    return H

def appendI(H,nB):
    '''Append a block of I matrics of size nB x nB below S'''
    r,n = H.shape
    ## make all zero matrix of required dimensions
    HB = Z2MatZeros((r+nB,n))
    ## add H to top of HB
    HB[:r,:] = H
    for i in range(nB):
        ## adding I blocks
        for j in range(n//nB):
            HB[r+i,i+j*nB] = 1
    return HB

#############################################################################
## Weights of single and two-block matrices
#############################################################################

def rowWeight(H,tB=1):
    '''Return weights of each row of H
    tB=1: sum the rows for Classical/CSS codes
    tB=2: Pauli weights for non-CSS codes'''
    if tB == 2:
        return weightTB2(H)
    else:
        return np.sum(H,axis=-1)

def weightTB2(S):
    '''Pauli weights for non-CSS codes given in 2-block format'''
    n = len(S.T)//2
    SX = S[:,:n] # X-component
    SZ = S[:,n:] # Z-component
    return np.sum((SX * SZ) ^ SX ^ SZ,axis=-1) # adding weights of X and Z components double-counts Y operators

#########################################################
## DEM Functions
#########################################################

def HL2DEM(H,L,tB,simplify=True):
    '''Convert stabilisers and logicals to 
    DEM: corresponding DEM matrix
    ix: map from cols of DEM to cols of S
    tB: number of blocks - 2 for non-CSS quantum codes
    simplify: true to eliminate all zero and duplicate columns'''
    HL = ZMatVstack([H,L])
    if tB == 2:
        HL = Two2ThreeBlock(HL)
    n3 = len(HL.T)
    if simplify:
        DEM, ix, priors = simplifyDEM(HL)
    else:
        ix = np.arange(n3)
        priors = [0.001]*n3
    return DEM, ix, priors, n3

def ChromobiusDEMFilter(DEM,desiredBasis):
    '''MW Simplified DEM Filter for testing of algorithm statement'''
    DList = set()
    for (D,coord) in DEM.get_detector_coordinates().items():
        coordType = 'Unknown'
        if len(coord) == 4:
            if coord[3] in {0,1,2}:
                coordType = 'X'
            if coord[3] in {3,4,5}:
                coordType = 'Z'
        if coordType == desiredBasis:
            DList.add(D)
    DEMFiltered = stim.DetectorErrorModel()
    count = {'detector':0, 'error':0}
    for instruction in DEM:
        if instruction.type in count :
            myTargets = {t.val for t in instruction.targets_copy() if t.is_relative_detector_id()}
            if myTargets.issubset(DList):
                DEMFiltered.append(instruction)
                count[instruction.type] += 1
        else:
            DEMFiltered.append(instruction)
    return DEMFiltered

def simplifyDEM(DEM):
    '''eliminate all-zero and duplicate columns of DEM and return error numbers which correspond to each column (ix)'''
    ## append all zero col
    DEM = ZMatHstack([Z2MatZeros((len(DEM),1)),DEM])
    ## remove duplicate cols
    DEM,ix,counts = np.unique(DEM,axis=-1,return_index=True,return_counts=True)
    ## first col is all-zero - update ix nd DEM
    DEM = DEM[:,1:]
    ix = ix[1:] - 1
    counts = list(counts[1:] * 0.01)
    return DEM, ix, counts

def DEM2L(y,ix,n,tB):
    '''Convert vector y back to a vector of length n where ix is a col map, tB is the number of blocks'''
    x = Z2MatZeros(n)
    x[ix] = y
    if tB==2:
        x = Three2TwoBlock(x)
    return x

###################################################################
## Convert to and from Stim objects
###################################################################

def Stabs2StimCircuit(H,L=None,tB=2):
    '''Convert stabilisers and logicals to stim circuit - based on Oscar Higgott's code
    modified to handle 2-block (Paulis) or single-block (CSS or classical)'''
    r,n = H.shape
    p = 0.01
    error_model = 'Z_ERROR'
    if tB==2:
        n = n//2
        ## X/Y/Z errors
        error_model = 'DEPOLARIZE1'
    SPaulis = Stabs2StimPaulis(H,tB)
    LPaulis = Stabs2StimPaulis(L,tB)
    circuit = stim.Circuit()

    _append_observable_includes_for_paulis(
        circuit=circuit, paulis=LPaulis)
    circuit.append("MPP", SPaulis)
    circuit.append(error_model, targets=list(range(n)), arg=p)
    circuit.append("MPP", SPaulis)

    for i in range(r):
        circuit.append(
            "DETECTOR",
            targets=[
                stim.target_rec(i - 2 * r),
                stim.target_rec(i - r)
            ]
        )

    _append_observable_includes_for_paulis(
        circuit=circuit, paulis=LPaulis)
    return circuit

def Stabs2StimPaulis(H,tB=2):
    '''Convert stabilisers to Stim Pauli objects - based on Oscar Higgott's code
    modified to handle 2-block (Paulis) or single-block (CSS or classical)'''
    r,n = H.shape
    H = H.astype(bool)
    if tB==1:
        Hx = H
        Hz = np.zeros(H.shape,dtype=bool)
    else:
        Hx = H[:,:n//2]
        Hz = H[:,n//2:]
    return  [stim.PauliString.from_numpy( xs=Hx[i], zs=Hz[i]) for i in range(r)]

def pauli_to_observable_include_target(pauli):
    '''Oscar Higgott's code'''
    obs_pauli_targets = []
    for i in range(len(pauli)):
        if pauli[i] != 0:
            obs_pauli_targets.append(stim.target_pauli(i, pauli[i]))
    return obs_pauli_targets

def _append_observable_includes_for_paulis(circuit, paulis):
    '''Oscar Higgott's code'''
    for i, obs in enumerate(paulis):
        circuit.append(
            "OBSERVABLE_INCLUDE",
            targets=pauli_to_observable_include_target(pauli=obs),
            arg=i
        )

def Stabs2StimDEM(S,p=0.01):
    '''Convert stabilsers to Stim DEM'''
    r,n = S.shape
    ## Stim doesn't like it if the error does not flip any detectors, so we will exclude these - buggers up the ordering though...
    colSum = np.sum(S,axis=0)
    temp = [f"error({p}) " + ' '.join([f"D{j}" for j in bin2Set(S.T[i])]) for i in range(n)  if colSum[i] > 0] + [f"detector({i},0,0) D{i}" for i in range(r)]
    return stim.DetectorErrorModel("\n".join(temp))

def StimDEM2HL(dem):
    '''Convert Stim DEM to stabilisers and logicals'''
    eStr = 'error'
    myDict = dict()
    ## nErr is the number of error mechanisms
    nErr = 0
    priors = []
    for instruction in dem.flattened():
        arglist = str(instruction).split(' ')
        if (arglist[0][:len(eStr)] == eStr):
            p = float(arglist[0][len(eStr)+1:-1])
            priors.append(p)
            for dl in arglist[1:]:
                ## convert entries of form 'D0','L20' to ('D',0) and ('L',20) to preserve ordering
                dl = (dl[0],int(dl[1:]))
                ## update myDict
                if dl not in myDict:
                    myDict[dl] = []
                myDict[dl].append(nErr)
            nErr += 1
    ## convert dict to ZMat and count number of stabilisers r
    HL = []
    r = 0
    for (i,dl) in enumerate(sorted(myDict.keys())):
        HL.append(set2Bin(nErr,myDict[dl],dtype=np.int8))
        if(dl[0] == 'D'):
            r+=1
    ## split into stabilisers and logicals
    HL = Z2Mat(HL)
    H,L = (HL[:r]),(HL[r:])
    return H,L,np.array(priors)

######################################################
## Weight Enumerator - Exhaustive Distance Method
######################################################

def weightEnumGray(A,tB=1,nThreads=0):
    '''Calculate weight enumerator of code with generator matrix A using Gray Codes'''
    r,n = A.shape
    ## split into kStep steps of length kLen
    k = r //2
    kStep = 1 << k
    kLen = 1 << (r-k)
    ## convert A to np.int8
    A = Z2Mat(A)
    ## if nThreads < 2, just run a single thread
    if nThreads < 2:
        return WEGThread(A,tB,kStep,0,kLen)
    ## P is the weight enumerator 
    P = ZMatZeros(n+1)
    ## kList identifies the ranges we want to run in parallel
    kList = [i * kLen//nThreads for i in range(nThreads)] + [kLen]
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        threadFuture = {executor.submit(WEGThread,A,tB,kStep,kList[i],kList[i+1]): i for i in range(nThreads)}
        for future in concurrent.futures.as_completed(threadFuture):
            P += future.result()
    return P

def WEGThread(A,tB,kStep,k0,k1):
    '''Thread process for weightEnumGray'''
    r,n = A.shape
    P = ZMatZeros(n+1)
    r = len(A)
    KA = grayCodeGen(A,kStep,k0,k1)
    weightEnumUpdate(KA,P,tB)
    for x in range(1,kStep):
        KA ^= A[grayBit(x,r)]
        weightEnumUpdate(KA,P,tB)
    return P

def weightEnumUpdate(A,P,tB):
    '''Update weight enumerator '''
    res = np.unique_counts (rowWeight(A,tB))
    P[res.values] += res.counts

# @nb.jit (nb.int64(nb.int64,nb.int64))
def grayBit(x,r):
    '''bit which is flipped when moving from grayCode(x-1) to grayCode(x)'''
    for i in range(r):
        if x % 2 == 1:
            return i
        x = x >> 1
    return 0

@nb.jit (nb.int64(nb.int64))
def grayCode(x):
    '''calculte Gray Code corresponding to integer x'''
    return  x ^ (x >> 1)

@nb.jit (nb.int8[:](nb.int64,nb.int64))
def int2bin(x,n):
    '''convert integer to binary representation'''
    temp = np.zeros(n,dtype=nb.int8)
    for i in range(n):
        if x == 0:
            break
        temp[i] = x % 2
        x = x >> 1
    return temp

def grayCodeGen(A,kStep,k0,k1):
    '''generate linear combinations of A corresponding to Gray Codes [k0 * kStep,..,k1 * kStep] '''
    r,n = A.shape
    K = Z2Mat([int2bin(grayCode(i * kStep),n) for i in range(k0,k1)])
    return matMulZ2(K,A)

######################################################
## MAGMA Methods
######################################################

def magmaDist(H,L=None,tB=1,params={}):
    '''Default Magma Distance finding using Brouwer-Zimmermann'''
    paramDefaults = {
        'method':'magmaMinWeight',
        'verbose': False,
        'maxTime': 3600 * 8
    }
    params = setDefaultParams(params,paramDefaults)
    method = params['method']
    if L is None or len(L) == 0 or tB==2:
        ## For classical codes - use the generator matrix, not H
        if tB == 1:
            H = KerZ2(H,np.arange(len(H.T)),0) 
    else:
        ## convert CSS code to 2-block for distance finding
        H = CSSSXLX2S(H,L)
        tB = 2
    m,n = H.shape
    d = 0
    x = Z2MatZeros(n)
    if method == 'magmaWEDist':
        d, stdOut = magmaWEDist(H,tB)
        return d,x,stdOut
    commands = []
    ## increasing number of threads appears to introduce errors and not significantly improve speed
    # commands.append('SetNthreads(8);')
    commands.append('SetVerbose("Code", true) ;')
    commands.append(ZMat2MagmaMat(H,mat_name='S'))
    commands.append(f'C := LinearCode(S);')
    commands.append(f'S;')
    if tB==2:
        commands.append(f'C := QuantumCode(C : ExtendedFormat := true);')
    commands.append(f'C;')
    commands.append('ResetMinimumWeightBounds(C) ;')
    if method == 'magmaMinWord':
        commands.append('MinimumWord(C) ;')
    else:
        commands.append('MinimumWeight(C) ;')
    stdOut,stdErr = CLIRun(['magma'],"\n".join(commands),maxTime=params['maxTime'])
    stdOut = magmaRemoveCodeDef(stdOut)
    ## parse output to find distance and logical operator x
    ## for minWeight, 4th last line of output is distance, providing the program terminates
    if (method != 'magmaMinWord') and stdOut.find("Total memory usage:") > 0:
        d = int(stdOut.split("\n")[-4])
    ## for minWord, we can extract a min weight LO, providing the program terminates
    if method == 'magmaMinWord':
        lo = magmaParseLO(stdOut)
        if len(lo) > 0:
            ## first and last characters of lo are "(", ")". Strip these and replace magma GF4 reps with I,X,Y,Z
            lo = lo[1:-1].replace(" ","").replace("$.1^2","Y").replace("$.1","Z").replace("1","X").replace("0","I")
            n2 = len(lo)
            d = n2
            for i, c in enumerate(lo):
                if c == 'X':
                    x[i] = 1
                elif c == 'Y':
                    x[i] = 1
                    x[i + n2] = 1
                elif c == 'Z':
                    x[i + n2] = 1
                else:
                    d -= 1
    ## if d is not yet set, parse the output to find upper and lower bounds
    if d == 0:
        lower, d = magmaParseLU(stdOut)
    return d,x, stdOut

def magmaRemoveCodeDef(myText):
    '''Remove the code definition from Magma output to reduce space'''
    pattern = "(\\[| +)(\\$\\.1\\^2|0|1|\\$\\.1)( +(\\$\\.1\\^2|0|1|\\$\\.1))*(\\]| +)?\\n"
    pattern = "\\n\\[(\\$\\.1\\^2|0|1|\\$\\.1| |\\n)+\\]"
    return re.sub(pattern, "", myText)

def ZMat2MagmaMat(A,mat_name='M'):
    '''convert A to Magma matrix string'''
    m, n = A.shape
    mat_str = [','.join(map(str, row)) for row in A]
    mat_str = ',\n'.join(mat_str)
    magma_code = f"{mat_name} := Matrix(GF(2), {m}, {n},\n" + "[" + mat_str + "]);\n"
    return magma_code 

def magmaParseLU(mytext):
    '''Parse magma distance-finding output - find upper and lower bounds'''
    lower,upper = 0,0
    cm = re.compile(r".*[lL]ower\s*[=:]\s+(\d+)[\.,].*[uU]pper\s*[=:]\s+(\d+)")
    cm2 = re.compile(r".*New codeword identified of weight (\d+),.*")
    for myline in mytext.split("\n"):
        myline = myline.strip()
        reMatch = cm.match(myline)
        if reMatch:
            lower, upper = reMatch.group(1,2)
        reMatch = cm2.match(myline)
        if reMatch:
            upper = reMatch.group(1)
    return int(lower),int(upper)

def magmaParseLO(mytext):
    '''Parse magma distance-finding output - find minWord'''
    loList = []
    endStr = "Final Results:"
    inLO = False
    atEnd = False
    for myline in mytext.split("\n"):
        myline = myline.strip()
        if atEnd:
            if len(myline) > 0 and myline[0] == "(":
                inLO = True
                lo = ""
            if inLO:
                lo += myline
                if myline[-1] == ")":
                    loList.append(lo)
                    inLO = False
                    atEnd = False
        else:
            atEnd = (myline.find(endStr) >= 0)
    if len(loList) > 0:
        return loList[-1]
    return ""

def magmaWEDist(S,tB=1,verbose=False):
    '''distance via Magma weight enumerator'''
    WE, stdOut = magmaWeightEnumerator(S,tB,verbose)
    for i in range(len(WE)):
        if WE[i] > 0:
            return i, stdOut

def magmaWeightEnumerator(S,tB=1,verbose=False):
    '''Calculate weight enumerator using MAGMA'''
    commands = []
    commands.append('SetVerbose("Code", 1) ;')
    commands.append(ZMat2MagmaMat(S,mat_name='S'))
    commands.append(f'C := LinearCode(S);')
    commands.append(f'S;')
    r,n = S.shape
    if tB==2:
        commands.append(f'C := QuantumCode(C : ExtendedFormat := true);')
        n = n//2
    commands.append(f'C;')
    commands.append('ResetMinimumWeightBounds(C) ;')
    commands.append('WeightEnumerator(C) ;')
    stdOut,stdErr = CLIRun(['magma'],"\n".join(commands))
    if verbose:
        print(stdOut)
    temp = ZMatZeros(n+1)
    progress = stdOut
    stdOut = stdOut.split("\n")
    mt = ""
    for myrow in stdOut:
        if (myrow.find("*") > 0):
            mt += myrow
    WE = mt.split(" + ")
    for w in WE:
        w = w.strip()
        w = w.split("*")
        if len(w) == 1:
            t = 1
        else:
            t = int(w[0])
            w = w[1:]
        b = 0
        for a in w:
            a = a.split("^")
            if (a[0]) == "$.2":
                b = int(a[1]) if len(a) == 2 else 1
        temp[b] = t
    return temp, progress

#########################################################
## MW Implementation of Pryadko's qDistRnd 
#########################################################

@nb.njit(nb.int64[:](nb.int8[:,:]))
def weightTB2nb(S):
    '''Pauli weights for non-CSS codes given in 2-block format'''
    n = len(S.T)//2
    SX = S[:,:n] # X-component
    SZ = S[:,n:] # Z-component
    return np.sum((SX * SZ) ^ SX ^ SZ,axis=-1) # adding weights of X and Z components double-counts Y operators

# @nb.njit(nb.types.Tuple((nb.int64,nb.int8[:]))(nb.int8[:,:],nb.int64))
def minWeightRowAll(L,tB):
    '''return all min weight rows of L'''
    wList = weightTB2nb(L) if tB==2 else np.sum(L,axis=-1)
    w = np.min(wList)
    return (w,L[wList==w],wList)

def mwUpdate(w,minRows, best=None, minWords=None,minWordCount=None):
    '''store counts of all lowest weight codewords'''
    if best is None or w < best[0]:
        best = (w,minRows[0])
        minWords = {}
        minWordCount = 0
    minWordCount += len(minRows)
    for lo in minRows:
        lo = tuple(lo) 
        if lo not in minWords:
            minWords[lo] = 0
        minWords[lo] += 1
    nExp = minWordCount/len(minWords)
    pErr = np.exp(-nExp)
    return best, minWords, minWordCount, nExp, pErr

def permMinRowsK(H,L,tB,ix):
    '''find Kernel of S with permuted rows ix, check whether anticommutes with logicals L and return min weight rows'''
    H,pivots,K = KerZ2(H,ix,0,True)
    if len(L) > 0:
        ## check for nontrivial logicals
        ixL = matMulZ2(K,L.T)
        ixL = np.sum(ixL,axis=-1) > 0
        ## include only nontrival logicals
        K = K[ixL]
    w, minRows,wRows = minWeightRowAll(K,tB)
    return w, minRows,wRows, pivots

####################################################
## Pyradko QDistRnd GAP script
####################################################

def DistRndGAP(H,L,tB,params={},seed=None):
    '''Pryadko distance finding of stabiliser codes using DistRndStab function'''
    if tB == 1:
        return GAPDistRandCSS(H,L,params,seed)
    if tB == 2:
        return GAPDistRndStab(H,params,seed)
    
def GAPDistRndStab(H,params={},seed=None):
    '''Pryadko distance finding of stabiliser codes using DistRndStab function'''
    '''Doesn't currently handle k=0'''
    myFile = randomFilename(seed=seed)
    r,n = H.shape
    nB = n//2
    ## convert to format X1 Z1 X2 Z2 ....
    G = Z2MatZeros((r,n))
    for i in range(nB):
        G[:,2*i] = H[:,i]
        G[:,2*i+1] = H[:,i+nB]
    filedir = f"{os.getcwd()}/tmp"
    dirCheck(filedir)
    Hfile = f'{filedir}/{myFile}H.mtx'
    
    ## Write MTX files for stab gens
    with open(Hfile,'w') as f:
        f.write(ZMat2mtx(G))

    ## Create GAP file
    steps = params['iterCount'] if 'iterCount'  in params else max(10,nB*nB//5)
    commands = ['LoadPackage("QDistRnd");;']
    commands.append(f'H := ReadMTXE("{Hfile}",0)[3];;')
    commands.append(f'DistRandStab(H,{steps},1,9);')
    commands = "\n".join(commands)
    gapFile = f'{filedir}/{myFile}.g'
    with open(gapFile,'w') as f:
        f.write(commands)
    
    ## run subprocess
    maxTime = params['maxTime'] if 'maxTime' in params else 8 * 60 * 60
    proclist = ['gap', '-b', gapFile, '-c', 'QUIT;']
    stdOut, stdErr = CLIRun(proclist,commands="\n".join(commands),maxTime=maxTime)

    ## Parse Result
    d,T,R,v = parseQDistRnd(stdOut)
    lo = Z2MatZeros(n)
    ## return to original order
    if len(v) == n:
        for i in range(nB):
            lo[i] = v[i*2]
            lo[i+nB] = v[i*2 + 1]
    ## clean up temp files
    os.remove(Hfile)
    os.remove(gapFile)
    return d,lo,stdOut

def GAPDistRandCSS(H,L,params={},seed=None):
    '''Pryadko distance finding of stabiliser codes using DistRndCSS function'''
    tB = 1
    if L is None:
        L = defaultLogicals(H,tB)
    myFile = randomFilename(seed=seed)
    r,n = H.shape
    nB = n//2
    ## convert to format X1 Z1 X2 Z2 ....
    filedir = f"{os.getcwd()}/tmp"
    dirCheck(filedir)
    Hfile = f'{filedir}/{myFile}H.mtx'
    Lfile = f'{filedir}/{myFile}L.mtx'
    
    ## Write MTX files for stab gens
    with open(Hfile,'w') as f:
        f.write(ZMat2mtx(H))
    with open(Lfile,'w') as f:
        f.write(ZMat2mtx(L))

    ## Create GAP file
    steps = params['iterCount'] if 'iterCount'  in params else max(10,nB*nB//5)
    commands = ['LoadPackage("QDistRnd");;']
    commands.append(f'H := ReadMTXE("{Hfile}",0)[3];;')
    commands.append(f'L := ReadMTXE("{Lfile}",0)[3];;')
    commands.append(f'DistRandCSS(H,L,{steps},1,9);')
    commands = "\n".join(commands)
    gapFile = f'{filedir}/{myFile}.g'
    with open(gapFile,'w') as f:
        f.write(commands)
    
    ## run subprocess
    maxTime = params['maxTime'] if 'maxTime' in params else 8 * 60 * 60
    proclist = ['gap', '-b', gapFile, '-c', 'QUIT;']
    stdOut, stdErr = CLIRun(proclist,commands="\n".join(commands),maxTime=maxTime)
    ## Parse Result
    d,T,R,v = parseQDistRnd(stdOut)
    lo = Z2MatZeros(n)
    ## return to original order
    if len(v) == n:
        lo = Z2Mat(lo)
    ## clean up temp files
    os.remove(Hfile)
    os.remove(Lfile)
    os.remove(gapFile)
    return d,lo,stdOut

def ZMat2mtx(A):
    '''convert matrix A to row col list format for dist-m4ri'''
    m,n = A.shape
    rList,cList = np.nonzero(A)
    r = len(rList)
    ## add 2 extra comment lines to deal with QDistRnd
    temp = ['%%MatrixMarket matrix coordinate integer general\n%\n%\n']
    temp += [f'{m} {n} {r}\n'] 
    temp += [f'{i+1} {j+1} {A[i,j]}\n' for i,j in zip(rList,cList)]
    return "".join(temp)

def parseQDistRnd(stdOut):
    '''parse result from DistRnd GAP Script'''
    T = 1
    R = 1
    d = 0
    v = Z2MatZeros(0)
    stdOut = stdOut.strip()
    if stdOut == "The found distance 1<=1 too small, exiting!":
        d = 1
        return d,T,R,v
    ## logical operators are often split over multiple lines ending with "\" - convert to single line 
    stdOut = stdOut.replace("\\\n "," ")
    stdOut = stdOut.split("\n")
    stdOut = [myrow.strip() for myrow in stdOut]
    stdOut = [myrow for myrow in stdOut if len(myrow) > 0]
    
    i = 0
    myrow = stdOut[i].split(" rounds of ")
    T = int(myrow[i])
    i+=1

    if stdOut[i] == 'First vector found with lowest weight:':
        i+=1
        v = stdOut[i].replace(".",'0').split(" ")
        v = Z2Mat(list(map(int,v)))
        i+=1
    myStr = 'Minimum weight vector found '
    l = len(myStr)
    myrow = stdOut[i]
    if myrow[:l] == myStr:
        myrow = myrow[l:].split(" ")
        R = int(myrow[0])
    i += 1
    myrow = stdOut[i]
    if myrow[:2] == "[[":
        myrow = myrow[2:].split("]];")
        params = myrow[0].split(",")
        n,k,d = map(int,params)
    return d,T,R,v

####################################################
## dist-m4ri from https://qec-pages.github.io/dist-m4ri/
####################################################

def dist_m4ri_CSS(H,L=None,method=1,params={},seed=0):
    '''Pryadko distance finding package in C for CSS codes and classical codes: https://github.com/QEC-pages/dist-m4ri
    H: check matrix 
    L: logicals (None for classical codes or states)
    params: parameters in dict form
    method: 1 or 2 for RW or CC algorithms respectively

    1: random window (RW) algorithm. Options:
        steps=[int]: how many information sets to use (1)
        wmin=[int]:  minimum distance of interest (1)
        wmax=[int]:  if non-zero, ignore vectors of this and larger wgt (0)

    2: connected cluster (CC) algorithm.  Options:
        wmax=[int]:  maximum cluster weight, exclusive (0)
                must be non-zero for CC only, otherwise use upper bound from RW
        smax=[int]:  maximum syndrome weight, inclusive (0)
                must be non-zero to calculate confinement
        start=[int]: use only this position to start (-1)
    '''
    myFile = randomFilename(seed=seed)
    myDir = f'{os.getcwd()}/tmp'
    dirCheck(myDir)
    finH = f"{myDir}/{myFile}H.mtx"
    finL = f"{myDir}/{myFile}L.mtx"
    k,n = H.shape
    stepDefault = max(10,n*n//5)
    ## fix bug - double-use of method
    if 'method' in params:
        del params['method']
    if 'CCstart' in params:
        params['start'] = params['CCstart']
    debugLevel = 0b100000011011
    if method==2:
        myparams = {'method':2,'debug':debugLevel,'wmax':0,'smax':0,'start':-1,'seed':seed,'finH':finH}
    else:
        myparams = {'method':1,'debug':debugLevel,'steps':stepDefault,'wmin':1,'wmax':0,'finH':finH}
    for k in myparams:
            if k in params:
                myparams[k] = params[k]
    
    if 'iterCount' in params and 'steps' in myparams:
        myparams['steps'] = params['iterCount']
    if L is not None and np.sum(L) == 0:
        L = None
    if L is not None:
        ## otherwise, save logicals to file
        myparams['finL'] = finL
        scipy.io.mmwrite(myparams['finL'],scipy.sparse.csr_array(L))
    ## save stabilisers to file
    scipy.io.mmwrite(myparams['finH'],scipy.sparse.csr_array(H))

    ## run subprocess
    maxTime = params['maxTime'] if 'maxTime' in params else 8*3600
    proclist = ['dist_m4ri'] + [f'{k}={v}' for k,v in myparams.items()]
    stdOut, stdErr = CLIRun(proclist,maxTime=maxTime,captureErrors=False)
    ## extract low-weight codeword from results
    temp = []
    for myline in stdOut.split("\n"):
        j = myline.find('[')
        if(j >=0):
            temp = [int(a) for a in myline[j+1:-1].split(" ")]
    os.remove(finH)
    if L is not None:
        os.remove(finL)
    return temp,stdOut

def dist_m4ri_RW(S,L,tB=2,params={},seed=0):
    '''Random Window Algorithm'''
    addI = False
    if tB == 2:
        if L is None:
            L = ZMatVstack(getLogicalPaulis(S))
        ## 2-block non-CSS code
        tB = params['GF4blockRep']
        if tB < 3 or tB > 4:
            ## 4-block by default - most accurate
            tB = 4
        ## addI - restrict to even-weight errors
        addI = True
        S = encode2block(S,addI=addI,tB=tB)
        L = encode2block(L,addI=False,tB=tB)    
    r,n = S.shape
    ## not helpful to set wmax for RW algorithm
    # params['wmax'] = min(n,99)
    progressText = []
    lo, p1 = dist_m4ri_CSS(S,L,method=1,params=params,seed=seed)
    progressText.append(p1)
    w = len(lo)
    ## when adding I blocks, all errors are even weight
    if addI:
        w = w//2 
    ## adjust from 1-base to 0-base
    lo = ZMat(lo) - 1
    lo = set2Bin(n,lo,dtype=np.int8)
    lo = decodeBlock(lo,tB=tB)
    return  w,lo, "; ".join(progressText)

def dist_m4ri_CC(S,L,tB=2,params={},seed=0):
    '''Connected Cluster algorithm'''
    if tB == 2:
        ## 2-block non-CSS code
        if L is None:
            L = ZMatVstack(getLogicalPaulis(S))
        tB = params['GF4blockRep']
        if tB < 3 or tB > 4:
            ## 3-block by default - fastest
            tB = 3
        ## addI for 4-block only - for 3-block, we allow odd-weight errors
        addI = True if tB==4 else False
        S = encode2block(S,addI=addI,tB=tB)
        L = encode2block(L,tB=tB)    
    r,n = S.shape
    ## due to error message: increase MAX_W=100 defined in 'util_io.h'
    params['wmax'] = min(n,99)
    progressText = []
    lo, p1 = dist_m4ri_CSS(S,L,method=2,params=params,seed=seed)
    progressText.append(p1)
    w = len(lo)
    ## when adding I blocks, all errors are even weight
    if addI:
        w = w//2 
    lo = set2Bin(n,lo,dtype=np.int8)
    lo = decodeBlock(lo,tB=tB)
    return  w,lo, "; ".join(progressText)

####################################################
## Meet in the Middle method from https://github.com/qiskit-community/qiskit-qec/blob/main/src/qiskit_qec/analysis/distance.py
####################################################

def MeetMiddleMW(S,L,tB,params={}):
    '''Meet in the middle distance finding from src/qiskit_qec/analysis/distance.py'''
    QiskitSyndromes = dict()
    if L is None:
        L = defaultLogicals(S,tB)
    k,nB = L.shape
    if tB == 2:
        # 3-block encoding for non-CSS codes
        S = encode2block(S,tB=3)
        L = encode2block(L,tB=3)
        nB = nB//2
    HT = [frozenset(bin2Set(c)) for c in S.T]
    LT = [frozenset(bin2Set(c)) for c in L.T]
    # check for weight 1 errors
    for e in range(nB):
        if len(HT[e]) == 0 and len(LT[e]) > 0:
            return 1
    for d in range(2,nB+1):
        w1 = d//2
        w2 = d-w1
        for l in range(k):
            sComm1,sAntiComm1 = getComm(HT,LT,nB,w1,l,tB,QiskitSyndromes)
            if w1 != w2:
                sComm2,sAntiComm2 = getComm(HT,LT,nB,w2,l,tB,QiskitSyndromes)
                if len(sComm1.intersection(sAntiComm2)) or len(sComm2.intersection(sAntiComm1)):
                    return d
            else:
                if len(sComm1.intersection(sAntiComm1)):
                    return d
    return d

def getSyndromes(HT,LT,nB,w,tB,QiskitSyndromes):
    if w not in QiskitSyndromes:
        temp = set()
        if tB==2:
            ## 3-block encoding
            for c in iter.combinations(range(nB),w):
                for t in iter.product(range(3),repeat=w):
                    E = [i + nB * j for (i,j) in zip(c,t)]
                    temp.add ((setXOR(HT,E),setXOR(LT,E)))
            QiskitSyndromes[w]  = temp
        else:
            QiskitSyndromes[w] = {(setXOR(HT,E),setXOR(LT,E)) for E in iter.combinations(range(nB),w)}
    return QiskitSyndromes[w]

def getComm(HT,LT,nB,w,l,tB,QiskitSyndromes):
    sComm = set()
    sAntiComm = set()
    if w > 0:
        for (Sh,Sl) in getSyndromes(HT,LT,nB,w,tB,QiskitSyndromes):
            if l not in Sl:
                sComm.add(Sh)
            else:
                sAntiComm.add(Sh)
    return sComm,sAntiComm

####################################################
## connectedClusterMW/UndetectableErrorMW
####################################################

def connectedClusterMW(H,L,method,params):
    '''Colour code distance function'''
    start = params['CCstart'] - 1 if 'CCstart' in params else -1
    wDict = None
    if method=='GraphLikeErrorMW':
        # return UndetectableErrorMW(DEMH,DEML,DEMax=2,SMax=2,SMin=2,SInc=False,start=start)
        d,lo,progress =  CCMW(H,L,DEMax=2,SMax=2,SInc=False,start=start)
    elif method =='ColourCodeDistMW':
        # return UndetectableErrorMW(DEMH,DEML,DEMax=3,SMax=3,SMin=3,SInc=False,start=start)
        d,lo,progress =  CCMW(H,L,DEMax=3,SMax=3,SInc=False,start=start)
    elif method =='UndetectableErrorMW':
        d,lo,wDict,progress = UndetectableErrorMW(H,L,start=start)
    else:
        d,lo,progress = CCMW(H,L,start=start)
    return d,lo,wDict,progress

def UndetectableErrorMW(DEMH,DEML,DEMax=0,SMax=10,SMin=0,SInc=True,start=-1):
    '''MW version of Stim search_for_undetectable_logical_errors function'''
    ## originl number of detectors and errors
    r, n = DEMH.shape
    minCount = 2
    best = [n,ZMatZeros(n)]
    wDict = ZMatZeros(n)
    # DEMax= r if DEMax==0 else DEMax
    SMin= 2 if SMin==0 else SMin
    ## increment DEMax up to errPerDetMax to find logicals
    progressText = []
    for wS in range(SMin,SMax+1):
        progressText.append(f'SMax={wS} DEMax={DEMax} SInc={SInc}')
        ## search for logical - if w>0 then we found a non-trivial logical
        w, lo, p1 = CCMW(DEMH,DEML,DEMax=DEMax,SMax=wS,SInc=SInc,start=start)
        progressText.append(p1)
        wDict[w] +=1
        if w > 0:
            if w <= best[0]:
                best = w, lo
                if terminateMinCount(wDict,minCount):
                    progressText.append(f'Breaking - same result {minCount} times')
                    break
    w, lo = best
    return w, lo, wDict,"; ".join(progressText)

def terminateMinCount(wDict,minCount=2):
    '''termination condition for Connected Cluster/Stim Undetectable Error methods
    wDict: np array with frequencies for each weight from 0 to n
    minCount: terminate if the smallest non-zero count is more than minCount
    '''
    ## non-zero entries - exclude w=0
    wList = np.nonzero(wDict[1:])
    ## smallest entry
    w = wList[0][0] + 1
    ## have we seen the smallest entry at least minCount times?
    return wDict[w] >= minCount

def CCMW(H,L,DEMax=0,SMax=0,SInc=True,start=-1):
    '''Pryadko Connected Cluster Algorithm for LDPC codes - main function'''
    r,n = H.shape
    HT = [set(bin2Set(c)) for c in H.T]
    LT = [set(bin2Set(c)) for c in L.T]
    eList = {i for i in range(n)} if DEMax == 0 else {i for i in range(n) if len(HT[i]) <= DEMax}
    HE = [eList.intersection(bin2Set(r)) for r in H]
    n = len(HT)
    progressText = []
    sList,lList = [set()] * n, [set()] * n
    for d in range(1,n):
        progressText.append(f'Checking w={d}')
        (eFirst,eLast) = (0, n) if (start <0) else (start,start+1)
        for e0 in range(eFirst,eLast):
            E0 = {e0}
            sList[1] = HT[e0]
            lList[1] = LT[e0]
            E = CCMWRec(HT,LT,HE,d,1,e0,E0,sList,lList,SMax,SInc)
            if E is not False:
                progressText.append(f'Found w:{d}')
                lo = set2Bin(n,E,dtype=np.int8)
                return d,lo,"; ".join(progressText) 
    return 0,Z2MatZeros(n),"; ".join(progressText) 

def CCMWRec(HT,LT,HE,d,w,e0,E0,sList,lList,SMax,incS):
    '''recursive step for CCMW'''
    if w == d:
        ## w == d: check logical parities
        if (len(sList[w]) == 0) and (len(lList[w]) > 0):
            return E0
    else:
        ## generate errors of weight w+1
        if len(sList[w]) > 0:
            j = min(sList[w])
            w += 1
            for k in HE[j]:
                if (k > e0) and (k not in E0):
                    sList[w] = sList[w-1] ^ HT[k]
                    if ((incS) or len(sList[w-1]) >= len(sList[w])) and ((SMax==0 ) or SMax >= len(sList[w])):
                        lList[w] = lList[w-1] ^ LT[k]
                        E0.add(k)
                        res = CCMWRec(HT,LT,HE,d,w,e0,E0,sList,lList,SMax,incS)
                        if res is not False:
                            return res
                        E0.remove(k)
    return False

############################################################################
## Stim Distance-finding Methods
############################################################################

def StimCodeDist(DEMH,DEML,params):
    '''Main wrapper function for H, L input'''
    ## construct a Stim quantum circuit object
    qc = Stabs2StimCircuit(DEMH,DEML,tB=1)
    return StimDist(qc,DEMH,DEML,params['method'])   

def StimDist(qc,DEMH,DEML,method):
    '''Main wrapper function for stim circuit input'''
    r = len(DEMH)
    if method == 'GraphLikeErrorStim':
        d,EList,wDict,progress  =  UndetectableErrorStim(qc,errPerDetMin=2,errPerDetMax=2,detPerErrMax=2,IncSyndrome=True)
    elif method == 'ColourCodeDistStim':
        d,EList,wDict,progress  =  UndetectableErrorStim(qc,errPerDetMin=3,errPerDetMax=3,detPerErrMax=r,IncSyndrome=True)
    else:
        d,EList,wDict,progress  =  UndetectableErrorStim(qc,errPerDetMin=2,errPerDetMax=10,detPerErrMax=r,IncSyndrome=True)
    ## UndetectableErrorStim returns a list of explained errors - convert these back to a logical operator of the DEM
    lo = StimExplainErr2lo(EList,DEMH,DEML)
    return d,lo,wDict,progress

def UndetectableErrorStim(qc,errPerDetMax=2,detPerErrMax=2,IncSyndrome=True,errPerDetMin=2):
    '''Iterative method calling StimUE with increasing DEMax from errPerDetMin to detPerErrMax'''
    DEMax = errPerDetMin
    n = qc.num_qubits
    wDict = ZMatZeros(n)
    best = (n,[])
    progressText = []
    minCount = 2
    while DEMax <= errPerDetMax:
        progressText.append(f'dont_explore_detection_event_sets_with_size_above={DEMax},dont_explore_edges_with_degree_above={detPerErrMax}')
        try:
            if DEMax == 2 and detPerErrMax==2:
                myErrs = qc.shortest_graphlike_error()
            else:
                myErrs = qc.search_for_undetectable_logical_errors(
                    ## only detector rows with weight at most DEMax - truncate this one
                    dont_explore_detection_event_sets_with_size_above=DEMax, 
                    ## degree = only detector cols with weight at most r - may not need to be truncated for small codes; 2 is graph-like search, 3 for colour code
                    ## can truncate this also for certain code types - eg CSS codes - upper bound on syndrome weight - eg set to 3 for colour codes
                    # code capacity CSS code - set to 
                    dont_explore_edges_with_degree_above= detPerErrMax, 
                    ## Choose False for dense codes and expanders
                    dont_explore_edges_increasing_symptom_degree=(not IncSyndrome),
                    canonicalize_circuit_errors=True)
            
            w = len(myErrs)
            progressText.append(f'Found w:{w}')
            wDict[w] +=1
            if w > 0:
                if w <= best[0]:
                    best = w, myErrs
                    if terminateMinCount(wDict,minCount):
                        progressText.append(f'Breaking - same result {minCount} times')
                        break
        except:
            progressText.append(f'no lo found')
            wDict[0] +=1
        DEMax += 1
    w, myErrs = best
    return len(myErrs),myErrs,wDict,"; ".join(progressText)

def StimExplainErr2lo(EList,DEMH,DEML):
    '''map a list of StimExplainErr objects back to a logical operator corresponding to DEMH,DEML'''
    r, n = DEMH.shape
    k, n = DEML.shape
    DEM = ZMatVstack([DEMH,DEML])
    ## dictionary of column vectors of DEM
    DEMDict = {tuple(DEM[:,i]): i  for i in range(len(DEM.T))}
    ## logical operator to update and return - binary vector of length n
    lo = Z2MatZeros(n)
    for e in EList:
        ## DEM column vector 
        v = Z2MatZeros(r+k)
        ## list of detectors/logicals in for D1, L2 etc
        DT = [str(T.dem_target) for T in e.dem_error_terms]
        ## process DT and update v
        for Tix in DT:
            T, ix = Tix[0],int(Tix[1:])
            ix = ix if T == 'D' else r + ix
            v[ix] = 1
        ## look v up in the DEMDict
        vTup = tuple(map(int,v))
        if vTup in DEMDict:
            ix = DEMDict[vTup]
            ## update logical operator
            lo[ix] = 1
        else:
            return Z2MatZeros(n)
    return lo

##########################################
## QDistEvol - Evolutionary Algorithm for Distance
##########################################

def QDistEvol(S,L,tB=1,params={},seed=None):
    '''Evolutionary algorithm
    For QDistRndMW to have same accuracy as m4riRW, use the following parameters:
    GF4blockRep = 4
    regroupPerm = 0
    Optimal settings for QDistEvol:
    GF4blockRep = 2
    regroupPerm = 1'''
    rnd = np.random.default_rng(seed)
    r,n = S.shape
    if L is None:
        L = ZMatVstack(getLogicalPaulis(S)) if tB==2 else Z2MatZeros((0,n))
    paramDefaults = {
        'method':'QDistEvol',
        'iterCount': 10000,
        'maxErr': -1,
        'GF4blockRep': 2,
        'regroupPerm': 1,
        'HL': False,
        'genCount':100,
        'offspring':10,
        'pMut':2.0,
        'sMut':1.0,
        'tabuLength':0,
        'swapPivot':1,
        'swapBlockorder':0,
        'pMutScale': 50
    }
    setDefaultParams(params,paramDefaults)
    ## QDistRnd is QDistEvol with 1 generation only
    if params['method'] == 'QDistRndMW':
        params['genCount'] = 1
    ## partition the total number of trials into 
    popSizes = partitionTrials(params['iterCount'], params['genCount'])
    ## for special case tabuLength = 1, set tabuLength to be the code length
    if params['tabuLength'] == 1:
        params['tabuLength'] = n
    ## convert to binary matrices
    S = Z2Mat(S)
    L = Z2Mat(L)
    progressText = []
    addI = False
    ## Handle non-CSS codes
    if tB == 2:
        tB = params['GF4blockRep']
        if tB > 1:
            addI = True if tB > 2 else False
            S = encode2block(S,addI,tB)
            L = encode2block(L,False,tB)
    else:
        params['GF4blockRep'] = 1
    r,n = S.shape
    wDict = np.zeros(n+1,dtype=int)
    w, minRows = n, Z2MatZeros((1,n))
    best, minWords, minWordCount, nExp, pErr = mwUpdate(w,minRows)
    population = []
    tabulist = []
    TrialId = 0
    for genId in range(params['genCount']):
        genText = "" if params['genCount'] ==  1 else f'Gen:{genId} '
        popMetric = []
        pivotList = []
        popCount = popSizes[genId]
        for i in range(popCount):
            ## generate random permutations at generation genId == 0:
            if genId == 0:
                ix = rnd.permutation(n)
                if params['regroupPerm'] and tB > 1:
                    ix = regroupPerm(ix,tB)
                ## only save to population if genCount > 1
                if params['genCount'] > 1:
                    population.append(ix)
            else:
                ix = population[i]
            w, minRows,wRows,nonPivots =  permMinRowsK(S,L,tB,ix)
            if addI:
                w = w//2
            wDict[w] += 1
            if w < best[0]:
                progressText.append(f'{genText}Trial:{TrialId} w:{w}')
                genText = ""
            ## only save popMetric and pivotList if genCount > 1
            if params['genCount'] > 1:
                popMetric.append((w,np.average(wRows)))
                pivotList.append(nonPivots)
            best, minWords, minWordCount, nExp, pErr = mwUpdate(w,minRows, best, minWords,minWordCount)
            TrialId += 1
        ## only select and mutate if we are not at the last generation
        if genId + 1 < params['genCount']:
            popSizeNext = popSizes[genId+1]
            mu = max(1,popSizeNext // params['offspring'])
            offspringCounts = partitionTrials(popSizeNext,mu)
            ixSort = argsort(popMetric)[:mu]
            parents = [(population[j],pivotList[j],offspringCounts[i]) for i,j in enumerate(ixSort)]
            population = []
            for (ix,nonPivots,offspring) in parents:
                population += distGeneticMutateBatch(ix,nonPivots,params,tabulist,rnd,count=offspring) 
    w,lo = best
    lo = decodeBlock(lo,tB=tB)
    progressText.append(f'pErr:{pErr:.4f}, weight {w} CW found:{len(minWords)}, wDict:{wDist2str(wDict)}')
    return w,lo,wDict, "; ".join(progressText)

def distGeneticMutateBatch(ix,pivots,params,tabulist,rnd,count=1):
    '''Mutate from parent permutation ix'''
    n = len(ix)
    tB = params['GF4blockRep']
    nB = n//tB
    if params['swapPivot']:
        if tB > 2:
            ## we added I block - so don't consider the last nB pivots
            S1 = np.mod(pivots[nB:],nB)
        elif tB == 2:
            S1 = np.mod(pivots,nB)
        else:
            S1 = pivots
        S2 = np.mod(invRange(nB,S1),nB)
    else:
        S1 = np.arange(nB)
        S2 = S1
    StB = np.arange(tB)
    temp = []
    pMut = params['pMut']
    sMut = params['sMut']
    ##  scale number of transpositions by code length
    if params['pMutScale'] > 0:
        pMut = n / params['pMutScale']
    for i in range(count):
        done = False
        transpCount = int(np.round(pMut * (1 + rnd.normal() * sMut )))
        if transpCount < 1:
            transpCount = 1
        c = 0
        while not done:
            ## intial perm of length n
            ix0 = np.arange(n)
            aList = rnd.choice(S1,size=transpCount)
            bList = rnd.choice(S2,size=transpCount)
            ## transposition of original n qubits - by block if tB > 1 eg 3-block x|z|x+z
            for a,b in zip(aList,bList):
                for j in range(tB):
                    ix0[a+j*nB],ix0[b+j*nB] = ix0[b+j*nB],ix0[a+j*nB]
            ## transpose between x|z|x+z block of the same qubit if tB > 1
            if tB > 1 and params['swapBlockorder']:
                for a in rnd.choice(S2,size=transpCount):
                    c,d = rnd.choice(StB,size=2,replace=False)
                    ix0[nB*c + a],ix0[nB*d + a] = ix0[nB*d + a],ix0[nB*c + a]
            ## permute original order of qubits, then apply ix
            ix2 = ix0[ix]
            if params['tabuLength'] > 0:
                ixtup = tuple(ix2)
                if ixtup not in set(tabulist):
                    if len(tabulist) >= params['tabuLength']:
                        tabulist.pop(0)
                    tabulist.append(ixtup)
                    done=True
                c+=1
                if c > 5:
                    done = True
            else:
                done = True
        temp.append(ix2)
    return temp

####################################################
## Mixed-Integer Programming - Gurobi
####################################################

def gurobiDist(S,L=None,tB=1,params={}):
    '''Distance via Gurobi Integer Programming
    Supports either two or three-block encoding for non-CSS codes
    Two-block representation gives faster results (due to lower number of variables)'''
    paramDefaults = {
        'verbose': False,
        'maxTime': 3600 * 8,
        'nThreads': 1,
        'GF4blockRep': 2
    }
    params = setDefaultParams(params,paramDefaults)
    if L is None:
        L = defaultLogicals(S,tB)
    if tB == 2:
        tB =  params['GF4blockRep']
        S = encode2block(S,tB=tB)
        L = encode2block(L,tB=tB)
    model = gurobiDistModel(S,L,tB)
    # Set flags
    model.params.OutputFlag = 1 if params['verbose'] else 0
    model.params.timelimit = params['maxTime']
    model.setParam("Threads", params['nThreads'])
    # Solve and return result
    model.update()
    # Formulate problem
    model.optimize()
    progress = ""
    r,n = S.shape
    if model.ObjVal == float("inf"):
        return 0,Z2MatZeros(n),progress
    else:
        x = Z2Mat([int(model.getVarByName(f"x_{i}").X) for i in range(n)])
            ## for quantum codes, we use the symplectic inner product
        x = decodeBlock(x,tB=tB)
        return round(model.ObjVal),x,progress

def gbWeight(x,tB=2):
    '''Weight calculation function for Gurobi variables'''
    if tB != 2:
        return sum(x)
    ## symplectic weight
    n = len(x)//2
    return sum(x) - sum(x[:n] * x[n:])

def gurobiDistModel(S,L,tB):
    '''Construct Gurobi model for distance finding'''
    licence = {
        'WLSACCESSID':'562a5ce8-f341-42e0-b7ce-eba4904a53ec',
        'WLSSECRET':'fbd35f02-f7a0-42ba-8f06-6b28907c5628',
        'LICENSEID':2757157
    }
    env = gp.Env(params=licence)
    model = gp.Model(env=env)
    r,n = S.shape
    # Define variables
    x = np.empty(n, dtype=gp.Var)
    for i in range(n):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")
    model.setObjective(gbWeight(x,tB=tB), GRB.MINIMIZE)
    model.addConstr(sum(x) >= 1, name='nontrivial')
    # Add slack variables for modulo 2 arithmetic
    slack1 = np.ndarray(shape=(r,), dtype=gp.Var)
    for j in range(r):
        slack1[j] = model.addVar(vtype=GRB.INTEGER, name=f"slack_1_{j}")
        # Constrain x to be in right kernel of GF(2) check_matrix
        model.addConstr(
            sum(x[bin2Set(S[j])]) == 2 * slack1[j], name=f"check_{j}"
        )
    # Constrain solution to have odd overlap with at least one logical
    if (L is not None) and (len(L) > 0):
        k = len(L)
        parity = np.empty(k, dtype=gp.Var)
        slack2 = np.ndarray(shape=(k,), dtype=gp.Var)
        for i in range(k):
            parity[i] = model.addVar(vtype=GRB.BINARY, name=f"parity_{i}")
            slack2[i] = model.addVar(vtype=GRB.INTEGER, name=f"slack_2_{i}")
            model.addConstr(
                sum(x[bin2Set(L[i])]) == parity[i] + 2 * slack2[i],
                name=f"logical_{i}",
            )
        model.addConstr(sum(parity) >= 1, name="nonzero_parity")
    return model

#############################################################
##  OR-Tools MIP: multiple solver-types supported
#############################################################

def MIPDist(DEMH,DEML=None,solverType='SCIP',params={}):
    '''Distance via OR-Tools Integer Programming
    Possible solverType:
    CBC_MIXED_INTEGER_PROGRAMMING or CBC
    SAT_INTEGER_PROGRAMMING or SAT or CP_SAT
    SCIP_MIXED_INTEGER_PROGRAMMING or SCIP
    GUROBI_MIXED_INTEGER_PROGRAMMING or GUROBI or GUROBI_MIP - requires full installation of Gurobi - https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer
    CPLEX_MIXED_INTEGER_PROGRAMMING or CPLEX or CPLEX_MIP - requires installation of CPLEX https://www.ibm.com/docs/en/icos
    XPRESS_MIXED_INTEGER_PROGRAMMING or XPRESS or XPRESS_MIP - requires installation of Xpress https://www.fico.com/fico-xpress-optimization/docs/latest/getting_started/dhtml/chap16.html
    GLPK_MIXED_INTEGER_PROGRAMMING or GLPK or GLPK_MIP - requires installation of GLPK: https://www.gnu.org/software/glpk/
    '''
    paramDefaults = {
        'verbose': False,
        'maxTime': 3600 * 8,
        'nThreads': 1
    }
    params = setDefaultParams(params,paramDefaults)
    r,n = DEMH.shape
    solver =  ortools.linear_solver.pywraplp.Solver.CreateSolver(solverType)
    infinity = solver.infinity()
    # Define variables
    x = {}
    objective = solver.Objective()
    constr = solver.RowConstraint(1, infinity, "x non-trivial")
    for i in range(n):
        x[i] = solver.BoolVar(f"x_{i}")
        objective.SetCoefficient(x[i], 1)
        constr.SetCoefficient(x[i], 1)
    objective.SetMinimization()

    # Add slack variables for modulo 2 arithmetic
    slackH = {}
    for j in range(r):
        slackH[j] = solver.IntVar(0, infinity, f"sH_{j}")
        # require x H = 0 mod 2
        constr = solver.RowConstraint(0, 0, f"slackH_{j}")
        constr.SetCoefficient(slackH[j],-2)
        for i in bin2Set(DEMH[j]):
            constr.SetCoefficient(x[i], 1)
        
    # Constrain solution to have odd overlap with at least one logical
    if (DEML is not None) and (len(DEML) > 0):
        k = len(DEML)
        parityL = {}
        slackL = {}
        for j in range(k):
            parityL[j] = solver.BoolVar(f"pL{j}")
            slackL[j] = solver.IntVar(0, infinity, f"sL{j}")
            # ensure x L = pL mod 2
            constr = solver.RowConstraint(0, 0, f"slackL_{j}")
            constr.SetCoefficient(slackL[j],-2)
            constr.SetCoefficient(parityL[j],-1)
            for i in bin2Set(DEML[j]):
                constr.SetCoefficient(x[i], 1)
        # ensure pL != 0
        constr = solver.RowConstraint(1,infinity, f"parityL_{j}")
        for i in range(k):
            constr.SetCoefficient(parityL[i], 1)
    # Set model parameters
    solver.SuppressOutput()
    solver.SetNumThreads( max(1,params['nThreads'])) 
    ## Time limit in milliseconds
    solver.set_time_limit( params['maxTime'] *1000 ) 
    # Solve and return result
    solver.Solve()
    d = round(solver.Objective().Value())
    x = Z2Mat([ round(x[j].solution_value()) for j in range(n)])
    progressText = []
    progressText.append(f"w:{d}")
    progressText.append(f"T:{solver.wall_time():d}ms")
    progressText.append(f"iterations:{solver.iterations():d}")
    # progress.append(f"Problem solved in {solver.nodes():d} branch-and-bound nodes")
    progressText = "; ".join(progressText)
    if d >= n:
        return 0,Z2MatZeros(n),progressText
    else:
        return d,x,progressText

####################################################
## SAT Methods
####################################################

def CLISATDist(H,L,params={},seed=0):
    '''run SAT solver from command line interface'''
    r,n = H.shape
    SATstr = DEM2WCNF(H,L)
    SATbinary = 'cashwmaxsatcoreplus'
    fileName = randomFilename()
    myDir = f'{os.getcwd()}/tmp'
    myFile = f"{myDir}/SAT{fileName}.wcnf"
    with open(myFile,'w') as f:
        f.write(SATstr)
    command = [SATbinary, '-bm', '-m', myFile ]
    stdOut,stdErr = CLIRun(command)
    d, lo = parseSATResult(stdOut)
    if len(lo) >= n:
        lo = lo[:n]
    else:
        lo = Z2MatZeros(n)
    os.remove(myFile)
    return d, lo, stdOut

def pySATdist(H,L,params):
    '''Distance via pySAT solver'''
    paramDefaults={
        ## solver options: cd, cd15, cd19, cms, gc3, gc4, g3, g4, g42, lgl, mcb, mcm, mpl, mg3, mc, m22, mgh
        'pySATsolver':'g4',
        ## binary options: rc2, fm, lsu
        'pySATbinary':'lsu',
        ## 8 hours max execution
        'maxTime':3600*8
    }
    params = setDefaultParams(params,paramDefaults)
    SATstr = DEM2WCNF(H,L)
    myDir = f'{os.getcwd()}/tmp'
    myFile = f"{myDir}/{randomFilename()}.wcnf"
    with open(myFile,'w') as f:
        f.write(SATstr)
    funcParams = [f'{params['pySATbinary']}.py', '-vv','-s',params['pySATsolver']]
    if params['pySATbinary'] == 'lsu':
        ## LSU doesn't print model by default
        funcParams.append("-m")
    funcParams.append(myFile)
    res, err = CLIRun(funcParams,maxTime=params['maxTime'])
    w,lo = parseSATResult(res)
    n = len(H.T)
    x = Z2MatZeros(n)
    ## lo is a series of variables - positive if set and negative if not set
    if len(lo) >= n:
        for i in range(n):
            if lo[i] > 0:
                x[i] = 1
    os.remove(myFile)
    return w,x,res.replace("\n","; ")

def DEM2WCNF(DEMH,DEML):
    '''convert DEM Matrix to WCNF SAT problem'''
    clauses,v = DEM2SATClauses(DEMH,DEML)
    return SATClauses2WCNF(clauses,v)

def DEM2SATClauses(DEMH,DEML):
    '''Convert DEM to SAT Clauses'''
    r = len(DEMH)
    DEM = ZMatVstack([DEMH,DEML])
    dl,nErr = DEM.shape
    ## errors - clause weight 1 to minimises Pauli weight
    clauses = [(1,[-e],0) for e in range(1,nErr+1)]
    v = nErr
    DLVars = ZMatZeros(dl)
    for e in range(1,nErr+1):
        for d in bin2Set(DEM.T[e-1]):
            dCurr = DLVars[d]
            if dCurr == 0:
                DLVars[d] = e
            else:
                ## new variable v for e XOR dCurr
                v+=1
                ## four new clauses to implement XOR - 0 means an essential clause
                clauses.extend([(0,[e,dCurr,-v],0),(0,[e,-dCurr,v],0),(0,[-e,dCurr,v],0),(0,[-e,-dCurr,-v],0)])
                ## update DLVars to v
                DLVars[d] = v
    ## add clauses ensuring that error commutes with stabilisers
    clauses.extend([(0,[-DLVars[d]],0)for d in range(r)])
    ## add clause ensuring that error anti-commutes with at least one logical 
    if r < dl:
        clauses.append((0,DLVars[r:],0))
    return clauses,v

def SATClauses2WCNF(clauses,v):
    '''Convert SAT Clauses to WCNF string'''
    wMax = len(clauses) + 1
    temp = [f'p wcnf {v} {len(clauses)} {wMax}']
    for c in clauses:
        wc = wMax if c[0]==0 else c[0]
        vListc = " ".join(map(str,c[1]))
        temp.append(f'{wc} {vListc} {c[2]}')
    return "\n".join(temp)

def parseSATResult(mytext):
    '''parse result of CLI SAT solver'''
    d = 0
    v = ""
    for myline in mytext.split("\n"):
        myline = myline.strip()
        if len(myline) > 0:
            if  myline[0] == "o":
                d = int(myline[2:])
            elif myline[0] == 'v':
                v = myline[2:]
    v = ZMat(list(map(int,v.split(" ")))) if len(v) > 0 else []
    return d, v

###########################################################
## Decoder Methods
###########################################################

def decoderDist(H,L,params={},seed=None):
    '''Distance via syndrome decoders
    set maxErr to a value 0 < maxErr <= 1 to terminate when probability of error less than pErr'''
    r,n = H.shape
    rnd = np.random.default_rng(seed)
    paramDefaults = {
        'decoder': 'bposd',
        'iterCount': 1,
        'maxErr':-1,
        'priors':None,
        'addStabs':0,
        'permuteDEM': 1
    }
    params = setDefaultParams(params,paramDefaults)
    k = len(L)
    s = Z2Mat([0] * r + [1])
    Hlo = ZMatVstack([H,Z2MatZeros((1,n))])
    wDict = ZMatZeros(n+1)
    progressText = []
    w, minRows = n, Z2MatZeros((1,n))
    best, minWords, minWordCount, nExp, pErr = mwUpdate(w,minRows)
    ix = np.arange(n)
    for TrialId in range(params['iterCount']):
        ## make a random non-trivial lo
        x = NonZeroVecRand(k)
        lo = Z2MatZeros(n)
        for j in bin2Set(x):
            lo ^= L[j]
        ## add randomly generated set of stabilisers to lo
        if params['addStabs']:
            x = rnd.integers(2,size=r)
            for j in bin2Set(x):
                lo ^= H[j]
        Hlo[-1] = lo
        ## find a correction
        decoderType = params['decoder']
        ## randomly permute cols of HL to more evenly sample from codewords
        if params['permuteDEM']:
            ix = rnd.permutation(n) 
        if decoderType=='bposd':
            lo = bposdDecode(Hlo[:,ix],s,params=params)
        elif decoderType=='bplsd':
            lo = bplsdDecode(Hlo[:,ix],s,params=params)
        elif decoderType in {'tesseract','tesseractLongbeam'}:
            lo = tesseractDecode(Hlo[:,ix],s,params=params)
        ## reverse permutation for lo
        if params['permuteDEM']:
            lo = lo[ixRev(ix)]
        w = np.sum(lo)
        wDict[w] += 1
        lo2D = ZMat2D(lo)
        ## check that the correction matches the syndrome
        check= np.sum(matMulZ2(lo2D,Hlo.T)[0] ^ ZMat2D(s)) == 0
        if check:
            minRows = lo2D
            if w < best[0]:
                progressText.append(f'Trial:{TrialId} w:{w}')
            best, minWords, minWordCount, nExp, pErr = mwUpdate(w,minRows, best, minWords,minWordCount)
    w,lo = best
    progressText.append(f'pErr:{pErr:.4f}, weight {w} CW found:{len(minWords)}')
    return w, lo, wDict, "; ".join(progressText)

def tesseractDecode(DEMH,s,params={},seed=0):
    '''apply tesseract Decoder with DEM matrix DEMH and syndrome s'''
    StimDEM = Stabs2StimDEM(DEMH)
    if params['decoder'] == 'tesseractLongbeam':
        defaultParams = {
            'pqlimit':1000000,
            'det_beam':20,
            'beam_climbing':True,
            'num_det_orders':21,
            'det_order':ts.utils.DetOrder.DetIndex,
            'no_revisit_dets':True
        }
    else:
        defaultParams = {
            'pqlimit':200000,
            'det_beam':15,
            'beam_climbing':True,
            'num_det_orders':16,
            'det_order':ts.utils.DetOrder.DetIndex,
            'no_revisit_dets':True
        }
    setDefaultParams(params, defaultParams)
    det_orders = ts.utils.build_det_orders(
        dem=StimDEM,
        num_det_orders=params['num_det_orders'],
        method=params['det_order'],
        seed=seed
    )
    config = ts.tesseract.TesseractConfig(
        dem=StimDEM,
        det_beam=params['det_beam'],
        beam_climbing=params['beam_climbing'],
        pqlimit=params['pqlimit'],
        no_revisit_dets=params['no_revisit_dets'],
        det_orders=det_orders,
    )
    decoder = ts.tesseract.TesseractDecoder(config)
    decoder.decode_to_errors(syndrome=np.array(s,dtype=bool))
    ## Tesseract jumbles the order of the errors - need to restore order!!
    nErr = len(DEMH.T)
    DEMDict = {tuple(bin2Set(DEMH[:,i])):i for i in range(nErr)}
    errMap = Z2MatZeros(nErr)
    ## decoder.errors are the errors and symptom.detector tells us which detectors they fire
    for i,myObj in enumerate(decoder.errors):
        errMap[i] = DEMDict[tuple(myObj.symptom.detectors)]
    return set2Bin(nErr,errMap[decoder.predicted_errors_buffer],dtype=np.int8)

def bposdDecode(DEMH,s,params={}):
    '''BP+OSD decoder'''
    paramDefaults = {
        'bp_method':"product_sum",
        'bp_max_iter':100,
        'bp_schedule':"parallel",
        'osd_method':"OSD_CS",
        'osd_order':1}
    params = setDefaultParams(params,paramDefaults)
    bpd = ldpc.BpOsdDecoder(
        DEMH,
        error_channel=params['priors'],
        bp_method=params['bp_method'],
        max_iter=params['bp_max_iter'],
        schedule=params['bp_schedule'],
        osd_method=params['osd_method'],
        osd_order=params['osd_order'],
    )
    # get correction
    return bpd.decode(s)

def bplsdDecode(DEMH,s,params={}):
    '''BP+LSD decoder'''
    paramDefaults = {'error_rate':0.01,
        'bp_method':"product_sum",
        'bp_max_iter':100,
        'bp_schedule':"parallel",
        'osd_method':"OSD_CS",
        'osd_order':1}
    params = setDefaultParams(params,paramDefaults)
    bpd = ldpc.bplsd_decoder.BpLsdDecoder(
        DEMH,
        error_rate=params['error_rate'],
        bp_method=params['bp_method'],
        max_iter=params['bp_max_iter'],
        schedule=params['bp_schedule'],
        osd_method=params['osd_method'],
        lsd_order=params['osd_order'],
    )
    # get correction
    return bpd.decode(s)

####################################################
## dist-qubitserf
####################################################

def dist_qubitserf(S,L=None,tB=1,params={}):
    '''Serban Cercelescu C Library https://github.com/qubitserfed/Qubitserf
    S: stabilizer generators
    tB: number of blocks - 2 for stabiliser code, 1 for classical code
    method: 1 for Brouwer-Zimmermann, 2 for meet in the middle
    threads: number of threads
    '''
    defaultParams={
        'method':'qubitserfMM',
        'maxTime' : 8 * 60 * 60,
        'nThreads' : 1
    }
    method = 1 if params['method'] == 'qubitserfBZ' else 2
    params = setDefaultParams(params,defaultParams)
    AddI = False
    if method==1 and tB==2 and (not isCSS(S)):
        ## method 1 only accepts CSS codes
        ## For non-CSS codes, use self-dual four-block encoding and add I to bottom of Sx/Sz
        ## all Pauli weights doubled
        AddI = True
        Sx = encode2block(S,AddI,4)
        S = CSS2twoBlock(Sx,Sx)
        S = getH(S,2)
    elif tB==1 and (L is not None) and (len(L) > 0):
        ## calculate full stabiliser group in 2-block form
        S = CSSSXLX2S(S,L)
        tB = 2
    myargs = ['qubitserf']
    # myargs.append('-v') ## doesn't seem to be effective
    if params['nThreads'] > 1:
        myargs.append('--threads')
        myargs.append(f"{params['nThreads']}") 
    if method == 1:
        myargs.append('--bz')
    S2 = pauli2str(S,tB)
    res, err = CLIRun(myargs,commands=S2,maxTime=params['maxTime'],captureErrors=True)
    if len(res) > 0:
        ## fix weird output for some cases
        res = int(res) % 100
        if AddI:
            res = res // 2
    else:
        res = 0
    return res,err

####################################################
## Brouwer-Zimmerman min distance algorithm
####################################################

def BZDistMW(H,L,tB=1,params={}):
    '''Brouwer-Zimmermann Distance Algorithm'''
    paramDefaults={
        'GF4blockRep':3
    }
    params = setDefaultParams(params,paramDefaults)
    if L is None:
        L = ZMatVstack(getLogicalPaulis(H)) if tB==2 else Z2MatZeros((0,len(H.T)))
    if tB == 1:
        H, L = CSSDual(H,L)
    k = len(L)
    progressText = []
    if tB==2:
        tB = params['GF4blockRep'] 
        H = encode2block(H,tB=tB)
        L = encode2block(L,tB=tB)
    ## check if all codewords are even weight - means that lower bounds will increase faster
    wMul = 1
    if tB == 1:
        t = codeEven(ZMatVstack([H,L]))
        wMul = 1 << t
    if tB in {3,4}:
        ## 3-block or 4-block reps are even codes 
        wMul = 2
        t = 1
    if wMul > 1:
        progressText.append(f'{t}-even code')
    r,n = H.shape
    SL = SLI(H,L)
    ## find information sets and construct bases for each of theses
    GammaList, kList = GammaOpt(SL,k=k,tB=tB)
    kList = np.array(kList,dtype=int)
    progressText.append(f'Constructing information sets: Block structure {kList}')
    ## update rank
    rk = max(kList)
    ## Upper bound - min weight of all matrices in GammaList
    vCount = 0
    tMax = rk
    dU = n
    dL = wMul
    t=1
    done = False
    mList = [0] * len(GammaList)
    x = np.zeros(n+k,dtype=np.int8)
    nB = n//2 if tB == 2 else n
    params = np.array([n,k,tB,nB,rk,t,dL,dU,vCount])
    while not done:
        progressText.append(f'Generating combinations of t={t} rows')
        params[5] = t
        ## generate linear combinations of t rows and update distance upper bound
        for i,Gi in enumerate(GammaList):
            update = combMat(Gi,params,x)
            if update:
                dU = params[7]
                progressText.append(f'Found w:{dU}')
                if dU <= dL:
                    progressText.append(f'Terminating: dU:{dU} <= dL:{dL},vCount:{params[-1]}')
                    done = True
                    break
            ## update distance lower bound
            mList[i] = max(0,t + 1 + kList[i] - rk)
            params[6] = adjustLB(np.sum(mList),wMul)
            progressText.append(f'Completed matrix {i},dU:{dU},dL:{dL},vCount:{params[-1]}')
            if dU <= dL:
                progressText.append(f'Terminating: dU:{dU} <= dL:{dL},vCount:{params[-1]}')
                done = True
                break
        ## calculate max number of rows to combine for BZ method
        tMaxNew = getTMax(kList,wMul,rk,dU,t)
        if tMaxNew != tMax:
            tMax = tMaxNew
            progressText.append(f'tMax:{tMax}')
            ## determine if we can exclude non-contributing information sets
            ix = [i for i in range(len(GammaList)) if kList[i] + tMax > rk ]
            ixDelta =  len(GammaList) - len(ix) 
            if ixDelta > 0:
                kList = kList[ix]
                GammaList = [GammaList[i] for i in ix]
                mList = [mList[i] for i in ix]
                progressText.append(f'Discarding {ixDelta} non-contributing matrix, new block structure {kList}')
        t += 1
        done = (t > tMax)
    progressText.append(f'Terminating: t:{t} > tMax:{tMax},vCount:{vCount}')
    if k > 0:
        x = x[:-k]
    x = decodeBlock(x,tB=tB)
    if tB in {3,4}:
        dU = dU//2
    return dU, x, "; ".join(progressText)

def GammaBasisGF2(M,ix):
    ## Calculate Information Sets of GF2 matrix M, with cols permuted by ix
    MList,kList = [],[]
    wCols = np.sum(M,axis=0)
    cols = np.array([c for c in ix if wCols[c] > 0],dtype=np.int64)
    rk = 0
    while len(cols) > 0:
        M,pivots = RREFZ2(M,cols,0)
        if rk == 0:
            rk = len(pivots)
            M = M[:rk]
        MList.append(M)
        kList.append(len(pivots)) 
        cols = np.array([j for j in cols if j not in set(pivots)],dtype=np.int64)
    return MList,kList

def GammaBasisGF4(M,ix):
    ## Calculate Information Sets over GF4: matrix M is 2-block representation, with cols permuted by ix
    MList,kList = [], []
    n2 = len(ix)//2
    r,n = M.shape
    ix = regroupPerm(ix,2)
    wCols = np.sum(M,axis=0)
    cols = ZMat([c for c in ix if wCols[c] > 0])
    while len(cols) > 0:
        M,pivots = RREFZ2(M,cols,0)
        pivots = np.mod(pivots,n2)
        extraRows = []
        for i in range(len(pivots)-1):
            if pivots[i] == pivots[i+1]:
                extraRows.append(M[i+1] ^ M[i])
        if len(extraRows) > 0:
            M1 = Z2MatZeros((r + len(extraRows),n))
            M1[:r,:] = M
            for i in range(len(extraRows)):
                M1[r+i,:] = extraRows[i]
        else:
            M1 = M
        cols = ZMat([c for c in cols if (c % n2) not in set(pivots)])
        kList.append(len(set(pivots)))
        MList.append(M1)
    return MList,kList

def GammaOpt(M,k=0,tB=1,maxTries=15):
    r,n = M.shape
    ix = np.arange(n-k)
    ## track best Gamma basis
    best = None
    ## try up to maxTries random permutations of ix
    for i in range(maxTries):
        if tB==2:
            MList,kList = GammaBasisGF4(M,ix)
        else:
            MList,kList = GammaBasisGF2(M,ix)
        kLen = len(set(kList))
        ord = (kLen,kList)
        if best is None or ord < best[0]:
            best = (ord, MList)
        ix = np.random.permutation(ix)
    (kLen,kList), MList = best
    return  MList, kList

@nb.njit (nb.int64(nb.int64,nb.int64))
def adjustLB(dL,wMul):
    '''calculate LB
    mList: weights of unenumerated codewords within each info set
    wMul: for even, doubly or triply even codes weights of CW are multiple of wMul'''
    dMod = dL % wMul
    if dMod > 0:
        dL += (wMul - dMod)
    return dL

@nb.njit (nb.int64(nb.int64[:],nb.int64,nb.int64,nb.int64))
def getdL(kList,wMul,r,t):
    '''calculate lower bound for BZ method'''
    dL = 0
    for kj in kList:
        if r - kj <= t:
            dL += (t + kj - r)
    return adjustLB(dL,wMul)

@nb.njit (nb.int64(nb.int64[:],nb.int64,nb.int64,nb.int64,nb.int64))
def getTMax(kList,wMul,r,dU,t):
    '''calculate maximum number of rows w0 that we need to combine for BZ method'''
    dL = 1
    while t < r:
        t+=1
        dL = getdL(kList,wMul,r,t)
        if dL > dU:
            return t-1
    return r

@nb.njit (nb.int64(nb.int8[:],nb.int64,nb.int64,nb.int64))
def vecWt(x,n,k,tB):
    if k > 0:
        if np.sum(x[-k:]) == 0:
            return 0
    if tB == 2:
        return np.sum(x[:n] | x[n:2*n])
    else:
        return np.sum(x[:n])

@nb.njit (nb.bool(nb.int64[:],nb.int8[:],nb.int8[:]))
def combUpdate(params,x,y):
    ## n,k,tB,n2,r,t,LB,UB,vCount = params
    ## 0 1  2  3 4 5  6  7    8
    ## update vCount
    params[-1] += 1
    w = vecWt(y,params[3],params[1],params[2])
    if w > 0 and (params[7] == 0 or w < params[7]):
        ## update UB
        params[7] = w
        ## update x
        x[:] = y[:]
        return True
    return False
    
@nb.njit (nb.int64(nb.int64[:],nb.int64[:]))
def CombWalsh(g,supp):
    '''loopless Liu-Tang from Generating Gray Codes  in O(1) Worst-Case Time per Word'''
    while True:
        Rise, i, c, d, m, n = supp
        if Rise:
            if i==n:
                x = m
            else:
                x = g[i+1] - 1
            if g[i] < x:
                # delta = {g[i],g[i] + 1}
                supp[2], supp[3] = g[i] + 1, g[i]
                g[i] = g[i] + 1
                if i > 1:
                    # delta = {g[i-1],g[i]}
                    # supp[2], supp[3] = g[i], g[i-1]
                    supp[3] = g[i-1]
                    g[i-1] = g[i] - 1
                    if i == 2:
                        # i = 1
                        supp[1] = 1
                        # Rise = False
                        supp[0] = 0
                    else:
                        # i = i - 2
                        supp[1] -= 2
                return 1  
        else:
            if i > n:
                return 0
            if g[i] > i:
                # delta = {g[i],g[i] - 1}
                supp[2], supp[3] = g[i], g[i] - 1
                g[i] = g[i] - 1
                if i > 1:
                    # delta = {i-1,g[i]+1}
                    # supp[2], supp[3] = g[i]+1, i-1
                    supp[3] = i-1
                    g[i - 1] = i - 1
                    # i = i-1
                    supp[1] -=  1
                    # Rise = True
                    supp[0] = 1
                return 1
        # i = i + 1
        supp[1] += 1
        # Rise = not Rise
        supp[0] = 1 - supp[0] 

@nb.njit (nb.bool(nb.int8[:,:],nb.int64[:],nb.int8[:]))
def combMat2(A,params,x):
    '''combinations of 2 rows of A + update A2'''
    ## n,k,tB,n2,r,t,LB,UB,vCount = params
    ## 0 1  2  3 4 5  6  7    8
    update = False
    r = params[4]
    for i in range(r-1):
        for j in range(i+1,r):
            y = A[i] ^ A[j]
            update |= combUpdate(params,x,y)
    return update

@nb.njit (nb.bool(nb.int8[:,:],nb.int64[:],nb.int8[:]))
def combMat1(A,params,x):
    '''combinations of single rows of A'''
    ## n,k,tB,n2,r,t,LB,UB,vCount = params
    ## 0 1  2  3 4 5  6  7    8
    update = False
    r = params[4]
    for i in range(r):
        update |= combUpdate(params,x,A[i])
    return update

@nb.njit (nb.bool(nb.int8[:,:],nb.int64[:],nb.int8[:]))
def combMatT(A,params,x):
    '''combinations of more than 2 rows of A'''
    ## n,k,tB,n2,r,t,LB,UB,vCount = params
    ## 0 1  2  3 4 5  6  7    8
    update = False
    n,k,r,s = params[0],params[1],params[4],params[5]
    supp = np.array([1,s,0,0,r,s],dtype=np.int64)
    y = np.zeros(n+k,dtype=np.int8)
    g = np.arange(s+1)
    for i in g[1:]:
        y ^= A[i-1]
    res = 1
    while res != 0:
        res = CombWalsh(g,supp)
        if res == 1:
            u, v = supp[2]-1,supp[3]-1
            y ^= (A[u] ^ A[v])
            update |= combUpdate(params,x,y)
    return update

@nb.njit (nb.bool(nb.int8[:,:],nb.int64[:],nb.int8[:]))
def combMat(A,params,x):
    '''Min weight of linear combinations of x rows of A
    Return True if new UB on weight encountered'''
    ## n,k,tB,n2,r,t,LB,UB,vCount = params
    ## 0 1  2  3 4 5  6  7    8
    params[4] = len(A)
    t = params[5]
    if t > params[4]:
        return False
    elif t == 1:
        return combMat1(A,params,x)
    elif t == 2:
        return combMat2(A,params,x)
    else:
        return combMatT(A,params,x)

@nb.njit (nb.int8[:,:](nb.int8[:,:],nb.int8[:,:]))
def SLI(H,L):
    '''Stack H and L and append I to right of L'''
    k = len(L)
    if k == 0:
        return H
    r,n = H.shape
    HL = np.zeros((r+k,n+k),dtype=nb.int8)
    HL[:r,:n] = H
    HL[r:,:n] = L
    for i in range(k):
        HL[r+i,n+i] = 1
    return HL

def codeEven(A,tB=1,tMax=3):
    '''check if code is even, doubly-even, triply even etc - tMax=3 indicates we find up to triply even'''
    rowIndices = range(len(A))
    t = 0
    while t < tMax:
        ## we will calculate GCD of row weight with N, which reduces with increasing t
        N = 1 << (tMax - t)
        ## track min GCD with N
        minGCD = N
        ## product of t+1 rows of A
        for ix in iter.combinations(rowIndices,t+1):
            ## check GCD of row weight with N
            w = rowWeight(np.prod(A[ix,:],axis=0),tB=tB)
            wGCD = np.gcd(N,w)
            ## if wGCD is 1, we are done and don't need to look at further combinations
            if wGCD == 1:
                minGCD = 1
                break
            ## otherwise, if GCD nonzero, check if we have a new minimum
            elif wGCD > 1 and wGCD < minGCD:
                minGCD = wGCD
        ## revise tMax = t + log base 2 of minGCD
        tMax = t + logCeil(minGCD)
        ## increment t
        t += 1
    return t - 1