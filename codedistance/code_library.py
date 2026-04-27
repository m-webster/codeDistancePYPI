from .common import *
from .NHow import *
from .distance import *
from .complex_utils import *

def CSS2Dict(Hx,Hz,name="",d=0):
    LX, LZ = CSSLXLZ(Hx,Hz)
    # SX,LX,SZ,LZ = CSSCode(SX=Hx,SZ=Hz)
    k,n = LX.shape
    if len(name) == 0:
        name = f'[[{n},{k},{d}]]'
    return {
        "name":name,
        'type':'CSS',
        'n':n,
        'k':k,
        'd':d,
        'SX':Hx,
        'LX':LX,
        'SZ':Hz,
        'LZ':LZ
        }

def codeTables2Dict(params,S,codeType):
    n,k,d = params
    S = Z2Mat(S)
    G = None
    if codeType == 'GF2':
        G = S
        S = KerZ2(S,np.arange(n),0)
    return {
        'name': f'[[{n},{k},{d}]]',
        'type':codeType,
        'n':n,
        'k':k,
        'd':d,
        'S':S,
        'L':None,
        'G':G
    }

def qc2Dict(qc,name=""):
    
    return {
        'name': name,
        'type':'circuit',
        'n':qc.num_qubits,
        # 'n':n,
        'k':qc.num_observables,
        'd':0,
        'qc':qc
    }

##########################################
## Logical Paulis of Stabiliser Code
##########################################

def CSSSXLX2S(SX,LX=None):
    '''convert CSS code given in terms checks and logicals of X type to 2-block check matrix'''
    # pList,nA,tB = [0.9,0.1],0,1
    # SX, probs = indepL(SX,pList,nA,tB)
    rx, n = SX.shape
    # R,V = HowRes(SX,LX,2)
    # print(func_name(),'SX/LX independence check',np.sum(R,axis=-1))
    SXLX = SX if LX is None or len(LX) == 0 else ZMatVstack([SX,LX])
    SZ = KerZ2(SXLX,np.arange(n),0)
    rz, n = SZ.shape
    S = ZMatZeros((rx+rz,2*n ))
    S[:rx,:n] = SX
    S[rx:,n:] = SZ
    # k = len(LX)
    # print(rx,rz,k,n-rx-rz-k)
    # codeDict = CSS2Dict(SX,SZ)
    # Lx,Lz = getLogicalPaulis(S)
    # print(f'getLogicalPaulis: {len(Lx)}, {len(Lz)}')
    # print(f"CSS2Dict: [{codeDict['n']},{codeDict['k']},{codeDict['d']}]")
    return S

def getLogicalPaulis(S):
    '''Calculate LX,LZ from stabiliser matrix S'''
    ## get tableau
    n,k,T = Stab2Tableau(S)
    ## r is the number of independent stabilisers
    r = n - k
    ## get rows corresponding to LX and LZ
    LX = T[r:n]
    LZ = T[n+r:]
    return LX,LZ

def Stab2Tableau(S):
    '''Return n,k tableau plus phases for stabilisers in binary form S
    Increased efficiency version for large codes'''
    S = Z2Mat(S)
    n = len(S.T) // 2
    ## RREF mod 2 - only consider first n columns, return pivots
    H, rCols = RREFZ2(S,np.arange(n),r0=0)
    ## independent X checks
    rCols = ZMat(rCols)
    r = len(rCols)
    kCols = ZMat(invRange(n,rCols)) + n
    H, sCols = RREFZ2(H,kCols,r0=r)
    ## independent Z checks 
    sCols = ZMat(sCols) - n
    s = len(sCols) 
    ## number of encoded qubits
    k = n - r - s
    ## remaining cols
    kCols = ZMat(invRange(n,np.hstack([rCols, sCols])))
    ## Extract key matrices
    Ark = H[:r,kCols]
    Crk = H[:r,kCols+n]
    Esk = H[r:r+s,kCols+n]
    T = np.zeros((2*n,2*n),dtype=np.int8)
    T[:r+s] = H[:r+s]
    ## Logical X and Z
    T[n-k:n,sCols] = Esk.T
    T[n-k:n,rCols + n] = Crk.T
    ## LZ
    T[2*n-k:2*n,rCols+n] = Ark.T
    for i in range(k):
        T[n-k + i,kCols[i]] = 1
        T[2*n-k+i,kCols[i] + n] = 1
    ## Form destabilisers
    for i in range(r):
        T[n+i,rCols[i]+n] = 1
    for i in range(s):
        T[n+r+i,sCols[i]] = 1
    return n,k,T

##########################################
## CSS Codes
##########################################

def CSSCode(SX,SZ):
    """Create CSS code from various input types.
    SX -- X-checks 
    SZ -- Z-checks 
    SimplifyGens -- optionally remove linearly dependent X and Z checks

    Can take either text or arrays as input.
    Returns SX,LX,SZ,LZ. 
    """
    SX = bin2ZMat(SX)
    SZ = bin2ZMat(SZ)

    ## find n - number of qubits
    n = np.max([0 if A is None else A.shape[1] for A in [SX,SZ]])
    ## ensure stabilisers have the same row length
    SX = ZMatZeros((0,n)) if SX is None else ZMat(SX,n)
    SZ = ZMatZeros((0,n)) if SZ is None else ZMat(SZ,n)
    # if simplifyGens:
    #     SX = indepLZ(None,SX)
    #     SZ = indepLZ(None,SZ)
    ## Make stabilisers into 2n length binary vectors
    SX2n = ZMatHstack([SX,ZMatZeros(SX.shape)])
    SZ2n = ZMatHstack([ZMatZeros(SZ.shape),SZ])
    S2n = ZMatVstack([SX2n,SZ2n])
    ## Get logical Paulis
    LX,LZ = getLogicalPaulis(S2n)
    ## Convert logicals from length 2n to length n binary vectors
    LX = LX[:,:n]
    LZ = LZ[:,n:]
    return SX,LX,SZ,LZ

def CSSCheck(Hx,Hz,LX,LZ):
    k = len(LX)
    SXLX = np.vstack([LX,Hx])
    SZLZ = np.vstack([LZ,Hz])
    C1 = matMul(SXLX,SZLZ.T,2)
    for i in range(k):
        C1[i,i] ^= 1
    return (np.sum(C1)==0)

def CSSLXLZ(Hx,Hz):
    '''Fast method for calculationg LX/LZ for CSS codes'''
    Hx = Z2Mat(Hx)
    Hz = Z2Mat(Hz)
    n = len(Hx.T)
    ## RREF mod 2 - only consider first n columns, return pivots
    Hx, rCols = RREFZ2(Hx,np.arange(n),r0=0)
    ## independent X checks
    # rCols = ZMat(rCols)
    r = len(rCols)
    kCols = ZMat(invRange(n,rCols))
    Hz, sCols = RREFZ2(Hz,kCols,r0=0)
    ## independent Z checks 
    # sCols = ZMat(sCols)
    s = len(sCols) 
    ## number of encoded qubits
    k = n - r - s
    LX = np.zeros((k,n),dtype=np.int8)
    LZ = np.zeros((k,n),dtype=np.int8)
    if k == 0:
        # print(func_name(),'Trivial codespace',n,r,s)
        return LX,LZ
    ## remaining cols
    kCols = ZMat(invRange(n,np.hstack([rCols, sCols])))
    ## Extract key matrices
    Ark = Hx[:r,kCols]
    # Crk = H[:r,kCols+n]
    Esk = Hz[:s,kCols]
    
    ## Logical X and Z
    LX[:,sCols] = Esk.T
    ## LZ
    LZ[:,rCols] = Ark.T
    for i in range(k):
        LX[i,kCols[i]] = 1
        LZ[i,kCols[i]] = 1
    
    return LX,LZ


# #####################################################
# ## Non-CSS codes
# #####################################################

def codeTableCode(n,k,codeDir):
    '''Get code from HTML page of codetables.de'''
    myfile = f'{n}-{k}.txt'
    f = open(f'{codeDir}/{myfile}','r')
    mytext = f.read()
    f.close()
    mytext = mytext.split('Construction of a [[')[1]
    mytext = mytext.split(']] quantum code:\n')
    params = mytext[0].split(",")
    mytext = mytext[1].split('stabilizer matrix:\n')
    if len(mytext) > 1:
        mystr = mytext[1].split('\n\n')[0] 
        S = codeTable2ZMat(mystr)
        L = ZMatVstack(getLogicalPaulis(S))
    else:
        S,L = ZMatZeros((0,0)),ZMatZeros((0,0))
    return [int(p)  for p in params], S, L

def codeTable2ZMat(mystr):
    # mystr = mystr.replace('</PRE>',"")
    # print(func_name(),mystr)
    '''get check matrix from string rep from codetables.de etc'''
    mystr = mystr.replace(" ","").replace("[","").replace("]","").replace("|","")
    S = bin2ZMat(mystr.strip())
    return S

def CodetablesImport(myFile):
    '''read series of codes in a file myFile'''
    f = open(myFile)
    mystr = f.read()
    mystr = mystr.split("\n\n")
    codeList = []
    for codeText in mystr:
        codeText = codeText.split("\n")
        params = codeText.pop(0).split(",")
        if len(params) == 3:
            params = [int(p) for p in params]
            HStr = "\n".join(codeText).strip()
            if len(HStr) > 4:
                S = bin2ZMat(HStr)
                codeList.append((params,S))
    return codeList

def CodetablesExport(myFile,params,S):
    myText = [",".join(map(str,params))]
    myText.append(ZMatPrint(S))
    myText = "\n".join(myText)
    with open(myFile,'a') as f:
        f.write(myText)
        f.write("\n\n")

def CodetableExportList(paramFile,sourceFiles,outFile):
    paramList = set()
    with open(paramFile,'r') as f:
        myText = f.read()
        myText = myText.replace("[","").replace("]","")
        myText = myText.split("\n")
        for params in myText:
            params = params.strip().split(",")
            if len(params) == 3:
                paramList.add(tuple(map(int,params)))
    for myFile in sourceFiles:
        sourceList = CodetablesImport(myFile)
        for params,S in sourceList:
            if tuple(params) in paramList:
                CodetablesExport(outFile,params,S)


#####################################
## Symmetric Hypergraph Product Code
#####################################

def SHPC(T):
    '''Make symmetric hypergraph product code from T.
    T can either be a string or np array.
    Returns SX, SZ.'''
    T = bin2ZMat(T)
    H = matMul(T.T, T,2)
    return HPC(H,H)

def HPC(A,B):
    '''Make hypergraph product code from clasical codes A, B
    A and B can either be a string or np array.
    Returns SX, SZ.'''
    A = bin2ZMat(A)
    B = bin2ZMat(B)
    ma,na = np.shape(A)
    mb,nb = np.shape(B)
    ## Generate SX
    C = np.kron(A,ZMatI(mb))
    D = np.kron(ZMatI(ma),B)
    SX = np.hstack([C,D])
    ## Generate SZ
    C = np.kron(ZMatI(na) ,B.T)
    D = np.kron(A.T,ZMatI(nb))
    SZ = np.hstack([C,D])
    return SX, SZ

########################
# repetition code
########################
def repCode(r,closed=True):
    '''Generate classical repetition code on r bits.
    If closed, include one dependent row closing the loop.'''
    s = r if closed else r-1 

    SX = ZMatZeros((s,r))
    for i in range(s):
        SX[i,i] = 1
        SX[i,(i+1)%r] = 1
    return SX

## build 2D toric code from repetition code and SHPC constr
def toric2D(r):
    '''Generate distance r 2D toric code using SHCP construction.
    Returns SX, SZ.'''
    A = repCode(r,closed=False)
    return SHPC(A)

##########################
# Bivariate Bicycle Codes
##########################

def SMatrix(n,p=1):
    S = np.eye(n,dtype=np.int8)
    if p == 0:
        return S
    return np.roll(S,p,axis=-1)

# def matPower(A,p):
#     return np.linalg.matrix_power(A,p)

# def BivariateBicycle(l,m,Apoly,Bpoly):
#     Sl = SMatrix(l)
#     Sm = SMatrix(m)
#     x = np.kron(Sl,ZMatI(m))
#     y = np.kron(ZMatI(l),Sm)
#     MList = [matPower(m,p) for (m,p) in zip([x,y,y],Apoly) ]
#     A = np.mod(np.sum(MList,axis=0),2)
#     MList = [matPower(m,p) for (m,p) in zip([y,x,x],Bpoly) ]
#     B = np.mod(np.sum(MList,axis=0),2)
#     SX = np.hstack([A,B])
#     SZ = np.hstack([B.T,A.T])
#     Hx = getH(SX,2)
#     Hz = getH(SZ,2)
#     r,n = Hx.shape
#     s,n = Hz.shape
#     k = n-r-s
#     print(f'n={n} r={r} s={s} k={k}')
#     return SX,SZ

# def matKronPowX(l,m,p):
#     return np.kron(SMatrix(l,p),ZMatI(m))

# def matKronPowY(l,m,p):
#     return np.kron(ZMatI(l),SMatrix(m,p))

# def BivariateBicycle(l,m,Apoly,Bpoly):
#     A = matKronPowX(l,m,Apoly[0]) ^ matKronPowY(l,m,Apoly[1]) ^ matKronPowY(l,m,Apoly[2])
#     B = matKronPowY(l,m,Bpoly[0]) ^ matKronPowX(l,m,Bpoly[1]) ^ matKronPowX(l,m,Bpoly[2])
#     SX = ZMatHstack([A,B])
#     SZ = ZMatHstack([B.T,A.T])
#     return SX,SZ

def BBIBM(l,m,Apoly,Bpoly):
    '''Bivariate Bicycle Code - IBM format'''
    uVec = [Apoly[0],0,0,0,Bpoly[1],Bpoly[2]]
    vVec = [0,Apoly[1],Apoly[2],Bpoly[0],0,0]
    return BBCSSCode(l,m,uVec,vVec,Alen=3)


def matListXOR(AList):
    '''Add matrices in AList together mod 2 - matrices need to be the same shape'''
    A = None
    for M in AList:
        if A is None:
            A = M.copy()
        else:
            A ^= M
    return A

def BBCSSCode(l,m,uVec,vVec,Alen=None):

    '''Bivariate Bicycle Code:
    Allows for construction of both IBM and BB5 codes from Quantum error correction for long chains of trapped ions
    l: length of first cycle
    m: length of second cycle
    uVec: powers to raise first cycle to mod l
    vVec: powers to raise second cycle to mod m
    Alen: number of matrices to add together for A matrix - remainder are added together for B matrix
    '''
    if Alen is None:
        Alen = len(uVec)//2
    ## construct A and B matrices
    MatList = [np.kron(SMatrix(l,u),SMatrix(m,v)) for (u,v) in zip(uVec,vVec)]
    A = matListXOR(MatList[:Alen])
    B = matListXOR(MatList[Alen:])
    ## Hx/Hz dimensions
    r = l * m
    n = 2 * r
    ## Hx
    Hx = np.empty((r,n),dtype=np.int8)
    Hx[:,:r] = A 
    Hx[:,r:] = B
    ## Hz
    Hz = np.empty((r,n),dtype=np.int8)
    Hz[:,:r] =  B.T
    Hz[:,r:] =  A.T
    return Hx,Hz

##################################
# Convert Cell complexes to Codes
##################################

def complex2ColourCode(C):
    '''Make a colour code from Complex
    Qubits: vertices
    SZ: 2-faces
    SX: D-faces'''
    ## express complex in terms of adjacecy matrices wrt 0-cells (vertices)
    C1 = complexCProduct(C[1:])
    ## 2D Faces
    SZ = C1[1]
    ## highest dim cells
    SX = C1[-1]
    return SX, SZ

def complex2SurfaceCode(C):
    '''Make a surface code from Complex
    Qubits: edges
    SZ: plaquettes
    SX: vertices'''
    ## express complex in terms of adjacecy matrices wrt 1-cells (edges)
    ## vertex operators
    SX = C[1]
    ## plaquette operators
    SZ = C[2].T
    return SX, SZ

def AdjListVMax(AdjList):
    return max(len(AdjList),np.max(np.array(AdjList).flatten()))+1

def AdjList2DRE(AdjList):
    n = AdjListVMax(AdjList)
    temp = []
    temp.append(f"n={n} g")
    gRows = []
    for myRow in AdjList:
        gRows.append(" ".join(map(str,myRow)))
    temp.append(";\n".join(gRows) + ".")
    return "\n".join(temp)

def mat2AdjList(A):
    '''m+n vertices - adjacency is row to col'''
    m,n = A.shape
    return [[m + ZMat(bin2Set(a))] for a in A]

##################################
## Dreadnaut automorphisms
##################################

    
def runDRE(AdjList,options,outputs):
    myText = [AdjList2DRE(AdjList)]
    for c in options:
        myText.append(c)
    myText.append("x")
    for c in outputs:
        myText.append(c)
    myText.append("q")
    myText = "\n".join(myText)
    proclist = ['dreadnaut']
    p = subprocess.run(proclist, input=myText, capture_output=True, text=True)
    return p.stdout

def getCertDRE(AdjList):
    CL = 'Canonical Labelling:'
    GC = 'Graph Certificate:'
    ## Call dreadnaut - canonical labelling, no automorphisms; 
    myText = runDRE(AdjList,options=['c','-a'],outputs=[f'"{CL}\n"','b',f'"{GC}\n"','z'])
    # print(myText)
    # split by CL
    myText = myText.split(CL)[1]
    # splict by GC
    CL,GC = myText.split(GC)
    ## split by newline, strip whitespace
    CL = CL.strip().split("\n")
    CL = [a.strip() for a in CL]
    ix = []
    i = 0
    ## join lines which don't terminate in ; (lines ending in ; are the graph adjacencies)
    while i < len(CL) and CL[i][-1] != ";":
        ix.append(CL[i])
        i+=1
    ix = " ".join(ix).split(" ")
    ## turn into list of integer
    ix = list(map(int,ix))
    GC = GC.strip()[1:-1].replace(" ","")
    return ix, GC


def str2perm(s,n):
    print(s,len(s))
    s = s.replace("(","")
    ix = np.arange(n)
    for c in s.split(")"):
        if len(c) > 0:
            c = list(map(int,c.split(" ")))
            lc = len(c)
            if lc > 1:
                ixc1 = ix[c[-1]]
                for i in range(lc):
                    if i == lc - 1:
                        ix[c[0]] = ixc1
                    else:
                        ix[c[i+1]] = ix[c[i]]
    return ix

def getAutsDRE(AdjList):
    '''Return generators of aut group plus orbits'''
    ORB = 'Orbits:'
    myText = runDRE(AdjList,options=[],outputs=[f'"{ORB}\n"','o'])
    # print(myText)
    AUT, ORB = myText.split(ORB)
    AUT = [a.strip() for a in AUT.split("\n")]
    AList = []
    n = AdjListVMax(AdjList)
    for a in AUT:
        if len(a) > 0 and a[0] == "(":
            AList.append(str2perm(a,n))
    ORB =  " ".join([a.strip() for a in ORB.split("\n")])
    OList = []
    for a in ORB.split("; "):
        a = a.strip().split(" (")[0]
        if len(a) > 0:
            temp = []
            for b in a.split(" "):
                # print(b.split(":"))
                b = list(map(int,b.split(":")))
                if len(b) == 1:
                    temp.append(b[0])
                else:
                    for c in range(b[0],b[1] + 1):
                        temp.append(c)
            OList.append(temp)
    return (AList, OList)