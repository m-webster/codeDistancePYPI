from codedistance import *
import numpy as np
import itertools as iter

################################################################################################
## Generate BB codes with various stabiliser weights, up to equivalence classes
################################################################################################

def BBCodeList(myFile = 'BB6/BB6Gen.txt'):
    '''Extract BB codes from myFile into a list of Dictionaries'''
    codeList = []
    with open(myFile,'r') as f:
        myText = f.read().split('\n')
        for myRow in myText[1:]:
            myRow = myRow.strip().split("\t")
            if len(myRow) > 6:
                myRow = list(map(int,myRow[:-1]))
                n,k,dZ,dX,l,m = myRow[:6]
                uv = myRow[6:]
                c = len(uv)//2
                uVec = uv[:c]
                vVec = uv[c:]
                Hx, Hz = BBCSSCode(l, m, uVec, vVec)
                codeName = f'BB{c}_[{n},{k}]:l={l},m={m},uVec={uVec},vVec={vVec}'
                myCode = CSS2Dict(Hx, Hz, name=codeName, d=0)
                if myCode['k'] > 0:
                    myCode['name'] = codeName
                    codeList.append(myCode)
    return codeList

def freqZeroCond(u):
    '''verify if 0 occurs more than any other integer in u'''
    if len(u) == 0: 
        ## trivially satisfied
        return True
    wDict = freqTable(u)
    if 0 not in wDict:
        ## 0 does not occur in u
        return False
    freq0 = wDict[0]
    ## 0 takes up at least half the vector
    if freq0 >= (len(u)+1)//2:
        return True
    ## remove 0 from dictionary and find max frequency
    del wDict[0]
    fMax = max(wDict.values())
    return freq0 >= fMax

partList = {}

def gcdPartition(m):
    '''partition the range [0..m-1] into equivalence classes by gcd(m)'''
    global partList
    if m in partList:
        return partList[m]
    gDict = {}
    for i in range((m+1)//2):
        ## add i and m-i to the list in the same iteration
        j = (m - i) % m
        g = np.gcd(m,i)
        if g not in gDict:
            gDict[g] = []
        gDict[g].append(i)
        if i != j:
            gDict[g].append(j)
    partList[m] = gDict
    return gDict

def gcdOrder(u,m):
    '''for each element of u, check that the smallest element of the GCD equivalence class has the highest frequency'''
    '''Check why this is valid...'''
    gDict = gcdPartition(m)
    wDict = freqTable(u)
    for g, gList in gDict.items():
        if len(gList) > 1:
            temp = []
            for i in gList:
                temp.append(wDict[i] if i in wDict else 0)
            ## ensure that the smallest element of the gcd equiv class has the highest frequency
            wMax = max(temp)
            if temp[0] < wMax:
                return False
    return True

def vec2cycle(l,uVec):
    '''convert uVec with cycle size l to a matrix'''
    c1 = len(uVec) // 2
    ## generate permutation matrices corresponding to uVec
    MatList = [SMatrix(l,u) for u in uVec]
    ## add the first c1 together to form A
    A = matListXOR(MatList[:c1])
    ## add the remaining ones togther to form B
    B = matListXOR(MatList[c1:])
    ## return [A|B]
    return np.hstack([A,B],dtype=np.int8)


def BBCycles(l,c):
    ''''return all cycles of length c and cycle size l'''
    ## file to either check or create
    myDir = f'BB{c}'
    if not os.path.isdir(myDir):
        ## create directory if it doesn't already exist
        os.mkdir(myDir)
    outfile = f'{myDir}/Cycles{c},{l}.txt'
    if not os.path.isfile(outfile):
        ## generate cycle file if it doesn't already exist
        GenBBCycles(l,c)
    with open(outfile,'r') as f:
        ## ready cycles from file
        temp = []
        myRows = f.read() 
        for myRow in myRows.splitlines()[1:]:
            myRow = myRow.split("\t")
            if len(myRow) == c + 4:
                myRow = list(map(int,myRow[3:-1]))
                temp.append(myRow)
    return temp


def GenBBCycles(l,c):
    '''Generate all possible cycles up to column permutations
    These are l x l matrices which are the addition of c different permutation matrices
    Cycles are non-trivial if there are dependent rows'''
    print(f'GenBBCycles: {l}, {c}')
    c2 = c//2
    c1 = c-c2
    outfile = f'BB{c}/Cycles{c},{l}.txt'
    with open(outfile,'a') as f:
        f.write(f'n\tk\tl\tuVec\tXCert\n')
    certList = set()
    for u1 in iter.combinations_with_replacement(range(l),c1-1):
        u1 = (0,) + u1
        ## constrain u1 to vectors where 0 occurs more than any other value and lowest in gcd class has highest frequency
        if freqZeroCond(u1) and gcdOrder(u1,l):
            for u2 in iter.combinations_with_replacement(range(l),c2-1):
                u2 = (0,) + u2
                ## constrain u2 to vectors where 0 occurs more than any other value
                if freqZeroCond(u2):
                    uVec =  u1 + u2
                    ## constrain combined u1 + u2 to vectors where lowest in gcd class has highest frequency
                    if gcdOrder(uVec,l):
                        ## matrix representataion of the cycle
                        Hx  = vec2cycle(l,uVec)
                        r,n = Hx.shape
                        ## check equivalence class of cycle up to col permutations
                        ix,XCert = getCertDRE(mat2AdjList(Hx))
                        if XCert not in certList:
                            certList.add(XCert)
                            ## check if the cycle is trivial
                            Hx,pivots = RREFZ2(Hx,np.arange(n),0)
                            k = r - len(pivots) 
                            if k > 0:
                                ## output to file if non-trivial and not previously seen
                                with open(outfile,'a') as f:
                                    f.write(f'{n}\t{k}\t{l}\t{"\t".join(map(str,uVec))}\t{XCert}\n')

def uvProcess(l, m, uVec, vVec,c2,certList,outfile):
    '''Process the code specified by l,m,uVec,vVec'''
    ## make a BB code
    Hx, Hz = BBCSSCode(l, m, uVec, vVec,c2)
    ## check equivalence class of code up to col permutations
    ix,XCert = getCertDRE(mat2AdjList(Hx))
    if XCert not in certList:
        certList.add(XCert)
        myCode = CSS2Dict(Hx, Hz, name=f"BB", d=0)
        r,n = Hx.shape
        dX,dZ = 0,0
        if myCode['k'] > 0:
            ## if non-trivial, get code distance
            lo,progress = dist_m4ri_CSS(myCode['SX'],myCode['LX'])
            dZ = len(lo)
            lo,progress = dist_m4ri_CSS(myCode['SZ'],myCode['LZ'])
            dX = len(lo)
        ## write to file
        with open(outfile,'a') as f:
            f.write(f"{myCode['n']}\t{myCode['k']}\t{dZ}\t{dX}\t{l}\t{m}\t{'\t'.join(map(str,uVec))}\t{'\t'.join(map(str,vVec))}\t{XCert}\n")


def BBSample(l,m,c,vCycles,outfile,sampleSize):
    '''Sample BB codes with stabiliser weight c, cycle size l and m'''
    c2 = c//2
    c1 = c - c2
    certList = set()
    for i in range(sampleSize):
        ## choose a random u vector with entries mod l
        u1 = np.random.randint(l,size=c1-1)
        u2 = np.random.randint(l,size=c2-1)
        ## improve by sampling from equivalance classes - eg max zero count, GCD cond
        uVec = [0] + list(u1) + [0] + list(u2)
        ## choose a random m-cycle
        ix = np.random.randint(len(vCycles))
        vVec = vCycles[ix]
        ## construct and save the code
        uvProcess(l, m, uVec, vVec,c2,certList,outfile)

def BBEnumerate(l,m,c,vCycles,outfile):
    '''Enumerate all BB codes with stabiliser weight c, cycle size l and m'''
    c2 = c//2
    c1 = c - c2
    certList = set()
    ## iterate through all u vectors with entries mod l
    for u1 in iter.product(range(l),repeat=c1-1):
        u1 = (0,) + u1
        if freqZeroCond(u1) and gcdOrder(u1,l):
            for u2 in iter.product(range(l),repeat=c2-1):
                u2 = (0,) + u2
                if freqZeroCond(u2):
                    uVec =  u1 + u2
                    if gcdOrder(uVec,l):
                        ## iterate through all m-cycles
                        for vVec in vCycles:
                            uvProcess(l, m, uVec, vVec,c2,certList,outfile)

def BBGen(l,m,c,sampleSize = 100):
    '''Generate BB codes with stabiliser weight c, cycle size l and m
    Enumerate all if the number of possibilities is small, otherwise take a sample'''
    myDir = f'BB{c}'
    if not os.path.isdir(myDir):
        ## create directory if it doesn't already exist
        os.mkdir(myDir)
    outfile = f'{myDir}/BB{c}Gen.txt'
    if not os.path.isfile(outfile):
        with open(outfile,'w') as f:
            f.write(f'n\tk\tdZ\tdX\tl\tm\tuVec{"\t" * (c-1)}\tvVec{"\t" * (c-1)}\tXCert\n')
    ## use cycles of larger number if possible
    if m < l:
        l,m = m,l
    vCycles = BBCycles(m,c)
    ## if there are no non-trivial cycles, swap l and m
    if len(vCycles) == 0:
        l,m = m,l
        vCycles = BBCycles(m,c)
    ## if there are no non-trivial cycles for either l or m, there are no codes with k > 0
    if len(vCycles) > 0:
        totalSize = len(vCycles) * (l ** c)
        if sampleSize > totalSize:
            BBSample(l,m,c,vCycles,outfile,sampleSize) 
        else:
            BBEnumerate(l,m,c,vCycles,outfile)


def BBAnalyse(infile):
    '''Analyse the codes generated by BBGen and saved to file'''
    codeDict = {}
    with open(infile,'r') as f:
        myText = f.read()
        for myRow in myText.splitlines()[1:-1]:
            myRow = myRow.split("\t")
            if len(myRow) > 3:
                n,k,dZ,dX,l,m  = list(map(int,myRow[:6]))
                d = min(dX,dZ)
                ## how much over the BPT bound are we?
                myRow.append(f'{k*d*d/n:.2f}')
                ## using ix identifies codes of form [[n*g, k*g, d]]
                if d > 1 and k > 0:
                    g = np.gcd(d,k)
                    ix = f'{n//g},{k//g},{d}'
                if (ix not in codeDict) or (int(codeDict[ix][0]) > n):
                    codeDict[ix] = myRow
    ixList = list(codeDict.keys())
    ## order by the BPT parameter - best codes to top of the list
    kd2n = [float(codeDict[ix][-1]) for ix in ixList]
    myOrd = argsort(kd2n,reverse=True)
    ## print code list to output
    print(f'n\tk\tdZ\tdX\tl\tm\tuVec{"\t" * (c-1)}\tvVec{"\t" * (c-1)}\tXCert\tkd2/n')
    for i in myOrd:
        myRow = codeDict[ixList[i]]
        print("\t".join(map(str,myRow)))

if __name__ == '__main__': 
    ## Change to vary the stabiliser weight - c=7 means BB7 codes
    c = 7

    ## Change to vary the number of codes sampled for each combination of (l, m)
    sampleSize = 100
    ## Search for codes with up to nMax physical qubits
    nMax = 150

    ## these calculate the (l,m) combinations to explore
    c2 = c//2
    c1 = c-c2
    nMax = nMax//2
    lMax = round(nMax ** 0.5)

    ## Generate Cycles in Serial
    # for l in range(c2,nMax//c2+1):
    #     BBCycles(l,c)
    #     print('BBCycles',l,c,'Done')

    ## Generate Codes in Serial
    # for l in range(c2,lMax+1):
    #     for m in range(1,nMax//l+1):
    #         BBGen(l,m,c,sampleSize)
    #         print('BBGen',l,m,'Done')
    
    ## Generate Cycles in Parallel
    # with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
    #     threadFuture = {executor.submit(BBCycles,l,c): l for l in range(c2,nMax//c2+1) }
    #     for future in concurrent.futures.as_completed(threadFuture):
    #         lm = (threadFuture[future])
    #         print(f'BBCycles {lm} Done')

    ## Generate Codes in Parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        threadFuture = {executor.submit(BBGen,l,m,c,sampleSize): (l,m) for l in range(c2,lMax+1) for m in range(1,nMax//l+1) }
        for future in concurrent.futures.as_completed(threadFuture):
            lm = (threadFuture[future])
            print(f'BBGen {lm} Done')

    ## Analysis - order codes by how much they exceed BPT bound and merge equivalent [[g*n,g*k,d]] codes
    # infile = f'BB{c}/BB{c}Gen.txt'
    # BBAnalyse(infile)

    ## Grab a list of codes from infile 
    # CodeList = BBCodeList(infile)
    # print(f'codes found: {len(CodeList)}')
