import numpy as np
from codedistance import *
from bivariate_bicycle import *
from lifted_product import *
from QT_codes import *
import argparse
import copy 

################################################################################################
## Parameter parser
################################################################################################

def defaultParser():
    '''parser for command line python scripts'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--runMode", help="How to run scenarios: slurm - generate slurm script; parallel - run in parallel using concurrent.futures; serial - run in serial",type=str, default='serial')
    parser.add_argument("--maxCount", help="Max number of codes/circuits to run from each dataset/datafile combination",type=int, default=10000) 
    parser.add_argument("--repeat", help="Number of times to repeat for each code/circuit",type=int, default=1) 
    parser.add_argument("--LOCheck", help="Check lowest weight logical operator",type=int, default=0) 

    parser.add_argument("--dataset", help="Source Type. Options are codeTables, lifted_product, bivariate_bicycle, all",type=str, default='codeTables')
    parser.add_argument("--datafile", help="File containing examples - used for codeTables, stim_circuit, hyperbolic codes",type=str, default='')
    parser.add_argument("--method", help="Distance finding method to use - options are DistRndGAP, QDistRndMW, dist_m4ri_RW, dist_m4ri_CC, GurobiDist, MIPDist, CLISATDist, pySATDist, decoderDist, UndetectableErrorStim, GraphLikeErrorStim, ColourCodeDistStim, connectedClusterMW, UndetectableErrorMW, GraphLikeErrorMW, ColourCodeDistMW, magmaMinWeight, magmaMinWord, magmaWEDist, qubitserfBZ, qubitserfMM, BZDistMW, MeetMiddleMW",type=str, default='dist_m4ri_RW')
    parser.add_argument("--ignoreTrivialCodes", help="Ignore codes with trivial codespaces",type=int, default=1)
    parser.add_argument("--resume", help="Resume distance finding from previous run - will skip over codes where distance has already been calculated",type=int, default=0)
    parser.add_argument("--nMin", help="Only run codes with n >= nMin",type=int, default=1)  
    parser.add_argument("--nMax", help="Only run codes with n <= nMax",type=int, default=100000)  

    ## generic parameters
    parser.add_argument("--iterCount", help="Number of Iterations to run",type=int, default=10000)  
    parser.add_argument("--maxTime", help="Max run time",type=int, default=3600*8)  
    parser.add_argument("--nThreads", help="Number of threads to run where parallel processing available",type=int, default=1)  
    parser.add_argument("--verbose", help="Display verbose debugging info",type=bool, default=False)  
    parser.add_argument("--maxErr", help="Track codewords and terminate when pErr < maxErr - for probabilistic methods only",type=float, default=-1)  
    parser.add_argument("--GF4blockRep", help="Number of blocks to use when representing non-CSS quantum codes - 2, 3 or 4",type=int, default=2)  
    parser.add_argument("--regroupPerm", help="For random information set algorithms on non-CSS codes, reorder permutations so that columns corresponding to the same qubit are next to each other",type=int, default=1)  

    ## Circuits only 
    parser.add_argument("--filterCircuit", help="Circuit distance only: filter circuits to include on X-Paulis and Z-measurements",type=int, default=1)  

    ## QDistEvol/QDistRndMW only
    parser.add_argument("--HL", help="QDistEvol/QDistRndMW only: use stacked HL rather than Pryadko kernel method",type=int, default=0)   
    parser.add_argument("--genCount", help="QDistEvol only: population size",type=int, default=100)  
    parser.add_argument("--offspring", help="QDistEvol only: offspring per parent",type=int, default=10) 
    parser.add_argument("--pMut", help="QDistEvol only: average number of transpositions per mutation",type=float, default=2.0) 
    parser.add_argument("--sMut", help="QDistEvol only: mutation standard deviation as proportion of pMut",type=float, default=0.2) 
    parser.add_argument("--tabuLength", help="QDistEvol only: length of tabu list",type=int, default=0) 
    parser.add_argument("--swapPivot", help="QDistEvol only: swap between pivots and non-pivots",type=int, default=1) 
    parser.add_argument("--swapBlockorder", help="QDistEvol only: swap blocks for same qubit",type=int, default=0) 
    parser.add_argument("--pMutScale", help="QDistEvol only: scale pMut (number of transpositions) by code length - set to 0 for user-defined pMut",type=int, default=50) 

    ## method==dist_m4ri_CC, connectedClusterMW, UndetectableErrorMW, GraphLikeErrorMW, ColourCodeDistMW only
    parser.add_argument("--CCstart", help="Include logicals with support on qubit start",type=int, default=-1)  

    ## method=='MIPDist' only
    parser.add_argument("--MIP_solver", help='''Type of MIP solver - options are CBC_MIXED_INTEGER_PROGRAMMING or CBC
    SAT_INTEGER_PROGRAMMING or SAT or CP_SAT
    SCIP_MIXED_INTEGER_PROGRAMMING or SCIP
    GUROBI_MIXED_INTEGER_PROGRAMMING or GUROBI or GUROBI_MIP - requires full installation of Gurobi - https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer
    CPLEX_MIXED_INTEGER_PROGRAMMING or CPLEX or CPLEX_MIP - requires installation of CPLEX https://www.ibm.com/docs/en/icos
    XPRESS_MIXED_INTEGER_PROGRAMMING or XPRESS or XPRESS_MIP - requires installation of Xpress https://www.fico.com/fico-xpress-optimization/docs/latest/getting_started/dhtml/chap16.html
    GLPK_MIXED_INTEGER_PROGRAMMING or GLPK or GLPK_MIP - requires installation of GLPK: https://www.gnu.org/software/glpk/''',type=str, default='SCIP')
    
    ## method=='pySATDist' only
    parser.add_argument("--pySATsolver", help="SAT solver to use - options are cd, cd15, cd19, cms, gc3, gc4, g3, g4, g42, lgl, mcb, mcm, mpl, mg3, mc, m22, mgh" ,type=str, default='g4')
    parser.add_argument("--pySATbinary", help="pySAT binary  to use - options are rc2, fm, lsu" ,type=str, default='lsu')

    ## method=='decoderDist' only
    parser.add_argument("--decoder", help="Type of decoder - for decoderDist only. Options are bposd, bplsd, sebplsd, tesseract, tesseractLongbeam",type=str, default='bposd')
    parser.add_argument("--addStabs",help="Add stabilisers to randomly generated error",type=int,default=0) 
    parser.add_argument("--permuteDEM",help="Apply random permutation to columns of DEM",type=int,default=1) 
                        
    ## method=='decoderDist' and decoder in {'bposd','bplsd','sebplsd'} only
    ## product_sum good for code capacity, bad for circuit level noise - don't have sparse matrices in this case
    parser.add_argument("--bp_method", help="Belief Propagation method",type=str, default='product_sum')
    parser.add_argument("--bp_schedule", help="Belief Propagation schedule type - options parallel or serial",type=str, default='parallel')
    ## code capacity ~ log(n) 
    parser.add_argument("--bp_max_iter", help="Belief Propagation max iterations",type=int, default=100)
    ## try OSD_E with low order - 10; OSD_CS usually has high error floor, esp circuit level noise - scales as n^2
    ## LSD_E - calls dense OSD subprocess - should work well for circuit level noise
    parser.add_argument("--osd_method", help="OSD method",type=str, default='OSD_CS') 
    parser.add_argument("--osd_order", help="OSD order",type=int, default=1)
    return parser

def matchFile(myDir,outfile):
    for qcFile in os.listdir(f'{myDir}'):
        if qcFile.find(outfile) >= 0:
            return qcFile
    return ""

def set_global_params(params):
    '''Convert params object to dict and remove irrelevant settings'''
    paramDict = vars(params)
    method = params.method
    decoder = params.decoder
    MIP_solver = paramDict['MIP_solver']
    if method != 'decoderDist' or decoder not in {'bposd','bplsd','sebplsd'}:
        for k in ["bp_method","bp_schedule", "bp_max_iter", "osd_method","osd_order"]:
            del(paramDict[k]) 
    elif params.decoder not in {'bposd'}:
        for k in ["osd_method","osd_order"]:
            del(paramDict[k]) 
    if method != 'decoderDist':
        del(paramDict['decoder'])
        decoder = ""
    if method != 'MIPDist':
        del paramDict['MIP_solver']
        MIP_solver = ""
    outfile = f'{params.dataset}-{params.datafile + "-" if len(params.datafile) > 0 else ""}{method}-{ decoder + "-" if len(decoder) > 0 else ""}{MIP_solver + "-" if len(MIP_solver) > 0 else ""}'
    myDir = f'{os.getcwd()}/results/'
    dirCheck(myDir)
    ## if resume parameter set to True, search for a previous results file to update
    qcFile = matchFile(myDir,outfile) if params.resume else ""
    if len(qcFile) > 0:
        paramDict['outfile'] = f'{myDir}{qcFile}'
    else:
        paramDict['outfile'] = f'{myDir}{outfile}{randomFilename()}.txt'
        write_params(paramDict)
    return paramDict

def dict2str(myDict):
    temp = []
    for k,v in myDict.items():
        temp.append(f"{k}: {v}")
    return "\n".join(temp)

def write_params(params):
    '''Write search parameters to file and std output'''
    temp = [dict2str(params)]
    temp.append("#########################################")
    temp.append("name\tn\tk\td\tdelta d\tTime\tR\tT\tProgress")
    temp.append("")
    mytext = "\n".join(temp)
    outfile = params['outfile']
    if outfile is not None:
        with open(outfile,'w') as f:
            f.write(mytext)
    print(mytext)

def saveRes(outfile,res,codeDict):
    res['t'] = elapsedTime()
    with open(outfile,'a') as f:
        progress = res['progress'].replace("\t","\\t").replace("\n","\\n")
        mystr = f"{codeDict['name']}\t{codeDict['n']}\t{codeDict['k']}\t{res['d']}\t{res['d'] - codeDict['d']}\t{res['t']}\t{res['R']}\t{res['T']}\t{progress}"
        f.write(mystr + "\n")

def distanceTest(params):
    count = 0
    paramDict = set_global_params(params)
    dataset,datafile,method,outfile = paramDict['dataset'],paramDict['datafile'],paramDict['method'],paramDict['outfile']
    ## extract text before first tab in each line to determine which codes we have already calculated distances for
    codesDone = set()
    if params.resume:
        with open(outfile) as f:
            temp = f.read()
            temp = [myLine.split("\t")[0].strip() for myLine in temp.split("\n")]
            
            afterName = False
            for k in temp:
                if k == 'name':
                    afterName = True
                elif afterName and len(k) > 0:
                    codesDone.add(k)
    ## construct codeList
    codeList = []
    if dataset == 'lifted_product':
        codeList = LPCodeList()
    elif dataset == 'lifted_product_GF2':
        codeList = LPCodeListGF2()
    elif dataset == 'bivariate_bicycle':
        codeList = BBCodeList()
    elif dataset == 'BB_test':
        codeList = [myCode for myCode in BBCodeList() if myCode['n'] == 756]
    elif dataset == 'tanner_codes':
        codeList = QTCodeList()
    elif dataset[:10] == 'hyperbolic':
        myFile = f'hyperbolic_codes/{datafile}'
        if not os.path.exists(myFile):
            print(f'file {myFile} not found')
        else: 
            RGCodes = importRGList(myFile)
            for myrow in RGCodes:
                if dataset == 'hyperbolic_surface_code':
                    ## Surface Code
                    Hz,Hx = complex2SurfaceCode(myrow[1])
                else:
                    ## Colour Code
                    Hx,Hz = complex2ColourCode(myrow[1])
                codeList.append(CSS2Dict(Hx,Hz))

    elif dataset=='codeTables':
        myFile = f"codeTables/{datafile}"
        if not os.path.exists(myFile):
            print(f'file {myFile} not found')
        else: 
            CTcodes = CodetablesImport(myFile)
            codeType = 'GF2' if (datafile[:3] == "GF2") else 'QECC'
            ## exclude codes with rate greater than kMax - set to None to stop this
            kMax = None
            for params,S in CTcodes:
                n,k,d = params
                if (kMax is None) or (k/n <= kMax):
                    codeList.append(codeTables2Dict(params,S,codeType))

    elif dataset=='stim_circuit':
        dirPath = f'stim_circuits/{datafile}'
        if not os.path.exists(dirPath):
            print(f'circuit directory {dirPath} not found')
        else: 
            # print (dirPath)
            for qcFile in sorted(os.listdir(dirPath)):
                # print(qcFile)
                filePath = os.path.join(dirPath,qcFile)
                if (qcFile[0] != ".") and os.path.isfile(filePath):
                    qc = stim.Circuit.from_file(filePath)
                    codeList.append(qc2Dict(qc,qcFile))
    else:
        print(f'could not find dataset {dataset}')

    ## calculate distances
    startTimer()
    for codeDict in codeList:
        n,k,d = codeDict['n'],codeDict['k'],codeDict['d']
        print(f"{codeDict['name']} [[{n},{k},{d}]]")
        if codeDict['name'] in codesDone:
            print('distance already calculated')
        elif (codeDict['n'] >= paramDict['nMin']) and (codeDict['n'] <= paramDict['nMax']) and  count < paramDict['maxCount'] and ((codeDict['k'] > 0 ) or (not paramDict['ignoreTrivialCodes'])):
            for c in range(paramDict['repeat']):
                count += 1
                ## distance finding function depends on code type
                if codeDict['type'] == 'QECC':
                    S, L = codeDict['S'],codeDict['L']
                    tB = 2
                    res = codeDistance(S,L,tB,method=method,params=paramDict)
                elif codeDict['type'] == 'GF2':
                    S = codeDict['S']
                    L = codeDict['L']
                    tB = 1
                    res = codeDistance(S,L,tB,method=method,params=paramDict)
                elif codeDict['type'] == 'CSS':
                    S,L = (codeDict['SX'],codeDict['LX'])
                    tB=1
                    res = codeDistance(S,L,tB,method=method,params=paramDict)
                elif codeDict['type'] == 'circuit':
                    res = circuitDistance(codeDict['qc'],method=method,params=paramDict)
                saveRes(outfile,res,codeDict)
                print(f"dCalc: {res['d']}; time: {res['t']}")
                
    
def paramObj2cmd(paramObj):
    temp = []
    for k,v in vars(paramObj).items():
        v = str(v)
        if len(v) > 0:
            temp.append(f"--{k} {v}")
    return " ".join(temp)


if __name__ == '__main__':
    parser = defaultParser()
    params = parser.parse_args()
    allDatasets = {
        'lifted_product':[],
        'bivariate_bicycle':[],
        'hyperbolic_surface_code':['RG-3-8.txt'],
        'hyperbolic_colour_code':['RG-3-8.txt'],
        'codeTables':['GF2_sample.txt','QECC_sample.txt'],
        'stim_circuit':['01_surfacecodes','03_colorcodes_midout','04_colorcodes_superdense','05_bivariatebicyclecodes']
    }
    ## results methods
    allMethods = ['QDistEvol','dist_m4ri_RW','QDistRndMW','UndetectableErrorStim','GraphLikeErrorStim','ColourCodeDistStim','decoderDist','qubitserfMM','dist_m4ri_CC','GurobiDist','MIPDist','pySATDist']
    
    methods = allMethods if params.method == 'all' else [params.method ]
    datasets = allDatasets if params.dataset == 'all' else {params.dataset : [params.datafile]}

    jobList = []
    for m in methods:
        for k,fileList in datasets.items():
            if len(fileList) == 0:
                fileList = [""]
            for datafile in fileList:
                paramObj = copy.deepcopy(params)
                paramObj.method = m
                paramObj.dataset = k
                paramObj.datafile = datafile
                jobList.append(paramObj)
        
    if params.runMode == 'slurm':
        ## print list of slurm commands
        print(f'#SBATCH --ntasks={len(jobList)}')
        for i,paramObj in enumerate(jobList):
            paramObj.iterCount = 1000 if paramObj.dataset == 'stim_circuit' else 10000 
            mem = 8
            df = "" if paramObj.datafile == "" else f'--datafile {paramObj.datafile}'
            print(f'srun -n1 --exclusive --ntasks=1 --mem-per-cpu {mem}GB  python3 runTest.py --method {paramObj.method} --iterCount {paramObj.iterCount} --dataset {paramObj.dataset} {df} --resume 1 &')
        print(f'wait')

    elif params.runMode == 'parallel':
        ## parallel processing using concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
            threadFuture = {executor.submit(distanceTest,paramObj): i for i,paramObj in enumerate(jobList)}
            for future in concurrent.futures.as_completed(threadFuture):
                print(threadFuture[future])
    else:
        ## serial processing - single thread
        for i,paramObj in enumerate(jobList):
            distanceTest(paramObj)
