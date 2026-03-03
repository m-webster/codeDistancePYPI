import numpy as np
from .common import *
from .NHow import *
import json

#################################
# Functions needed to work with complexes
#################################

def complexCheck(AList):
    """Check if AList is a valid complex"""
    for i in range(len(AList) - 1):
        Ai = AList[i]
        mi, ni = np.shape(Ai)
        Aj = AList[i + 1]
        mj, nj = np.shape(Aj)
        ## check dimension of matrices
        if ni != mj:
            print(f"ni={ni} != mj={mj} for i={i},j={i+1}")
            return False
        ## check that successive operators multiply to zero
        AiAj = matMul(Ai, Aj, 2)
        if not np.sum(AiAj) == 0:
            print(f"Ai@Aj != 0 for i={i},j={i+1}")
            return False
    return True


def complexTrim(AList):
    """Remove any all zero matrices from beginning of AList."""
    temp = []
    i = 0
    while np.sum(AList[i]) == 0:
        i += 1
    return AList[i:]


def complexAppendZero(AList):
    """Add zero operator to beginning of AList"""
    m, n = np.shape(AList[0])
    return [ZMatZeros((1, m))] + AList


def complexNew(AList):
    """Make a new complex - make sure there's a zero operator at the end"""
    AList = complexTrim(AList)
    AList = complexAppendZero(AList)
    return AList


def complexDims(AList):
    """Return dimensions of each space acted upon by AList."""
    return [np.shape(A)[1] for A in AList]


def RG2Complex(myrow):
    """convert string format to RG complex boundary operators"""
    boundaryOperators = []
    ## read boundary maps for each level of the complex
    for myLabel in [f"Z{i}" for i in range(4, -1, -1)]:
        if myLabel in myrow:
            ## convert to boundary operators
            boundaryOperators.append(str2ZMatdelim(myrow[myLabel]))
    return complexNew(boundaryOperators)


def importRGList(myfile):
    """Import hyperbolic surface codes stored in myfile.
    Records in myfile are stored in JSON format.
    Parse each record and return list of dict codeList."""
    # mypath = sys.path[0] + "/hyperbolic_codes/"
    # f = open(mypath + myfile, "r")
    f = open(myfile, "r")
    mytext = f.read()
    mytext = mytext.replace("\\\n", "").replace("\n", "")
    mytext = mytext.replace("{", "\n{")
    mytext = mytext.split("\n")
    codeList = []
    for myline in mytext:
        if len(myline) > 0 and myline[0] != "#":
            myrow = json.loads(myline)
            codeList.append([myrow["index"], RG2Complex(myrow)])
    f.close()
    return codeList


def printRGList(codeList, myfile, checkValid=False):
    """Print parameters of the hyperbolic codes stored in list of dict codeList"""
    temp = []
    temp.append(f"Codes in File {myfile}:\n")
    valTxt = "\tValid" if checkValid else ""
    D = len(codeList[0][1])
    myrow = f"i\tindex{valTxt}"
    for i in range(D):
        myrow += f"\t|C{i}|"
    temp.append(myrow)
    for i in range(len(codeList)):
        myrow = codeList[i]
        ix = myrow[0]
        C = myrow[1]
        rowDesc = [i, ix]
        if checkValid:
            rowDesc += [complexCheck(C)]
        rowDesc += complexDims(C)
        temp.append("\t".join([str(a) for a in rowDesc]))
    return "\n".join(temp)


def complexCProduct(C):
    if len(C) == 0:
        return []
    P = C[0]
    temp = [P.T]
    for i in range(1, len(C)):
        P = mod1(P @ C[i])
        temp.append(P.T)
    return temp
