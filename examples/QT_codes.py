from codedistance import *
import os

def addMTXComments():
    '''for use with GAP QDistRnd - first 3 lines of MTX file need to be comments'''
    myDir = 'tanner_codes'
    for name in os.listdir(myDir):
        myfile = os.path.join(myDir, name)
        if os.path.isfile(myfile):
            myText = []
            with open(myfile,'r') as f:
                myText = f.read().split("\n")
            if len(myText) > 0:
                i = 0
                while i < len(myText) and len(myText[i]) > 0 and myText[i][0] == "%":
                    i += 1
                j = 3-i
                if j > 0:
                    myRows = myText[:i] + (["%"] * j ) + myText[i:]
                    with open(myfile,'w') as f:
                        print(name)
                        f.write("\n".join(myRows))



def QTCodeList():
    myDir = 'tanner_codes'
    codeNames = set()
    for name in os.listdir(myDir):
        if os.path.isfile(os.path.join(myDir, name)):
            name = name[3:-4]
            nameTup = name.split("_")
            if len(nameTup) == 3:
                params = tuple(map(int, nameTup))
                codeNames.add((params,name))
    
    codeList = []
    startTimer()
    for params,name in sorted(codeNames):
        n,k,d = params
        Hx=scipy.io.mmread(f'{myDir}/HX_{name}.mtx').todense()
        Hz=scipy.io.mmread(f'{myDir}/HZ_{name}.mtx').todense()
        # d,T,R,v, stdOut = DistRandCSS(Hx,Hz,params={'iterCount':10000},seed=0)
        # print(f'{name}\t{d}\t{elapsedTime()}')
        myCode = CSS2Dict(Hx, Hz, name=name, d=d)
        if myCode['k'] > 0:
            codeList.append(myCode)
    return codeList

# QTCodeList()
# addMTXComments()
