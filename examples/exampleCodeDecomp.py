import numpy as np
from codedistance import *

## Illustrate Decomposition of Codes into Direct Sums - Slepian, 1960


## Eberhardt108
n,k,d = 108,16,6
l,m,ALen = 6,9,3
uVec = '0 0 0 0 2 4'
vVec = '0 1 2 3 0 0'

## Eberhardt128
n,k,d = 128,14,12
l,m,ALen = 8,8,4
uVec = '2 0 0 0 0 1 3 4'
vVec = '0 1 3 4 2 0 0 0'

## Eberhardt162-4
n,k,d = 162,4,16
l,m,ALen = 9,9,3
uVec = '0 1 0 3 0 0'
vVec = '0 0 1 0 1 2'

## Eberhardt162-12
n,k,d = 162,12,8
l,m,ALen = 9,9,3
uVec = '0 1 0 0 2 3'
vVec = '0 0 6 3 0 0'

## Eberhardt162-24
n,k,d = 162,24,6
l,m,ALen = 9,9,3
uVec = '0 0 0 0 3 6'
vVec = '0 1 2 3 0 0'

uVec = list(map(int,uVec.split(" ")))
vVec = list(map(int,vVec.split(" ")))
SX,SZ = BBCSSCode(l,m,uVec,vVec,ALen)
qubitPartitions = codeDecomp(ZMatVstack([SX,SZ]))
if len(qubitPartitions) == 1:
    print('No partitions found')
else:
    print(f'{len(qubitPartitions)} partitions found')
    for i,ix in enumerate(qubitPartitions):
        ix = sorted(ix)
        print(f'Subcode {i}:',ix)
        print('SX')
        print(ZMatPrint(RemoveZeroRows(SX[:,ix])))
        print('SZ')
        print(ZMatPrint(RemoveZeroRows(SZ[:,ix])))



