from codedistance import *
import numpy as np

################################################################################################
## Bivariate bicycle codes
################################################################################################

def BBCodeList():
    codeList = []
    generated_bb_codes = [
        ## IBM Codes
        ##  IBM 72-qubit code d=6
        {
                "Apoly": [3, 1, 2],
                "Bpoly": [3, 1, 2],
                "l": 6,
                "m": 6,
                "n": 72,
                "k": 12,
                "d": 6,
        },
        # 90-qubit code d=10
        {
                "Apoly": [9, 1, 2],
                "Bpoly": [0, 2, 7],
                "l": 15,
                "m": 3,
                "n": 90,
                "k": 8,
                "d": 10,
        },
        # 108-qubit code d=10
        {
                "Apoly": [3, 1, 2],
                "Bpoly": [3, 1, 2],
                "l": 9,
                "m": 6,
                "n": 108,
                "k": 8,
                "d": 10,
        },
        # 144-qubit code d=12
        {
                "Apoly": [3, 1, 2],
                "Bpoly": [3, 1, 2],
                "l": 12,
                "m": 6,
                "n": 144,
                "k": 12,
                "d": 12,
        },
        # 288-qubit code d=18
        {
                "Apoly": [3, 2, 7],
                "Bpoly": [3, 1, 2],
                "l": 12,
                "m": 12,
                "n": 288,
                "k": 12,
                "d": 18,
        },
        # 360-qubit code - d < 24
        {
                "Apoly": [9, 1, 2],
                "Bpoly": [3, 25, 26],
                "l": 30,
                "m": 6,
                "n": 360,
                "k": 12,
                "d": 24,
        },
        # 756-qubit code - d < 34
        {
                "Apoly": [3, 10, 17],
                "Bpoly": [5, 3, 19],
                "l": 21,
                "m": 18,
                "n": 756,
                "k": 16,
                "d": 34,
        },

        ## Abe Jacobs generated codes
        {
            "Apoly": [3, 4, 14],
            "Bpoly": [2, 4, 0],
            "l": 6,
            "m": 18,
            "n": 216,
            "k": 8,
            "d": 10,
        },
        {
            "Apoly": [8, 2, 6],
            "Bpoly": [5, 5, 6],
            "l": 12,
            "m": 24,
            "n": 576,
            "k": 8,
            "d": 24,
        },
        {
            "Apoly": [7, 23, 21],
            "Bpoly": [1, 3, 2],
            "l": 12,
            "m": 36,
            "n": 864,
            "k": 4,
            "d": 40,
        },
        {
            "Apoly": [6, 10, 17],
            "Bpoly": [16, 15, 4],
            "l": 18,
            "m": 18,
            "n": 648,
            "k": 4,
            "d": 32,
        },
        {
            "Apoly": [15, 13, 14],
            "Bpoly": [23, 9, 17],
            "l": 18,
            "m": 30,
            "n": 1080,
            "k": 4,
            "d": 54,
        },
        {
            "Apoly": [15, 8, 16],
            "Bpoly": [25, 17, 9],
            "l": 18,
            "m": 36,
            "n": 1296,
            "k": 4,
            "d": 48,
        },
        {
            "Apoly": [21, 16, 2],
            "Bpoly": [1, 10, 6],
            "l": 24,
            "m": 24,
            "n": 1152,
            "k": 4,
            "d": 36,
        },
        {
            "Apoly": [23, 15, 11],
            "Bpoly": [35, 0, 14],
            "l": 24,
            "m": 36,
            "n": 1728,
            "k": 4,
            "d": 64,
        },
        {
            "Apoly": [16, 39, 2],
            "Bpoly": [31, 8, 18],
            "l": 24,
            "m": 42,
            "n": 2016,
            "k": 8,
            "d": 54,
        },
        # {
        #     "Apoly": [20, 2, 8],
        #     "Bpoly": [10, 19, 5],
        #     "l": 30,
        #     "m": 30,
        #     "n": 1800,
        #     "k": 16,
        #     "d": 8,
        # },
        {
            "Apoly": [18, 13, 35],
            "Bpoly": [3, 26, 25],
            "l": 30,
            "m": 36,
            "n": 2160,
            "k": 8,
            "d": 64,
        },
        {
            "Apoly": [24, 8, 13],
            "Bpoly": [23, 16, 6],
            "l": 30,
            "m": 42,
            "n": 2520,
            "k": 8,
            "d": 54,
        },
        {
            "Apoly": [8, 25, 3],
            "Bpoly": [31, 2, 18],
            "l": 36,
            "m": 36,
            "n": 2592,
            "k": 8,
            "d": 36,
        },
        {
            "Apoly": [3, 26, 19],
            "Bpoly": [28, 23, 12],
            "l": 36,
            "m": 42,
            "n": 3024,
            "k": 4,
            "d": 78,
        },
    ]
    startTimer()
    for codeDict in generated_bb_codes:
        Hx, Hz = BBIBM(codeDict["l"], codeDict["m"], codeDict["Apoly"], codeDict["Bpoly"])
        name = f'BB{codeDict['n']}'
        myCode = CSS2Dict(Hx, Hz, d=codeDict["d"])
        if myCode['k'] > 0:
            codeList.append(myCode)
    return codeList