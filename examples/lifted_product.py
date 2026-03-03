from typing import Tuple
import numpy as np
import numpy.typing as npt
import numba as nb
from numba import njit
from codedistance import *


@njit(nb.uint8[:, :](nb.uint8[:, :], nb.int64, nb.int64))
def block_kron_eye(M: npt.NDArray[np.uint8], eye_dim: int, ell: int):
    """Compute kron(M, I) where I is an eye_dim x eye_dim identity
    matrix and every matrix element is an element of a matrix ring
    M_{ell}(F_2).

    M is a block matrix where each block has ell rows and columns.
    Specifically, M is a matrix over some matrix ring R where each
    element of R is an ell by ell binary matrix.
    This method computes kron(M, I) where I is an identity matrix
    over the same matrix ring R. I has eye_dim rows and columns (
    when considering each element to be an element of R), or
    eye_dim * ell rows and columns when considering its full binary
    representation in numpy. Here kron is the kronecker product.

    Parameters
    ----------
    M : npt.NDArray[np.uint8]
        The full binary representation of a matrix over a matrix ring R
    eye_dim : int
        The dimension of the identity matrix to take the kronecker product
        with.
    ell : int
        The number of rows and columns of an element of the matrix ring R
    """
    if M.shape[0] % ell != 0 or M.shape[1] % ell != 0:
        raise ValueError("M should have dimensions which are a multiple of ell")

    out = np.zeros((M.shape[0] * eye_dim, M.shape[1] * eye_dim), dtype=np.uint8)

    for i in range(M.shape[0] // ell):
        for j in range(M.shape[1] // ell):
            Mij_eye = np.zeros((ell * eye_dim, ell * eye_dim), dtype=np.uint8)
            Mij = M[i * ell : (i + 1) * ell, j * ell : (j + 1) * ell]
            for q in range(eye_dim):
                Mij_eye[q * ell : (q + 1) * ell, q * ell : (q + 1) * ell] = Mij

            out[
                i * ell * eye_dim : (i + 1) * ell * eye_dim,
                j * ell * eye_dim : (j + 1) * ell * eye_dim,
            ] = Mij_eye
    # out2 = np.kron(M,np.eye(eye_dim,dtype=np.uint8))
    # print('block_kron_eye',np.sum(out ^ out2))
    # print(out.shape, out2.shape)
    # print(ZMatPrint(out))
    return out


# @njit(nb.uint8[:, :](nb.uint8[:, :], nb.int64))
def eye_block_kron(M: npt.NDArray[np.uint8], eye_dim: int):
    """Compute kron(I, M) where I is an eye_dim x eye_dim identity
    matrix and every matrix element is an element of a matrix ring
    M_{ell}(F_2).

    M is a block matrix where each block has ell rows and columns.
    Specifically, M is a matrix over some matrix ring R where each
    element of R is an ell by ell binary matrix.
    This method computes kron(I, M) where I is an identity matrix
    over the same matrix ring R. I has eye_dim rows and columns (
    when considering each element to be an element of R), or
    eye_dim * ell rows and columns when considering its full binary
    representation in numpy. Here kron is the kronecker product.

    Parameters
    ----------
    M : npt.NDArray[np.uint8]
        The full binary representation of a matrix over a matrix ring R
    eye_dim : int
        The dimension of the identity matrix to take the kronecker product
        with.

    """
    out = np.zeros((M.shape[0] * eye_dim, M.shape[1] * eye_dim), dtype=np.uint8)

    for i in range(eye_dim):
        out[
            i * M.shape[0] : (i + 1) * M.shape[0], i * M.shape[1] : (i + 1) * M.shape[1]
        ] = M
    # out2 = np.kron(np.eye(eye_dim,dtype=np.uint8),M)
    # print('eye_block_kron',np.sum(out ^ out2))
    # print(out.shape, out2.shape)
    # print(ZMatPrint(out))
    return out


def lifted_product_check_matrices(
    A: npt.NDArray[np.uint8], B: npt.NDArray[np.uint8], ell: int
):
    """Construct the X and Z check matrices for the lifted product of A and B.
    A and B are each matrices over a matrix ring R, where each element of R is
    represented by a binary ell x ell matrix.
    A and B are each represented as a binary block matrix, where each block is
    the corresponding element of R.
    See Section III(D) of https://arxiv.org/abs/2012.04068 for a complete
    definition.

    Parameters
    ----------
    A : npt.NDArray[np.uint8]
        A binary matrix. Represents a matrix over some matrix ring M_{ell}(F_2)
    B : npt.NDArray[np.uint8]
        A binary matrix. Represents a matrix over some matrix ring M_{ell}(F_2)
    """

    if (
        A.shape[0] % ell != 0
        or A.shape[1] % ell != 0
        or B.shape[0] % ell != 0
        or B.shape[1] % ell != 0
    ):
        raise ValueError("A and B should have dimensions that are a multiple of ell")

    m_a = A.shape[0] // ell
    n_a = A.shape[1] // ell
    m_b = B.shape[0] // ell
    n_b = B.shape[1] // ell

    # Hx = [A \otimes I_{m_b}, I_{m_a} \otimes B]
    # Hx has shape (m_b * m_a) x (n_a * m_b + m_a * n_b) where each element is an ell x ell block
    Hx = np.zeros((ell * m_b * m_a, ell * (n_a * m_b + m_a * n_b)), dtype=np.uint8)
    Hx[:, 0 : n_a * m_b * ell] = block_kron_eye(A, eye_dim=m_b, ell=ell)
    Hx[:, n_a * m_b * ell :] = eye_block_kron(B, eye_dim=m_a)

    # Hz = [I_{n_a} \otimes B^*, A^* \otimes I_{n_b}]
    # Hz has shape (n_a * n_b) x (n_a * m_b + m_a * n_b) where each element is an ell x ell block
    Hz = np.zeros((ell * n_a * n_b, ell * (n_a * m_b + m_a * n_b)), dtype=np.uint8)
    Hz[:, 0 : n_a * m_b * ell] = eye_block_kron(B.T, eye_dim=n_a)
    Hz[:, n_a * m_b * ell :] = block_kron_eye(A.T, n_b, ell=ell)

    return Hx, Hz


# def cyclic_right_shift(size: int) -> npt.NDArray[np.uint8]:
#     s = np.diag(np.ones(size - 1, dtype=np.uint8), k=-1)
#     s[0, size - 1] = 1
#     return s


def cyclic_right_shift_by(size: int, shift: int) -> npt.NDArray[np.uint8]:
    return np.roll(np.eye(size,dtype=np.uint8),shift,axis=0)
    # if size == 0:
    #     return np.eye(size, dtype=np.uint8)

    # s = np.diag(np.ones(size - shift, dtype=np.uint8), k=-shift)
    # s += np.diag(np.ones(shift, dtype=np.uint8), k=size - shift)
    # print(f'cyclic_right_shift_by - size:{size} shift:{shift}')
    # print(ZMatPrint(s))
    # return s


def protograph_to_quasi_cyclic_matrix(protograph: npt.NDArray[np.int64], ell: int):
    """Construct a quasi-cyclic matrix of circulant size ell from a protograph.

    Each element of the protograph is the power to which the cyclic right shift matrix
    should be raised to construct the corresponding ell x ell block in the
    quasi-cyclic matrix. In other words, we restrict here to the case where
    each element of the protograph can be a monomial from the quotient
    polynomial ring R_l = F_2[x]/(x^l - 1).

    Parameters
    ----------
    protograph : npt.NDArray[np.uint8]
        The protograph
    ell : int
        The circulant size
    """
    A = protograph
    M = np.zeros((A.shape[0] * ell, A.shape[1] * ell), dtype=np.uint8)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            M[i * ell : (i + 1) * ell, j * ell : (j + 1) * ell] = cyclic_right_shift_by(
                size=ell, shift=A[i, j]
            )
    return M


def quasi_cyclic_lifted_product_check_matrices_from_protographs(
    A: npt.NDArray[np.int64], B: npt.NDArray[np.int64], ell: int
) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    A_mat = protograph_to_quasi_cyclic_matrix(protograph=A, ell=ell)
    B_mat = protograph_to_quasi_cyclic_matrix(protograph=B, ell=ell)

    return lifted_product_check_matrices(A=A_mat, B=B_mat, ell=ell)


def quasi_cyclic_lifted_product_check_matrices_from_protograph(
    A: npt.NDArray[np.int64], ell: int
) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Construct the lifted product LP(A, A*) from the protograph
    A, with circulant size ell. Here A* is the conjugate
    transpose of A.

    Each element of the protograph A is the power to which the
    cyclic right shift matrix should be raised to construct the
    corresponding ell x ell block in the quasi-cyclic matrix.
    In other words, we restrict here to the case where
    each element of the protograph can be a monomial from the quotient
    polynomial ring R_l = F_2[x]/(x^l - 1).
    See Section III(D) of https://arxiv.org/abs/2012.04068 for a complete
    definition.

    Parameters
    ----------
    A : npt.NDArray[np.int64]
        The protograph. Element A[i, j] represents the
        associated monomial x^{A[i, j]} of a circulant matrix.
        i.e. It represents a monomial from the quotient
        polynomial ring R_l = F_2[x]/(x^l - 1).
    ell : int
        The circulant size l.

    Returns
    -------
    Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]
        The check matrices Hx and Hz defining the (CSS)
        lifted product code LP(A, A*).
    """
    A_mat = protograph_to_quasi_cyclic_matrix(protograph=A, ell=ell)
    return lifted_product_check_matrices(A=A_mat, B=A_mat.T, ell=ell)


# def LPClassicalCode(A,l):
#     return ZMatVstack( [ ZMatHstack( [SMatrix(l,p) for p in myRow]) for myRow in A] )


def LPCodeList():
    codeList = []
    nList = []
    # startTimer()
    LPC = LPCodes()
    for codeDict in LPC:
        A = np.array(codeDict["protograph"], dtype=int)
        l = codeDict["ell"]
        Hx, Hz = quasi_cyclic_lifted_product_check_matrices_from_protograph(A, l)
        nList.append( len(Hx.T))
        myCode = CSS2Dict(Hx, Hz, name=codeDict["name"], d=codeDict["d"])
        if myCode['k'] > 0:
            codeList.append(myCode)
    ix = argsort(nList)
    return [codeList[i] for i in ix]


def LPCodeListGF2():
    codeList = []
    nList = []
    LPC = LPCodes()
    for codeDict in LPC:
        A = np.array(codeDict["protograph"], dtype=int)
        l = codeDict["ell"]
        G = protograph_to_quasi_cyclic_matrix(A,l)
        r,n = G.shape
        H = KerZ2(G,np.arange(n),0)
        k = len(H)
        if k > 0:
            d = codeDict["d"]
            nList.append(n)
            myCode = codeTables2Dict([n,k,d],H,'GF2')
            codeList.append(myCode)
    ix = argsort(nList)
    return [codeList[i] for i in ix]

def LPCodes():
    return [
        ## Example 4 from https://arxiv.org/abs/2012.04068
        {
            "name": "PK_31",
            "protograph": [[1, 2, 4, 8, 16], [5, 10, 20, 9, 18], [25, 19, 7, 14, 28]],
            "ell": 31,
            "n": 1054,
            "k": 140,
            "d": 20,
        },
        # The following are from Xu et al: https://arxiv.org/abs/2308.08648. See Methods
        # Equations 5 and 6 on page 10 for the protographs. These lifted products
        # are the codes studied in their main text (See Figure 3(b) for upper
        # bounds on their distance, as well as their logical error rates).
        {
            "name": "xu_16",
            "protograph": [[0, 0, 0, 0, 0], [0, 2, 4, 7, 11], [0, 3, 10, 14, 15]],
            "ell": 16,
            "n": 544,
            "k": 80,
            "d": 12,
        },
        {
            "name": "xu_21",
            "protograph": [[0, 0, 0, 0, 0], [0, 4, 5, 7, 17], [0, 14, 18, 12, 11]],
            "ell": 21,
            "n": 714,
            "k": 100,
            "d": 16,
        },
        {
            "name": "xu_30",
            "protograph": [[0, 0, 0, 0, 0], [0, 2, 14, 24, 25], [0, 16, 11, 14, 13]],
            "ell": 30,
            "n": 1020,
            "k": 136,
            "d": 16,
        },
        {
            "name": "xu_42",
            "protograph": [[0, 0, 0, 0, 0], [0, 6, 7, 9, 30], [0, 40, 15, 31, 35]],
            "ell": 42,
            "n": 1428,
            "k": 184,
            "d": 24,
        },
        # New Codes Generated by Abe Jacob
        {
                "name": "AJ_01",
                "protograph": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                "ell": 1,
                "n": 34,
                "k": 20,
                "d": 2,
        },
        {
                "name": "AJ_04",
                "protograph": [[3, 3, 3, 2, 1], [0, 3, 1, 2, 0], [0, 2, 3, 2, 0]],
                "ell": 4,
                "n": 136,
                "k": 32,
                "d": 4,
            
        },
        {
                "name": "AJ_07",
                "protograph": [[5, 1, 6, 6, 4], [0, 2, 5, 0, 0], [1, 2, 2, 4, 6]],
                "ell": 7,
                "n": 238,
                "k": 44,
                "d": 6,
            
        },
        {
                "name": "AJ_10",
                "protograph": [[7, 5, 2, 0, 2], [1, 2, 8, 0, 8], [4, 8, 6, 9, 9]],
                "ell": 10,
                "n": 340,
                "k": 56,
                "d": 8,

        },
        {
            "name": "AJ_13",
                "protograph": [[0, 3, 5, 5, 1], [1, 7, 0, 8, 7], [8, 4, 12, 8, 0]],
                "ell": 13,
                "n": 442,
                "k": 68,
                "d": 10,
            
        },
        {
            "name": "AJ_52",
                "protograph": [
                    [23, 2, 22, 36, 7],
                    [13, 22, 21, 14, 30],
                    [20, 4, 26, 41, 38],
                ],
                "ell": 52,
                "n": 1768,
                "k": 224,
                "d": 22,
            
        },
        {
            "name": "AJ_58",
                "protograph": [
                    [2, 27, 47, 2, 2],
                    [48, 16, 55, 50, 12],
                    [25, 57, 5, 6, 5],
                ],
                "ell": 58,
                "n": 1972,
                "k": 248,
                "d": 20,
            
        },
        {"name":"AJ_60",
                "protograph": [
                    [2, 37, 9, 16, 9],
                    [24, 23, 3, 7, 35],
                    [31, 47, 41, 10, 55],
                ],
                "ell": 60,
                "n": 2040,
                "k": 256,
                "d": 20
        },
        {
            "name": "AJ_61",
                "protograph": [
                    [1, 0, 1, 28, 56],
                    [45, 49, 39, 58, 35],
                    [31, 2, 7, 33, 5],
                ],
                "ell": 61,
                "n": 2074,
                "k": 260,
                "d": 24,
            
        },
        {
            "name": "AJ_67",
                "protograph": [
                    [6, 25, 0, 4, 17],
                    [45, 34, 14, 6, 35],
                    [49, 53, 45, 52, 11],
                ],
                "ell": 67,
                "n": 2278,
                "k": 284,
                "d": 22,
            
        },
        {
            "name": "AJ_73",
                "protograph": [
                    [20, 11, 26, 12, 60],
                    [69, 40, 66, 60, 6],
                    [1, 36, 40, 42, 67],
                ],
                "ell": 73,
                "n": 2482,
                "k": 308,
                "d": 24,
            
        },
        {
            "name":"AJ_80",
                "protograph": [
                    [64, 10, 18, 49, 4],
                    [6, 23, 41, 78, 3],
                    [66, 62, 77, 22, 44],
                ],
                "ell": 80,
                "n": 2720,
                "k": 336,
                "d": 22
        },
        {
            "name":"AJ_85",
            "protograph": [
                [1, 52, 54, 57, 58],
                [32, 74, 40, 31, 46],
                [65, 5, 13, 68, 23],
            ],
            "ell": 85,
            "n": 2890,
            "k": 356,
            "d": 24,
            
        },
        {
            "name":"AJ_95",
                "protograph": [
                    [73, 81, 62, 11, 31],
                    [92, 88, 2, 39, 0],
                    [67, 50, 60, 3, 90],
                ],
                "ell": 95,
                "n": 3230,
                "k": 396,
                "d": 24
        },
    ]

