# Trilaterate
'''Tool for triangulating position in space with arbitrary dimensions'''

'''
Copyright (c) 2016 Joseph Kozak

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
'''


def _determinant(matrix):
    '''Determinant'''
    m, n = len(matrix), len(matrix[0])
    if m != n:
        return False
    if m == 1:
        return matrix[0][0]
    elif m == 2:
        determinant = (matrix[0][0]*matrix[1][1]) - (matrix[0][1]*matrix[1][0])
        return determinant
    else:
        D = 0
        for i, a in enumerate(matrix[0]):
            M = []
            for row in matrix[1:]:
                newrow = []
                for j, val in enumerate(row):
                    if j != i:
                        newrow.append(val)
                M.append(newrow)
            D += (-1**i)*a*_determinant(M)
        return D


def _mofmin(matrix):
    '''Matrix of Minors'''
    m, n = len(matrix), len(matrix[0])
    if m != n:
        return False
    if m == 1:
        return matrix
    else:
        MoM = []
        for i, row in enumerate(matrix):
            newrow = []
            for j, val in enumerate(row):
                M = []
                for k, mrow in enumerate(matrix):
                    if k != i:
                        mnewrow = []
                        for l, mval in enumerate(mrow):
                            if l != j:
                                mnewrow.append(mval)
                        M.append(mnewrow)
                newrow.append(_determinant(M))
            MoM.append(newrow)
        return MoM


def _mofcof(matrix):
    '''Matrix of Cofactors'''
    m, n = len(matrix), len(matrix[0])
    if m != n:
        return False
    if m == 1:
        return matrix
    else:
        MoM = _mofmin(matrix)
        MoC = []
        for i, row in enumerate(MoM):
            newrow = []
            for j, val in enumerate(row):
                newrow.append((-1**(i+j))*val)
            MoC.append(newrow)
        return MoC


def _transpose(matrix):
    m, n = len(matrix), len(matrix[0])
    T = []
    for i in range(n):
        newrow = []
        for j in range(m):
            newrow.append(matrix[i][j])
        T.append(newrow)
    return T


def _adjugate(matrix):
    '''Adjugate Matrix'''
    m, n = len(matrix), len(matrix[0])
    if m != n:
        return False
    if m == 1:
        return matrix
    else:
        MoC = _mofcof(matrix)
        adj = _transpose(MoC)
        return adj


def _conmul(matrix, c):
    '''Multiply Matrix by constant'''
    m, n = len(matrix), len(matrix[0])
    A = []
    for row in matrix:
        newrow = []
        for val in row:
            newrow.append(c*val)
        A.append(newrow)
    return A


def _matmul(A, B):
    '''Matrix Multiplication'''
    m1, n1 = len(A), len(A[0])
    m2, n2 = len(B), len(B[0])
    if n1 != m2:
        if n2 != m1:
            return False
        else:
            C = B
            B = A
            A = C
            m1, n1 = len(A), len(A[0])
            m2, n2 = len(B), len(B[0])
    C = []
    for i, row in enumerate(A):
        newrow = []
        for j, column in enumerate(B[0]):
            newval = 0
            for k, val in enumerate(row):
                newval += val*B[k]
            newrow.append(newval)
        C.append(newrow)
    return C


def _invert(matrix):
    '''Inverse Matrix'''
    adj = _adjugate(matrix)
    det = _determinant(matrix)
    if det:
        invdet = 1/det
        inv = _conmul(adj, invdet)
        return inv
    return False


def _syssolve(matrix, B):
    '''Solver of a system of linear equations'''
    A = _invert(matrix)
    if A:
        X = _matmul(A, B)
        return X
    return False


def _dot(u, v):
    '''Dot product'''
    total = 0
    for i in range(len(u)):
        total += u[i]*v[i]
    return total


def _offset(point1, point2):
    '''Vector subtraction'''
    offset = []
    for i in range(len(point1)):
        offset.append(point1[i]-point2[i])
    return offset


def trilaterate(pointdata):
    '''takes [distance,coordinate] pairs of the form [d,[x1,x2,x3,...,xn]]
    requires n+1 points minimum, uses first n+1 points given
    returns n dimensional coordinates of unknown point as array
    returns false if data is invalid (incorrectly formatted or unsolvable)
    returns false if first n+1 points are not unique and non-colinear'''
    m, n = len(pointdata), len(pointdata[0][1])
    if m < n+1:
        return False
    dsquareds = []
    qsquareds = []
    solvermat = []
    solutnmat = []
    k = (pointdata[n][0]**2)-_dot(pointdata[n][1], pointdata[n][1])
    t = pointdata[n][1]
    for i in range(n):
        dsquareds.append(pointdata[i][0]**2)
        qsquareds.append(_dot(pointdata[i][1], pointdata[i][1]))
        solvermat.append(_offset(pointdata[i][1], t))
        solutnmat.append(dsquareds[i]-qsquareds[i]-k)
    P = _syssolve(solvermat, solutnmat)
    return P
