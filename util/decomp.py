# alg 4
import sympy
import scipy.io
import sys, itertools, math
import numpy as np
import scipy, sys, itertools, math, os, random
from os.path import join, basename
import scipy.sparse
import inspect
from util.save import *
from util.utils import *
from util.log import *


def first_decompose(mat, dup):
    # Logger.log(type(dup))
    # Logger.log(dup)
    t, n2 = mat.shape
    r = t + len(dup) - sum([dup[i].shape[0] for i in range(len(dup))])
    phi = np.zeros((r, n2))
    mat_phi = np.zeros((t, r))
    done = []
    for i in range(len(dup)):
        phi[i, :] = mat[dup[i][0]]
        for j in dup[i]:
            mat_phi[j, i] = 1
        done += list(dup[i])
    i = len(dup)
    for j in range(t):
        if j in done:
            continue
        mat_phi[j, i] = 1
        phi[i, :] = mat[j, :]
        i += 1
    return mat_phi, phi


def decomp_dup_rows(P):
    # Logger.log("\t" + str(P.shape))
    # Logger.log("\t" + str(len(P)))
    # return old_decomp_dup_rows(P)
    # Logger.log("\tcalculating unique")
    rows, indices, inverse_indices, counts = np.unique(np.concatenate((P, -P)), axis=0, return_index=True,
                                                       return_inverse=True, return_counts=True)
    # Logger.log("\tentering loop")
    dup_indices = []
    tmp = []
    check = 0
    dups = 0
    inverse_indices_pos = inverse_indices[0:len(P)]
    inverse_indices_neg = inverse_indices[len(P):]
    for i in range(rows.shape[0]):
        v = counts[i]
        check += v
        if (indices[i] > len(P) or v < 2):
            continue
        b = np.sort(np.concatenate((np.where(inverse_indices_pos == i)[0], np.where(inverse_indices_neg == i)[0])))
        c = False
        for a in dup_indices:
            if (len(a) == len(b)):
                if (a == b).all():
                    c = True
                    break
        if c:
            continue
        # Logger.log(v)
        # Logger.log(b)
        dups += v - 1
        dup_indices.append(b)

    dup_matrixes = []
    for i in np.unique([len(v) for v in dup_indices]):
        dup_matrixes.append(np.unique([v for v in dup_indices if len(v) == i], axis=0))
    dup_indices = [v for V in dup_matrixes for v in V]
    if check / 2 != P.shape[0]:
        Logger.log("ERROR: decomp_dup_rows wrong number of rows:")
        Logger.log(check)
    return dups, dup_indices
