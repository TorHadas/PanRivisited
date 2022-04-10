import sys, itertools, math
import numpy as np
from util.test import *
from util.save import *
from util.decomp import *
from util.log import *
from util.print import *
from util.utils import *
from util.terms import *
from alg_skeleton import get_mats

import code

ALGORITHM_NAME = "alg82revisit"


def get_fix_set(i, j, s, n):
    dmult = 1.0
    if i == j:
        dmult -= 9.0 / s
    i_, j_, _ = shift_indices(i, j, 0, s, n)
    terms = []

    if (s == 9 and i == j):
        # 2-4
        terms.append(([(s, i_[0], j_[0])],
                      [(1, i_[1], j_[0])],
                      [(1, i_[0], j_[1])]))
        terms.append(([(s, i_[0], j_[1])],
                      [(1, i_[0], j_[0])],
                      [(1, i_[1], j_[0])]))
        terms.append(([(-s, i_[1], j_[0])],
                      [(1, i_[0], j_[1])],
                      [(1, i_[0], j_[0])]))
        # 6-8
        terms.append(([(s, i_[1], j_[1])],
                      [(1, i_[0], j_[1])],
                      [(1, i_[1], j_[0])]))
        terms.append(([(s, i_[1], j_[0])],
                      [(1, i_[1], j_[1])],
                      [(1, i_[0], j_[1])]))
        terms.append(([(-s, i_[0], j_[1])],
                      [(1, i_[1], j_[0])],
                      [(1, i_[1], j_[1])]))
        return terms

    # M1
    terms.append(([(-s, i_[0], j_[0]), (-s * dmult, i_[1], j_[1])],
                  [(1, i_[0], j_[0]), (1, i_[1], j_[1])],
                  [(dmult, i_[0], j_[0]), (1, i_[1], j_[1])]))
    # M2
    terms.append(([(1, i_[0], j_[1]), (dmult, i_[1], j_[1])],
                  [(-s, i_[0], j_[0])],
                  [(-1, i_[1], j_[0]), (-1, i_[1], j_[1])]))
    # M3
    terms.append(([(-s, i_[0], j_[0])],
                  [(1, i_[1], j_[0]), (-1, i_[1], j_[1])],
                  [(-1, i_[0], j_[1]), (1, i_[1], j_[1])]))
    # M4
    terms.append(([(-s * dmult, i_[1], j_[1])],
                  [(1 / dmult, i_[0], j_[1]), (-1, i_[0], j_[0])],
                  [(dmult, i_[0], j_[0]), (-1, i_[1], j_[0])]))
    # M5
    terms.append(([(1, i_[0], j_[0]), (1, i_[1], j_[0])],
                  [(-s, i_[1], j_[1])],
                  [(-dmult, i_[0], j_[0]), (-1, i_[0], j_[1])]))
    # M6
    terms.append(([(1, i_[0], j_[1]), (-1, i_[0], j_[0])],
                  [(1, i_[0], j_[0]), (1, i_[1], j_[0])],
                  [(-s, i_[1], j_[1])]))
    # M7
    terms.append(([(1, i_[1], j_[0]), (-dmult, i_[1], j_[1])],
                  [(1 / dmult, i_[0], j_[1]), (1, i_[1], j_[1])],
                  [(-s * dmult, i_[0], j_[0])]))
    return terms


def get_fixes(n):
    s = n // 2
    terms = []
    for i, j in itertools.product(range(s), range(s)):
        tc_terms = get_fix_set(i, j, s, n)
        terms += tc_terms

    return terms


def get_fix_set_2(i_, j_, a_, b_, c_):
    terms = []

    # M1
    terms.append(([(a_[0][0], i_[0], j_[0]), (a_[1][1], i_[1], j_[1])],
                  [(b_[0][0], i_[0], j_[0]), (b_[1][1], i_[1], j_[1])],
                  [(c_[0][0], i_[0], j_[0]), (c_[1][1], i_[1], j_[1])]))
    # M2
    terms.append(([(a_[0][1], i_[0], j_[1]), (a_[1][1], i_[1], j_[1])],
                  [(b_[0][0], i_[0], j_[0])],
                  [(c_[1][0], i_[1], j_[0]), (-c_[1][1], i_[1], j_[1])]))
    # M3
    terms.append(([(a_[0][0], i_[0], j_[0])],
                  [(b_[1][0], i_[1], j_[0]), (-b_[1][1], i_[1], j_[1])],
                  [(c_[0][1], i_[0], j_[1]), (c_[1][1], i_[1], j_[1])]))
    # M4
    terms.append(([(a_[1][1], i_[1], j_[1])],
                  [(b_[0][1], i_[0], j_[1]), (-b_[0][0], i_[0], j_[0])],
                  [(c_[0][0], i_[0], j_[0]), (c_[1][0], i_[1], j_[0])]))
    # M5
    terms.append(([(a_[0][0], i_[0], j_[0]), (a_[1][0], i_[1], j_[0])],
                  [(b_[1][1], i_[1], j_[1])],
                  [(-c_[0][0], i_[0], j_[0]), (c_[0][1], i_[0], j_[1])]))
    # M6
    terms.append(([(a_[0][1], i_[0], j_[1]), (-a_[0][0], i_[0], j_[0])],
                  [(b_[0][0], i_[0], j_[0]), (b_[1][0], i_[1], j_[0])],
                  [(c_[1][1], i_[1], j_[1])]))
    # M7
    terms.append(([(a_[1][0], i_[1], j_[0]), (-a_[1][1], i_[1], j_[1])],
                  [(b_[0][1], i_[0], j_[1]), (b_[1][1], i_[1], j_[1])],
                  [(c_[0][0], i_[0], j_[0])]))

    return terms


def get_fix_set_2_straight(i_, j_, a_, b_, c_):
    terms = []
    terms.append(([(a_[0][0], i_[0], j_[0])],
                  [(b_[1][0], i_[1], j_[0])],
                  [(c_[0][1], i_[0], j_[1])]))

    terms.append(([(a_[0][1], i_[0], j_[1])],
                  [(b_[0][0], i_[0], j_[0])],
                  [(c_[1][0], i_[1], j_[0])]))

    terms.append(([(a_[1][0], i_[1], j_[0])],
                  [(b_[0][1], i_[0], j_[1])],
                  [(c_[0][0], i_[0], j_[0])]))

    terms.append(([(a_[1][1], i_[1], j_[1])],
                  [(b_[0][1], i_[0], j_[1])],
                  [(c_[1][0], i_[1], j_[0])]))

    terms.append(([(a_[1][0], i_[1], j_[0])],
                  [(b_[1][1], i_[1], j_[1])],
                  [(c_[0][1], i_[0], j_[1])]))

    terms.append(([(a_[0][1], i_[0], j_[1])],
                  [(b_[1][0], i_[1], j_[0])],
                  [(c_[1][1], i_[1], j_[1])]))

    return terms


def get_fixes_2(n):
    s = n // 2
    fix_const = -s
    a_ = np.ones((2, 2))
    b_ = np.ones((2, 2))
    c_ = np.ones((2, 2))

    for ind in range(2):
        a_[ind, ind] = - 1.0
        b_[ind, ind] = - 1.0

    org_ = np.array(a_), np.array(b_), np.array(c_)

    terms = []
    tc_terms = []
    for i, j in itertools.product(range(s), range(s)):
        i_, j_, _ = shift_indices(i, j, 0, s, n)
        a_, b_, c_ = np.array(org_[0]), np.array(org_[1]), np.array(org_[2])

        if s == 9 and i == j:
            tc_terms = get_fix_set_2_straight(i_, j_, a_, b_, c_)
        else:
            if i == j:
                x = 1.0 - 9.0 / s
                a_[1][1] *= x
                b_[0][1] *= 1 / x
                c_[0][0] *= x
            tc_terms = get_fix_set_2(i_, j_, a_, b_, c_)
        terms += terms_mult_by(tc_terms, fix_const)
    return terms


def stats_82revisit(n0, log=True):
    n = n0 + 2
    s = n // 2
    strassen = 7
    t = 2 * len(get_S_hat(s))
    t += 2 * s ** 3
    t += (strassen * s ** 2)
    if log:
        Logger.log("\t-> Projected: n0 = ", n0, " t = ", t, " w0 = ", math.log(t, n0))
    return t


def gen_terms_82revisit(n):
    s = n // 2
    t0_terms = []

    triplets = get_S_hat(s)
    t0_terms += get_table_1(dot_group(triplets, s, n), n)

    triplets = itertools.product(range(s), range(s), range(s))
    t0_terms += get_table_2(dot_group(triplets, s, n), n)

    tc_terms = get_fixes_2(n)
    all_terms = t0_terms + tc_terms
    return all_terms


def decomp_mats_82revisit(U, V, W, n0):
    t = U.shape[0]
    n = n0 + 2
    transf = gen_transf(n, t)
    tIn02 = transpose_W(np.identity(n0 * n0), n0)
    Us, Vs, Ws = [U, transf], [V, transf], [W, dot(transf, tIn02)]
    return Us, Vs, Ws


def test_82revisit(ss=[4, 8]):
    for s in ss:
        n = 2 * s
        get_mats(n, n + 2, ALGORITHM_NAME, gen_terms_82revisit, decomp_mats_82revisit, force_new=True)


if __name__ == "__main__":
    test_82revisit()
