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

ALGORITHM_NAME = "alg80revisit"


def gen_t19_terms(const, i_, j_, s, n):
    delta_ij = delta(i_[0], j_[0])
    terms = []
    s_minus_delta = (s - delta_ij)
    # Term 1 (+2nd in 22)
    a_terms = [(const, i_[0], j_[0]), (const, i_[0], j_[1])]
    b_terms = [(1.0, i_[0], j_[0]), (-1.0, i_[1], j_[0])]
    c_terms = []
    c_terms.append((+ 0.5 * s_minus_delta, i_[0], j_[0]))
    c_terms.append((- 0.5 * s_minus_delta, i_[1], j_[0]))
    c_terms.append((+ 0.5 * s_minus_delta, i_[0], j_[1]))
    c_terms.append((- 0.5 * s_minus_delta, i_[1], j_[1]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        c_terms.append((1.0, k_[0], i_[0]))
        c_terms.append((-1.0, k_[1], i_[0]))
        c_terms.append((1.0, j_[0], k_[0]))
        c_terms.append((1.0, j_[0], k_[1]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 2 (+4th in 22)
    a_terms = [(const, i_[1], j_[1]), (const, i_[1], j_[0])]
    b_terms = [(1.0, i_[1], j_[1]), (-1.0, i_[0], j_[1])]
    c_terms = []
    c_terms.append((+ 0.5 * s_minus_delta, i_[1], j_[1]))
    c_terms.append((- 0.5 * s_minus_delta, i_[0], j_[1]))
    c_terms.append((+ 0.5 * s_minus_delta, i_[1], j_[0]))
    c_terms.append((- 0.5 * s_minus_delta, i_[0], j_[0]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        c_terms.append((1.0, k_[1], i_[1]))
        c_terms.append((-1.0, k_[0], i_[1]))
        c_terms.append((1.0, j_[1], k_[1]))
        c_terms.append((1.0, j_[1], k_[0]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 3 (+1st in 22, +2nd in 28)
    a_terms = [(const, i_[0], j_[0]), (-const, i_[0], j_[1])]
    b_terms = [(1.0, i_[0], j_[0]), (-1.0, i_[1], j_[0])]
    c_terms = []
    c_terms.append((- delta_ij, i_[0], j_[0]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        c_terms.append((1.0, j_[0], k_[0]))
        c_terms.append((-1.0, j_[0], k_[1]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 4 (+3rd in 22, +4th in 28)
    a_terms = [(const, i_[1], j_[1]), (-const, i_[1], j_[0])]
    b_terms = [(1.0, i_[1], j_[1]), (-1.0, i_[0], j_[1])]
    c_terms = []
    c_terms.append((- delta_ij, i_[1], j_[1]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        c_terms.append((1.0, j_[1], k_[1]))
        c_terms.append((-1.0, j_[1], k_[0]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 5 (+1st in 28)
    a_terms = [(const, i_[0], j_[0]), (const, i_[0], j_[1])]
    b_terms = [(1.0, i_[0], j_[0]), (1.0, i_[1], j_[0])]
    c_terms = []
    c_terms.append((+ 0.5 * s_minus_delta, i_[0], j_[0]))
    c_terms.append((- 0.5 * s_minus_delta, i_[1], j_[0]))
    c_terms.append((- 0.5 * s_minus_delta, i_[0], j_[1]))
    c_terms.append((+ 0.5 * s_minus_delta, i_[1], j_[1]))
    c_terms.append((- delta_ij, i_[0], j_[0]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        c_terms.append((1.0, k_[0], i_[0]))
        c_terms.append((1.0, k_[1], i_[0]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 6 (+3rd in 28)
    a_terms = [(const, i_[1], j_[1]), (const, i_[1], j_[0])]
    b_terms = [(1.0, i_[1], j_[1]), (1.0, i_[0], j_[1])]
    c_terms = []
    c_terms.append((+ 0.5 * s_minus_delta, i_[1], j_[1]))
    c_terms.append((- 0.5 * s_minus_delta, i_[0], j_[1]))
    c_terms.append((- 0.5 * s_minus_delta, i_[1], j_[0]))
    c_terms.append((+ 0.5 * s_minus_delta, i_[0], j_[0]))
    c_terms.append((- delta_ij, i_[1], j_[1]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        c_terms.append((1.0, k_[1], i_[1]))
        c_terms.append((1.0, k_[0], i_[1]))
    terms.append((a_terms, b_terms, c_terms))
    return terms


def gen_t24_terms(const, i_, j_, s, n):
    delta_ij = delta(i_[0], j_[0])
    terms = []
    s_minus_delta = (s - delta_ij)
    # Term 1 (+1st in 23)
    b_terms = [(const, i_[0], j_[0]), (-const, i_[0], j_[1])]
    c_terms = [(1.0, i_[0], j_[0]), (1.0, i_[1], j_[0])]
    a_terms = []
    a_terms.append((+ 0.5 * s_minus_delta, i_[0], j_[0]))
    a_terms.append((- 0.5 * s_minus_delta, i_[0], j_[1]))
    a_terms.append((+ 0.5 * s_minus_delta, i_[1], j_[1]))
    a_terms.append((- 0.5 * s_minus_delta, i_[1], j_[0]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        a_terms.append((1.0, k_[0], i_[0]))
        a_terms.append((-1.0, k_[1], i_[0]))
        a_terms.append((1.0, j_[0], k_[0]))
        a_terms.append((-1.0, j_[0], k_[1]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 2 (+3rd in 23)
    b_terms = [(const, i_[1], j_[1]), (-const, i_[1], j_[0])]
    c_terms = [(1.0, i_[1], j_[1]), (1.0, i_[0], j_[1])]
    a_terms = []
    a_terms.append((+ 0.5 * s_minus_delta, i_[1], j_[1]))
    a_terms.append((- 0.5 * s_minus_delta, i_[0], j_[1]))
    a_terms.append((- 0.5 * s_minus_delta, i_[1], j_[0]))
    a_terms.append((+ 0.5 * s_minus_delta, i_[0], j_[0]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        a_terms.append((1.0, k_[1], i_[1]))
        a_terms.append((-1.0, k_[0], i_[1]))
        a_terms.append((1.0, j_[1], k_[1]))
        a_terms.append((-1.0, j_[1], k_[0]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 3 (+2nd in 30)
    b_terms = [(const, i_[0], j_[0]), (-const, i_[0], j_[1])]
    c_terms = [(1.0, i_[0], j_[0]), (-1.0, i_[1], j_[0])]
    a_terms = []
    a_terms.append((- delta_ij, i_[0], j_[0]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        a_terms.append((1.0, k_[0], i_[0]))
        a_terms.append((1.0, k_[1], i_[0]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 4 (+4th in 30)
    b_terms = [(const, i_[1], j_[1]), (-const, i_[1], j_[0])]
    c_terms = [(1.0, i_[1], j_[1]), (-1.0, i_[0], j_[1])]
    a_terms = []
    a_terms.append((- delta_ij, i_[1], j_[1]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        a_terms.append((1.0, k_[1], i_[1]))
        a_terms.append((1.0, k_[0], i_[1]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 5 (+2nd in 23, +1st in 30)
    b_terms = [(const, i_[0], j_[0]), (const, i_[0], j_[1])]
    c_terms = [(1.0, i_[0], j_[0]), (1.0, i_[1], j_[0])]
    a_terms = []
    a_terms.append((+ 0.5 * s_minus_delta, i_[0], j_[0]))
    a_terms.append((- 0.5 * s_minus_delta, i_[0], j_[1]))
    a_terms.append((- 0.5 * s_minus_delta, i_[1], j_[1]))
    a_terms.append((+ 0.5 * s_minus_delta, i_[1], j_[0]))
    a_terms.append((- delta_ij, i_[0], j_[0]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        a_terms.append((1.0, j_[0], k_[0]))
        a_terms.append((1.0, j_[0], k_[1]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 6 (+4th in 23, +3rd in 30)
    b_terms = [(const, i_[1], j_[1]), (const, i_[1], j_[0])]
    c_terms = [(1.0, i_[1], j_[1]), (1.0, i_[0], j_[1])]
    a_terms = []
    a_terms.append((- 0.5 * s_minus_delta, i_[0], j_[0]))
    a_terms.append((+ 0.5 * s_minus_delta, i_[0], j_[1]))
    a_terms.append((+ 0.5 * s_minus_delta, i_[1], j_[1]))
    a_terms.append((- 0.5 * s_minus_delta, i_[1], j_[0]))
    a_terms.append((- delta_ij, i_[1], j_[1]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        a_terms.append((1.0, j_[1], k_[1]))
        a_terms.append((1.0, j_[1], k_[0]))
    terms.append((a_terms, b_terms, c_terms))
    return terms


def gen_squared_terms(const, i_, j_, s, n):
    delta_ij = delta(i_[0], j_[0])
    terms = []
    s_minus_delta = (s - delta_ij)
    # Term 1
    c_terms = [(const, i_[0], j_[0]), (-const, i_[0], j_[1])]
    a_terms = [(1.0, i_[0], j_[0]), (1.0, i_[1], j_[0])]
    b_terms = []
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        b_terms.append((1.0, k_[0], i_[0]))
        b_terms.append((1.0, k_[1], i_[0]))
        b_terms.append((1.0, j_[0], k_[0]))
        b_terms.append((1.0, j_[0], k_[1]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 2
    c_terms = [(const, i_[1], j_[1]), (-const, i_[1], j_[0])]
    a_terms = [(1.0, i_[1], j_[1]), (1.0, i_[0], j_[1])]
    b_terms = []
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        b_terms.append((1.0, k_[1], i_[1]))
        b_terms.append((1.0, k_[0], i_[1]))
        b_terms.append((1.0, j_[1], k_[1]))
        b_terms.append((1.0, j_[1], k_[0]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 3 (+2nd in 29)
    c_terms = [(const, i_[0], j_[0]), (-const, i_[0], j_[1])]
    a_terms = [(1.0, i_[0], j_[0]), (-1.0, i_[1], j_[0])]
    b_terms = []
    b_terms.append((- delta_ij, i_[0], j_[0]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        b_terms.append((1.0, k_[0], i_[0]))
        b_terms.append((-1.0, k_[1], i_[0]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 4 (+4th in 29)
    c_terms = [(const, i_[1], j_[1]), (-const, i_[1], j_[0])]
    a_terms = [(1.0, i_[1], j_[1]), (-1.0, i_[0], j_[1])]
    b_terms = []
    b_terms.append((- delta_ij, i_[1], j_[1]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        b_terms.append((1.0, k_[1], i_[1]))
        b_terms.append((-1.0, k_[0], i_[1]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 5 (+1st in 29)
    c_terms = [(const, i_[0], j_[0]), (const, i_[0], j_[1])]
    a_terms = [(1.0, i_[0], j_[0]), (1.0, i_[1], j_[0])]
    b_terms = []
    b_terms.append((- delta_ij, i_[0], j_[0]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        b_terms.append((1.0, j_[0], k_[0]))
        b_terms.append((-1.0, j_[0], k_[1]))
    terms.append((a_terms, b_terms, c_terms))

    # Term 6 (+3rd in 29)
    c_terms = [(const, i_[1], j_[1]), (const, i_[1], j_[0])]
    a_terms = [(1.0, i_[1], j_[1]), (1.0, i_[0], j_[1])]
    b_terms = []
    b_terms.append((- delta_ij, i_[1], j_[1]))
    for k in rangestar(s, i_[0], j_[0]):
        _, _, k_ = shift_indices(0, 0, k, s, n)
        b_terms.append((1.0, j_[1], k_[1]))
        b_terms.append((-1.0, j_[1], k_[0]))
    terms.append((a_terms, b_terms, c_terms))
    return terms


def delta(i, j):
    return 1.0 if i == j else 0.0


def gen_t1_terms(const, i_, j_, s, n):
    delta_ij = -delta(i_[0], j_[0])
    terms = []
    d_ij = 0.5 * (s - delta(i_[0], j_[0]))

    k_range = range(s)
    if i_[0] >= s:
        k_range = range(s, n)
    if i_[0] == j_[0]:
        k_range = list(k for k in k_range if k != i_[0])

    term_1_sum, term_2_sum, term_3_sum = [], [], []
    for k in k_range:
        _, _, k_ = shift_indices(0, 0, k, s, n)
        term_1_sum += [
            (1.0, k_[0], i_[0]),
            (-1.0, k_[1], i_[0]),
            (1.0, j_[0], k_[0]),
            (-1.0, j_[0], k_[1])
        ]

        term_2_sum += [
            (1.0, k_[0], i_[0]),
            (1.0, k_[1], i_[0]),
        ]

        term_3_sum += [
            (1.0, j_[0], k_[0]),
            (1.0, j_[0], k_[1])
        ]

    term_1 = (
        [
            (+ d_ij, i_[0], j_[0]),
            (- d_ij, i_[0], j_[1]),
            (+ d_ij, i_[1], j_[1]),
            (- d_ij, i_[1], j_[0])
        ] + term_1_sum,
        [(1.0, i_[0], j_[0]), (-1.0, i_[0], j_[1])],
        [(1.0, i_[0], j_[0]), (1.0, i_[1], j_[0])]
    )

    term_2 = (
        [
            (delta_ij, i_[0], j_[0])
        ] + term_2_sum,
        [(1.0, i_[0], j_[0]), (-1.0, i_[0], j_[1])],
        [(1.0, i_[0], j_[0]), (-1.0, i_[1], j_[0])]
    )

    term_3 = (
        [
            (delta_ij, i_[0], j_[0]),
            #
            (+ d_ij, i_[0], j_[0]),
            (- d_ij, i_[0], j_[1]),
            (- d_ij, i_[1], j_[1]),
            (+ d_ij, i_[1], j_[0])

        ] + term_3_sum,
        [(1.0, i_[0], j_[0]), (1.0, i_[0], j_[1])],
        [(1.0, i_[0], j_[0]), (1.0, i_[1], j_[0])]
    )

    return [term_1, term_2, term_3]


def gen_t2_terms(const, i_, j_, s, n):
    delta_ij = -delta(i_[0], j_[0])
    terms = []
    d_ij = 0.5 * (s - delta(i_[0], j_[0]))

    k_range = range(s)
    if i_[0] >= s:
        k_range = range(s, n)
    if i_[0] == j_[0]:
        k_range = list(k for k in k_range if k != i_[0])

    term_1_sum, term_2_sum, term_3_sum = [], [], []
    for k in k_range:
        _, _, k_ = shift_indices(0, 0, k, s, n)
        term_1_sum += [
            (1.0, k_[0], i_[0]),
            (1.0, k_[1], i_[0]),
            (1.0, j_[0], k_[0]),
            (1.0, j_[0], k_[1])
        ]

        term_2_sum += [
            (1.0, k_[0], i_[0]),
            (-1.0, k_[1], i_[0]),
        ]

        term_3_sum += [
            (1.0, j_[0], k_[0]),
            (-1.0, j_[0], k_[1])
        ]

    term_1 = (
        [(1.0, i_[0], j_[0]), (1.0, i_[1], j_[0])],
        term_1_sum,
        [(1.0, i_[0], j_[0]), (-1.0, i_[0], j_[1])]
    )

    term_2 = (
        [(1.0, i_[0], j_[0]), (-1.0, i_[1], j_[0])],
        [
            (delta_ij, i_[0], j_[0])
        ] + term_2_sum,
        [(1.0, i_[0], j_[0]), (-1.0, i_[0], j_[1])]
    )

    term_3 = (
        [(1.0, i_[0], j_[0]), (1.0, i_[1], j_[0])],
        [
            (delta_ij, i_[0], j_[0])
        ] + term_3_sum,
        [(1.0, i_[0], j_[0]), (1.0, i_[0], j_[1])]
    )

    return [term_1, term_2, term_3]


def gen_t3_terms(const, i_, j_, s, n):
    delta_ij = -delta(i_[0], j_[0])
    terms = []
    d_ij = 0.5 * (s - delta(i_[0], j_[0]))

    k_range = range(s)
    if i_[0] >= s:
        k_range = range(s, n)
    if i_[0] == j_[0]:
        k_range = list(k for k in k_range if k != i_[0])

    term_1_sum, term_2_sum, term_3_sum = [], [], []
    for k in k_range:
        _, _, k_ = shift_indices(0, 0, k, s, n)
        term_1_sum += [
            (1.0, k_[0], i_[0]),
            (-1.0, k_[1], i_[0]),
            (1.0, j_[0], k_[0]),
            (1.0, j_[0], k_[1])
        ]

        term_2_sum += [
            (1.0, k_[0], i_[0]),
            (1.0, k_[1], i_[0]),
        ]

        term_3_sum += [
            (1.0, j_[0], k_[0]),
            (-1.0, j_[0], k_[1])
        ]

    term_1 = (
        [(1.0, i_[0], j_[0]), (1.0, i_[0], j_[1])],
        [(1.0, i_[0], j_[0]), (-1.0, i_[1], j_[0])],
        [
            (+ d_ij, i_[0], j_[0]),
            (- d_ij, i_[1], j_[0]),
            (+ d_ij, i_[0], j_[1]),
            (- d_ij, i_[1], j_[1])
        ] + term_1_sum
    )

    term_2 = (
        [(1.0, i_[0], j_[0]), (1.0, i_[0], j_[1])],
        [(1.0, i_[0], j_[0]), (+1.0, i_[1], j_[0])],
        [
            (delta_ij, i_[0], j_[0]),
            #
            (+ d_ij, i_[0], j_[0]),
            (- d_ij, i_[1], j_[0]),
            (- d_ij, i_[0], j_[1]),
            (+ d_ij, i_[1], j_[1])
        ] + term_2_sum
    )

    term_3 = (
        [(1.0, i_[0], j_[0]), (-1.0, i_[0], j_[1])],
        [(1.0, i_[0], j_[0]), (-1.0, i_[1], j_[0])],
        [
            (delta_ij, i_[0], j_[0])
        ] + term_3_sum
    )

    return [term_1, term_2, term_3]


def get_fixes_80(n):
    s = n // 2
    const = -0.5

    # everything is minus at A terms because it's T1,T2,T3
    t1_terms = []
    t2_terms = []
    t3_terms = []
    triplets = list(itertools.product(range(s), range(s))) + list(itertools.product(range(s, n), range(s, n)))
    for i, j in triplets:
        i_, j_, _ = shift_indices(i, j, 0, s, n)
        t1_terms += gen_t1_terms(const, i_, j_, s, n)  # Generating the TABLE19 terms, plus table 22 and 28
        t2_terms += gen_t2_terms(const, i_, j_, s, n)  # Generating the TABLE24 terms, plus table 23 and 30
        t3_terms += gen_t3_terms(const, i_, j_, s, n)  # Generating the TABLE24 terms, plus table 29

    # Generating the Tiii terms, IS IT MINUS????
    tiii_terms = [([(-2.0, i, i)], [(1.0, i, i)], [(1.0, i, i)]) for i in range(n)]

    return terms_mult_by(t1_terms + t2_terms + t3_terms, const) + tiii_terms


def stats_80revisit(n0, log=True):
    s = n0 // 2
    t = 2 * len(get_S_hat(s))
    t += 2 * (s ** 3 - s)
    t += 18 * s ** 2
    t += 2 * s
    if log:
        Logger.log("\t-> Projected: n0 = ", n0, " t = ", t, " w0 = ", math.log(t, n0))
    return t


def gen_terms_80revisit(n):
    s = n // 2
    t0_terms = []

    triplets = get_S_hat(s)
    t0_terms += get_table_1(dot_group(triplets, s, n), n)

    triplets = [(i, j, k) for i, j, k in itertools.product(range(s), range(s), range(s)) if (i != j or j != k)]
    t0_terms += get_table_2(dot_group(triplets, s, n), n)

    tc_terms = get_fixes_80(n)
    all_terms = t0_terms + tc_terms
    # print_sterms(all_terms, n)
    return all_terms


def decomp_mats_80revisit(U, V, W, n0):
    r_u, dup_u = decomp_dup_rows(U)
    Us = first_decompose(U, dup_u)
    del U

    r_v, dup_v = decomp_dup_rows(V)
    Vs = first_decompose(V, dup_v)
    del V

    r_w, dup_w = decomp_dup_rows(W)
    Ws = first_decompose(W, dup_w)
    del W

    return Us, Vs, Ws


def test_80revisit(ss=[4, 8]):
    for s in ss:
        n = 2 * s
        stats_80revisit(n)
        get_mats(n, n, ALGORITHM_NAME, gen_terms_80revisit, decomp_mats_80revisit, force_new=True, trans_W=False)


if __name__ == "__main__":
    test_80revisit()
