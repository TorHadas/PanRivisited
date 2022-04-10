import sys, itertools, math
import numpy as np
import scipy, sys, itertools, math, os, random
from os.path import join, basename
import scipy.sparse
import inspect
import sympy as sym
from util.test import *
from util.log import *


def print_orders(orders):
    for J in orders:
        if np.isclose(J[1], 0):
            continue
        if J[1] > 0:
            Logger.log(' + ', newline=False)
        else:
            Logger.log(' - ', newline=False)
        Logger.log(np.abs(J[1]), newline=False)
        Logger.log("* n ^ (", newline=False)
        Logger.log(J[0], newline=False)
        Logger.log(")", newline=False)
    Logger.log("")


def test_mats(U, V, W, n0):
    Logger.log("Checking:")
    # Logger.log(Logger.get_char())
    Logger.log("\t-> checking random matrix...")
    # Calculating the product of two matrices using U, V, W
    if not mult_rand_mat_test(U, V, W, n0):
        Logger.log_fail("Failed Random Matrix Test")
        return False

    Logger.log_pass("Passed Random Matrix Test")
    # print_passed("Passed Random Matrix Test!")
    Logger.log("\t-> checking triple product...")
    # Making sure the matrices satisfy the triple product condition
    if (n0 > 12):
        Logger.log("Triple Product: didn't check due to size")
    elif not check_triple_product(U, V, W, n0, U.shape[0]):
        Logger.log_fail("Failed Triple Product Condition")
        return False
    else:
        Logger.log_pass("Passed Triple Product Condition")
    return True


def print_sterms(terms, n):
    a_vars, b_vars, c_vars = {}, {}, {}
    for (i, j) in itertools.product(range(n), range(n)):
        a_vars[i, j] = sym.Symbol('a_%d_%d' % (i, j))
        b_vars[i, j] = sym.Symbol('b_%d_%d' % (i, j))
        c_vars[i, j] = sym.Symbol('c_%d_%d' % (i, j))
    expr = sym.Rational(0, 1)

    G = np.zeros((n, n, n, n, n, n))

    for term in terms:
        for a in term[0]:
            for b in term[1]:
                for c in term[2]:
                    G[a[1], a[2], b[1], b[2], c[1], c[2]] += a[0] * b[0] * c[0]
    Logger.log("'")
    for i1, i2, j1, j2, k1, k2 in itertools.product(range(n), range(n), range(n), range(n), range(n), range(n)):
        x = G[i1, i2, j1, j2, k1, k2]
        if (np.abs(x) > 0.001):
            sign = "+" if x > 0 else "-"
            Logger.log(sign + " " + str(np.abs(x)) + ' * a_%d_%d * b_%d_%d * c_%d_%d' % (i1, i2, j1, j2, k1, k2),
                       newline=False)
    Logger.log("\n")
    # expr = sym.simplify(expr)
    # Logger.log(expr)
