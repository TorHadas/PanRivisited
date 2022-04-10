import sys, itertools, math
import numpy as np
import scipy, sys, itertools, math, os, random
from os.path import join, basename
import scipy.sparse
import scipy.linalg
import inspect
from util.save import *
from util.decomp import *
from util.utils import *
from util.log import *


def check_triple_product(u, v, w, nmk, t):
    '''
    :brief Checks the Triple Product Condition for the given triplet U,V,W.
    :param u The first encoding matrix.
    :param v The second encoding matrix.
    :param w The decoding matrix.
    :param n,m,k The dimensions of the input matrices, <n,m,k>
    :param t The number of scalar multiplications.
    '''
    n = m = k = 1
    if type(nmk) == tuple:
        n, m, k = nmk
    else:
        n = m = k = nmk

    # Calculating the resulting tensor
    tensor = np.kron(np.kron(np.array(u[0, :]), np.array(v[0, :])), np.array(w[0, :]))
    for r in range(1, t):
        # if r % 10 == 0:
        #    Logger.log("'", newline=False)
        if r % 100 == 0:
            Logger.log(r, " / ", t)
        tensor += np.kron(np.kron(np.array(u[r, :]), np.array(v[r, :])), np.array(w[r, :]))
    Logger.log("'")
    tensor = tensor.reshape((n * m, m * k, n * k))

    # Calculating the target tensor
    target_tensor = np.zeros((n * m, m * k, n * k))
    for i1 in range(n):
        for k1 in range(m):
            for j1 in range(k):
                target_tensor[i1 * m + k1, k1 * k + j1, i1 * k + j1] = 1.0

    # Comparing both tensors
    return np.all(np.isclose(tensor, target_tensor))


def mult_rand_mat_test(U, V, W, n, times=20):
    for i in range(times):
        if not rand_mat_test(U, V, W, n):
            return False
    return True


def rand_mat_test(U, V, W, n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    C = np.dot(A, B)
    A_flat = np.reshape(A, (n * n, 1))
    B_flat = np.reshape(B, (n * n, 1))
    UA_VB = np.dot((U), A_flat) * np.dot((V), B_flat)
    C2 = np.dot(np.transpose(W), UA_VB)
    return not np.any(np.abs(C - np.reshape(C2, (n, n))) > 0.0001)


def get_uvw_q(U, V, W):
    Q_u = (nnz(U) + nns(U) - U.shape[0] + 0.0)
    Q_v = (nnz(V) + nns(V) - V.shape[0] + 0.0)
    Q_w = (nnz(W) + nns(W) - W.shape[1] + 0.0)
    if (U.shape[0] > U.shape[1]):
        Q_u /= (U.shape[0] - U.shape[1])
    else:
        Q_u /= U.shape[0]

    if (V.shape[0] > V.shape[1]):
        Q_v /= (V.shape[0] - V.shape[1])
    else:
        Q_v /= V.shape[0]

    if (W.shape[0] > W.shape[1]):
        Q_w /= (W.shape[0] - W.shape[1])
    else:
        Q_w /= W.shape[0]

    # Logger.log(Q_u, Q_v, Q_w)
    return Q_u, Q_v, Q_w


def get_all_orders(Us, Vs, Ws):
    t = Us[0].shape[0]
    n = math.sqrt(Us[-1].shape[1])
    orders = []

    q_u_old, q_v_old, q_w_old = 0, 0, 0
    orders.append((1, math.log(t, n)))
    for i in range(0, len(Us)):
        U, V, W = Us[i], Vs[i], Ws[i]
        q_u, q_v, q_w = get_uvw_q(U, V, W)

        orders.append((q_u - q_u_old, math.log(U.shape[0], n)))
        orders.append((q_v - q_v_old, math.log(V.shape[0], n)))
        orders.append((q_w - q_w_old, math.log(W.shape[0], n)))
        q_u_old, q_v_old, q_w_old = q_u, q_v, q_w
    a2 = - (q_u_old + q_v_old + q_w_old)
    orders.append((a2, 2.0))

    o = []
    for j in orders:
        o.append(j[1])

    o = np.array(o)
    o.sort()
    i = 0
    while i < len(o) - 1:
        if o[i] + 0.0001 > o[i + 1]:
            o = np.concatenate((o[:i + 1], o[i + 2:]))
        else:
            i += 1
    orders_sorted = []
    for j in o:
        orders_sorted.append([j, 0.0])
    # Logger.log(orders)
    for i in range(len(orders_sorted)):
        for k in orders:
            if (np.abs(k[1] - orders_sorted[i][0]) < 0.0001):
                orders_sorted[i][1] += k[0]
    orders_sorted.reverse()
    return orders_sorted
