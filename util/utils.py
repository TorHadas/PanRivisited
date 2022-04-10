import itertools
import numpy as np


def dot(*args):
    ans = 0
    first = True
    for arg in args:
        if first:
            ans = arg
            first = False
        else:
            ans = np.dot(ans, arg)
    return ans


def nnz(U, rows=[]):
    if (len(rows) < 1):
        return len(np.nonzero(np.abs(U) > 0.0001)[0])
    return len(np.nonzero(np.abs(U[rows, :]) > 0.0001)[0])


def nns(U, rows=[]):
    if (len(U.shape) < 2):
        m = 1
        n = U.shape[0]
    else:
        n, m = U.shape

    if (len(rows) > 0):
        n = len(rows);

    zeros = n * m - nnz(U, rows)
    positive = n * m - nnz(U - 1, rows)
    negative = n * m - nnz(U + 1, rows)

    return n * m - zeros - positive - negative


def bar_i(i, s, n):
    return (i + s) % n


def bar_tr(tr, s, n):
    return (bar_i(tr[0], s, n), bar_i(tr[1], s, n), bar_i(tr[2], s, n))


def dot_group(S, s, n):
    S_dot = []
    for (i, j, k) in S:
        triplets = [
            (i, j, k),
            bar_tr((i, j, k), s, n),
            bar_tr(bar_tr((i, j, k), s, n), s, n)
        ]
        for tr in triplets:
            if (tr not in S_dot):
                S_dot.append(tr)
    return S_dot


def get_S_hat(s):
    S_hat = []
    for (i, j, k) in itertools.product(range(s), range(s), range(s)):
        if i <= j < k or i >= j > k:
            S_hat.append((i, j, k))
    return S_hat


def shift_indices(i, j, k, s, n):
    i_ = [(i + s * ind) % n for ind in range(2)]
    j_ = [(j + s * ind) % n for ind in range(2)]
    k_ = [(k + s * ind) % n for ind in range(2)]
    return i_, j_, k_


def undecomp_mats(Us, Vs, Ws):
    U, V, W = Us[0], Vs[0], Ws[0]
    for i in range(1, len(Us)):
        U = dot(U, Us[i])
    for i in range(1, len(Vs)):
        V = dot(V, Vs[i])
    for i in range(1, len(Ws)):
        W = dot(W, Ws[i])
    return U, V, W


def gen_transf(n, t):
    s = n // 2
    n0 = n - 2
    s0 = n0 // 2

    transf = np.zeros((n * n, n0 * n0))
    full = np.zeros((2, 2, n0 * n0))
    rows = np.zeros((2, 2, s0, n0 * n0))
    cols = np.zeros((2, 2, s0, n0 * n0))
    for i in range(s0):
        for j in range(s0):
            for li in range(2):
                for lj in range(2):
                    rows[li, lj, i, (s0 * li + i) * n0 + s0 * lj + j] = 1
                    cols[li, lj, i, (s0 * li + j) * n0 + s0 * lj + i] = 1
                    full[li, lj, (s0 * li + i) * n0 + s0 * lj + j] = 1

    for i, j in itertools.product(range(n), range(n)):
        scalar_over_s0 = 1.0 / (s0 + 1)
        z0, z1 = 0, 0
        # Logger.log(str((i, j)) + ", scalar = " + str(scalar_over_s0) + "->")
        while (i > s0):
            z0 += 1
            i -= s0 + 1
        while (j > s0):
            z1 += 1
            j -= s0 + 1
        # Logger.log("\t" + str((i, j, z0, z1)))
        if (i == 0):
            if (j == 0):
                transf[(i + s * z0) * n + j + s * z1, :] += full[z0, z1, :] * scalar_over_s0
            else:
                transf[(i + s * z0) * n + j + s * z1, :] -= cols[z0, z1, j - 1, :] * scalar_over_s0
        elif (j == 0):
            transf[(i + s * z0) * n + j + s * z1, :] += full[z0, z1, :] * scalar_over_s0
            transf[(i + s * z0) * n + j + s * z1, :] -= rows[z0, z1, i - 1, :] * 1.0
        else:
            transf[(i + s * z0) * n + j + s * z1, (i - 1 + s0 * z0) * n0 + (j - 1 + s0 * z1)] += 1.0
            transf[(i + s * z0) * n + j + s * z1, :] -= cols[z0, z1, j - 1, :] * scalar_over_s0
    return transf


def transpose_W(w: np.ndarray, n):
    W = w.copy()
    for i in range(n):
        for j in range(n):
            if (i >= j):
                continue
            # Logger.log(W[:, i*n + j])
            # Logger.log(W[:, j*n + i])
            x = np.array(W[:, i * n + j])
            y = np.array(W[:, j * n + i])
            W[:, i * n + j] = y
            W[:, j * n + i] = x
            # Logger.log(W[:, i*n + j])
    return W
