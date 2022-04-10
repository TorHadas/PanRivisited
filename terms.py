from util.utils import *


def terms_mult_by(terms, mult_const):
    new_terms = []
    for term in terms:
        factor_len = max([len(term[i]) for i in range(3)])
        factor_ind = [i for i in range(3) if len(term[i]) == factor_len][0]  # RANDOM?
        new_term = []
        for i in range(3):
            factor = term[i]
            new_factor = []
            if (i == factor_ind):
                for element in factor:
                    con, ind1, ind2 = element
                    con *= mult_const
                    new_factor.append((con, ind1, ind2))
            else:
                new_factor = factor
            new_term.append(new_factor)
        new_term = tuple(new_term)
        new_terms.append(new_term)
    return new_terms


def get_table_2(triplets, n):
    c = -1
    a = -1
    terms = []
    for i, j, k in triplets:
        i_, j_, k_ = shift_indices(i, j, k, n // 2, n)
        terms.append(
            ([(a, i_[0], j_[0]), (1, j_[1], k_[0]), (1, k_[0], i_[1])],
             [(1, j_[0], k_[1]), (1, k_[0], i_[0]), (1, i_[1], j_[0])],
             [(c, k_[1], i_[0]), (1, i_[0], j_[1]), (1, j_[0], k_[0])]
             )
        )
    return terms


def get_table_1(triplets, n):
    terms = []
    for i, j, k in triplets:
        terms.append(
            ([(1, i, j), (1, j, k), (1, k, i)],
             [(1, j, k), (1, k, i), (1, i, j)],
             [(1, k, i), (1, i, j), (1, j, k)]
             )
        )
    return terms


def terms_to_mats(all_terms, n, trans_W=True):
    # Constructing the encoding/decoding matrices
    t = len(all_terms)
    U, V, W, WT = np.zeros((t, n * n)), np.zeros((t, n * n)), np.zeros((t, n * n)), np.zeros((t, n * n))
    for r, term in enumerate(all_terms):
        a_terms, b_terms, c_terms = term
        for scalar, i, j in a_terms:
            U[r, i * n + j] += scalar
        for scalar, j, k in b_terms:
            V[r, j * n + k] += scalar
        for scalar, k, i in c_terms:
            W[r, k + i * n] += scalar
            WT[r, i + k * n] += scalar  # SWAP INDICES

    # Logger.log("~_!" + str(np.allclose(transpose_W(W, n), WT)))
    if trans_W:
        return U, V, WT
    return U, V, W
