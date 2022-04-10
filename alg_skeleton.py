from util.test import *
from util.save import *
from util.decomp import *
from util.utils import *
from util.log import *
from util.print import *
from util.utils import *
from util.terms import *


def create_mats(gen_terms, decomp_mats, n0, n=0, trans_W=0):
    if n0 % 2 != 0:
        Logger.log("n must be divisible by ", 2)
        return
    if n == 0:
        n = n0
    Logger.log("\t-> Generating terms...")
    all_terms = gen_terms(n)
    Logger.log("\t-> Actual: n0 = ", n0, " t = ", len(all_terms), " w0 = ", math.log(len(all_terms), n0))
    # print_sterms(all_terms, n)

    Logger.log("\t-> Converting  initial terms to matrices...")
    U, V, W = terms_to_mats(all_terms, n, trans_W)
    del all_terms

    Logger.log("\t-> Creating Decomposition...")
    Us, Vs, Ws = decomp_mats(U, V, W, n0)

    Logger.log("\t-> Got U,V,W!")
    return Us, Vs, Ws


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


def get_mats(n0, n, algorithm_name, gen_terms, decomp_mats, force_new=False, trans_W=True):
    Us, Vs, Ws = read_matrices(n0, algorithm_name, orders=2)
    newly_saved = len(Us) < 1 or force_new
    if newly_saved:
        Logger.log("generating matrix first")
        Us, Vs, Ws = create_mats(gen_terms, decomp_mats, n0, n, trans_W)

    U, V, W = undecomp_mats(Us, Vs, Ws)

    if newly_saved:
        if not test_mats(U, V, W, n0):
            return False
        Logger.log("SAVING MATRICES!")
        write_matrices(Us, Vs, Ws, n0, algorithm_name)
        write_matrices([U], [V], [W], n0, algorithm_name)

    new_orders = get_all_orders(Us, Vs, Ws)
    old_orders = get_all_orders([U], [V], [W])

    Logger.log("Orders, old to new:")
    print_orders(old_orders)
    print_orders(new_orders)
    # code.interact(local=locals()
