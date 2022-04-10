import numpy as np
import scipy.io
from util.log import *
import os.path


def get_path(mat, n, order, orders, algorithm_name, part=0):
    path = "mat/" + algorithm_name + "/" + str(n) + "/" + str(orders) + "/" + mat + "/" + mat + "_o" + str(order) + ("" if part < 0 else "_p" + str(part)) + ".mat"
    return path

def create_dirs(path):
    for i in range(1, len(path.split('/'))):
        try:
            os.mkdir('/'.join(path.split('/')[:i]))
        except:
            continue

def write_matrix_order(M, mat, n, order, orders, algorithm_name):
    MAX_ROWS = 70000
    if(M.shape[0] < MAX_ROWS):
        write_matrix_part(M, mat, n, order, orders, algorithm_name)
        return

    parts = []
    for part in range(M.shape[0] // MAX_ROWS + 1):
        if(M.shape[0] <= part * MAX_ROWS):
            continue
        M_part = M[part * MAX_ROWS:(part+1) * MAX_ROWS]
        write_matrix_part(M_part, mat, n, order, orders, algorithm_name, part)


def write_matrix_part(M, mat, n, order, orders, algorithm_name, part=-1):
    path = get_path(mat, n, order, orders, algorithm_name, part)
    create_dirs(path)
    scipy.io.savemat(path, {mat: M})


def read_matrix_order(mat, n, order, orders, algorithm_name):
    path = get_path(mat, n, order, orders, algorithm_name)
    if(os.path.isfile(path)):
        return scipy.io.loadmat(path)[mat]
    path = get_path(mat, n, order, orders, algorithm_name, 0)
    if (os.path.isfile(path)):
        M = scipy.io.loadmat(path)[mat]
        for part in range(1,20):
            path = get_path(mat, n, order, orders, algorithm_name, part)
            if not os.path.isfile(path):
                break
            M = np.concatenate(M, scipy.io.loadmat(path)[mat])
        return M
    raise Exception()



def write_matrices(Us, Vs, Ws, n, algorithm_name):
    orders = max(len(Us), len(Vs), len(Ws))
    for m in [(Us, "U"), (Vs, "V"), (Ws, "W")]:
        mat = m[1]
        orders = len(m[0])
        for order in range(orders):
            M = m[0][order]
            write_matrix_order(M, mat, n, order, orders, algorithm_name)


def read_matrices(n, algorithm_name, orders=-1):
    if orders < 0:
        orders = 1000
    Us, Vs, Ws = [], [], []

    for order in range(orders):
        try:
            M = read_matrix_order("U", n, order, orders, algorithm_name)
            #Logger.log("Got U!");
            Us.append(M)
        except:
            break

    for order in range(orders):
        try:
            M = read_matrix_order("V", n, order, orders, algorithm_name)
            #Logger.log("Got V!");
            Vs.append(M)
        except:
            break

    for order in range(orders):
        try:
            M = read_matrix_order("W", n, order, orders, algorithm_name)
            #Logger.log("Got W!");
            Ws.append(M)
        except:
            break

    return Us, Vs, Ws
