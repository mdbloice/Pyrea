# Late Fusion Algorithms
import numpy as np

######
# CONSENSUS
# cl is a list of cluster solutions
# each entry is a numpy array
#######
def consensus(cl):

    n_samp    = len(cl[0])
    cl_cons   = np.zeros((n_samp,), dtype=int)

    n_cl = len(cl)

    k = 1
    for xx in range(0, n_samp):

        ids = np.where(cl[0] == cl[0][xx])
        
        for yy in range(1, n_cl):	

            m = np.where(cl[yy] == cl[yy][xx])
            ids = np.intersect1d(ids, m)
            
        check = np.sum(cl_cons[ids])
        if check == 0:
            cl_cons[ids] = k
            k = k + 1

    return(cl_cons)

######
# AGREEMENT MATRIX
# cl is a list of cluster solutions
# each entry is a numpy array
# RETURN is a matrix
#######

def agreement(cl):

    n_samp  = len(cl[0])
    mat  = np.zeros((n_samp, n_samp), dtype=int)
    for xx in range(0, len(cl)):

        l = cl[xx]
        res = [[int(x == y) for y in l] for x in l]
        res = np.matrix(res)
        mat = mat + res

    return(mat)


######
# DISAGREEMENT MATRIX
# cl is a list of cluster solutions
# each entry is a numpy array
# RETURN is a matrix
#######

def disagreement(cl):

    n_samp  = len(cl[0])
    mat  = np.zeros((n_samp, n_samp), dtype=int)
    for xx in range(0, len(cl)):

        l = cl[xx]
        res = [[int(x != y) for y in l] for x in l]
        res = np.matrix(res)
        mat = mat + res

    return(mat)

