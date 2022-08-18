# Late Fusion Algorithms
import numpy as np

# cl is a list of cluster solutions
# each entry is a numpy array

def consensus(cl):

    n_samp    = len(cl[0])
    cl_cons   = np.zeros((n_samp,), dtype=int)

    n_cl = len(cl)

    k = 1
    for xx in range(0, n_samp):

        ids = np.where(cl[0] == cl[0][xx])
        #ids <- which(CL_LIST[[1]]==CL_LIST[[1]][xx])

        for yy in range(1, n_cl):	

            m = np.where(cl[yy] == cl[yy][xx])
	        #m   <- which(CL_LIST[[yy]]==CL_LIST[[yy]][xx])
	        
            ids = np.intersect1d(ids, m)
            #ids <- intersect(ids,m) 

        check = np.sum(cl_cons[ids])
        if check == 0:
            cl_cons[ids] = k
            k = k + 1
            
    return(cl_cons)

