# GCC
# gcc -c -fPIC HC_fused_cpp_opt6.cpp -o HC_fused_cpp_opt6.o
# (Other options that have been suggested are: -lstdc++)
# Then make the shared library:
# gcc HC_fused_cpp_opt6.o -shared -o libhcfused.so
# g++ did not work, using clang++ instead

# clang
# clang++ -c -fPIC HC_fused_cpp_opt6.cpp -o HC_fused_cpp_opt6.o
# clang++ HC_fused_cpp_opt6.o -shared -o libhcfused.so
# One liner:
# clang++ -shared -o libhcfused.so -fPIC HC_fused_cpp_opt6.cpp

from ctypes import cdll, c_int, c_float, POINTER
import pathlib
import numpy as np

# For distribution use:
# mylib_path = ctypes.util.find_library("./mylib") ## useful for multi platform

lib_path = pathlib.Path().absolute() / "libhcfused.so"
hc_fused = cdll.LoadLibrary(lib_path)

array_dim = 10

hc_fused.HC_fused_cpp_opt6.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C'), c_int, c_int, c_int]
hc_fused.HC_fused_cpp_opt6.restype =  POINTER(c_int * (array_dim * array_dim))

d1 = np.random.randint(2, size=(array_dim * array_dim), dtype=np.int32)
d2 = np.random.randint(2, size=(array_dim * array_dim), dtype=np.int32)
d3 = np.random.randint(2, size=(array_dim * array_dim), dtype=np.int32)

d = np.array([d1, d2, d3])

n_cluster_arrays = np.shape(d)[0]
cluster_size = np.shape(d)[1]
n_iterations = 10

c_call = hc_fused.HC_fused_cpp_opt6(d, n_cluster_arrays, cluster_size, n_iterations)  # Data, number of arrays, length of one cluster, and the number of iterations

print([i for i in c_call.contents])
