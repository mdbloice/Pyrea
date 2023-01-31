#ifdef _MSC_VER
    #define EXPORT_SYMBOL __declspec(dllexport)
#else
    #define EXPORT_SYMBOL
#endif

#ifdef __cplusplus
extern "C" {
#endif

EXPORT_SYMBOL int* HC_fused_cpp_opt6(int* MAT_array, int n_clusters, int cluster_size, int n_iter);

#ifdef __cplusplus
}
#endif