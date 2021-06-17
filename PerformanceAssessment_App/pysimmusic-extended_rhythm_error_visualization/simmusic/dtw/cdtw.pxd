cdef extern from "dtw.h":

    ctypedef struct DTW_path:
        int plen
        int *px
        int *py

    ctypedef struct dtwParams_t:
        int distType
        int hasGlobalConst
        int globalType
        int bandwidth
        int initCostMtx
        int reuseCostMtx
        int delStep
        int moveStep
        int diagStep
        int initFirstCol
        int isSubsequence

    double dtw1d_std(double *x, double*y, int x_len, int y_len, double*cost, int dist_type)
    int path(double *cost, int n, int m, int startx, int starty, DTW_path *p)
    int pathLocal(double *cost, int n, int m, int startx, int starty, DTW_path *p, dtwParams_t params)
    double dtw_GLS(double *x, double*y, int x_len, int y_len, double*cost, dtwParams_t params)
    double dtw_vect_std(double *x, double *y, int size, int x_len, int y_len, double* cost)