import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from cdtw cimport *

np.import_array()

def dtw_vector(x, y): 
    """
    dtw_vector(x, y):

    This function performs dynamic time warping between multi dimensional vector input signals (whole subsequence matching).
    This variant of DTW does not implement any local or global constraints.
    :Parameters:
      x : 2d numpy array (Vector time series)
      y : 2d numpy array (Vector time series)
      
    
    :Returns:
      distance : unnormalized minimum-distance of warp path between sequences
      length :  path length
      path : tuple of two 1d numpy array (path_x, path_y) warp path
      cost : Accumulated cost matrix

    :TODO:
    Implement other distance measures, currently only L2 norm (euclidean distance) implemented

    References
    .. [Muller07] M Muller. Information Retrieval for Music and Motion. Springer, 2007.
    .. [Keogh01] E J Keogh, M J Pazzani. Derivative Dynamic Time Warping. In FirsWt SIAM International Conference on Data Mining, 2001.
    """
    cdef np.ndarray[np.float_t, ndim=2] x_arr
    cdef np.ndarray[np.float_t, ndim=2] y_arr
    cdef np.ndarray[np.float_t, ndim=2] cost_arr
    cdef np.ndarray[np.int_t, ndim=1] px_cord
    cdef np.ndarray[np.int_t, ndim=1] py_cord

    cdef double udist
    cdef DTW_path path_t

    x_arr = np.ascontiguousarray(x, dtype=np.float)
    y_arr = np.ascontiguousarray(y, dtype=np.float)
    
    cost_arr = np.empty((x_arr.shape[0], y_arr.shape[0]), dtype=np.float)
    
    udist = dtw_vect_std(<double *>x_arr.data, <double*>y_arr.data, x_arr.shape[1], x_arr.shape[0],y_arr.shape[0], <double*>cost_arr.data)
    
    #udist = cost_arr[-1][-1]
    path(<double*>cost_arr.data, cost_arr.shape[0], cost_arr.shape[1], -1, -1, &path_t)

    px_cord = np.empty(path_t.plen, dtype=np.int)
    py_cord = np.empty(path_t.plen, dtype=np.int)
    for i in range(path_t.plen):
        px_cord[i] = path_t.px[i]
        py_cord[i] = path_t.py[i]
    free (path_t.px)
    free (path_t.py)

    return udist, path_t.plen, (px_cord, py_cord), cost_arr

def dtw(x, y, dist_type=0):
    """
    dtw(x, y, dist_type=0)

    This function performs dynamic time warping between input signals (whole subsequence matching).
    This variant of DTW does not implement any local or global constraints.
    :Parameters:
      x : 1d numpy array
      y : 1d numpy array
      dist_type : distance type used for computing time warping
                  OPTIONS: 0 - Euclidean (which for 1 dimensional single point data is same as cityBlock)
                           1 - Squared Euclidean

    :Returns:
      distance : unnormalized minimum-distance of warp path between sequences
      length :  path length
      path : tuple of two 1d numpy array (path_x, path_y) warp path
      cost : Accumulated cost matrix

    References
    .. [Muller07] M Muller. Information Retrieval for Music and Motion. Springer, 2007.
    .. [Keogh01] E J Keogh, M J Pazzani. Derivative Dynamic Time Warping. In FirsWt SIAM International Conference on Data Mining, 2001.
    """
    
    cdef np.ndarray[np.float_t, ndim=1] x_arr
    cdef np.ndarray[np.float_t, ndim=1] y_arr
    cdef np.ndarray[np.float_t, ndim=2] cost_arr
    cdef np.ndarray[np.int_t, ndim=1] px_cord
    cdef np.ndarray[np.int_t, ndim=1] py_cord

    cdef double udist
    cdef DTW_path path_t

    x_arr = np.ascontiguousarray(x, dtype=np.float)
    y_arr = np.ascontiguousarray(y, dtype=np.float)

    cost_arr = np.empty((x_arr.shape[0], y_arr.shape[0]), dtype=np.float)

    udist = dtw1d_std(<double *>x_arr.data, <double*>y_arr.data, x_arr.shape[0],y_arr.shape[0], <double*>cost_arr.data, dist_type)

    path(<double*>cost_arr.data, cost_arr.shape[0], cost_arr.shape[1], -1, -1, &path_t)

    px_cord = np.empty(path_t.plen, dtype=np.int)
    py_cord = np.empty(path_t.plen, dtype=np.int)
    for i in range(path_t.plen):
        px_cord[i] = path_t.px[i]
        py_cord[i] = path_t.py[i]
    free (path_t.px)
    free (path_t.py)

    return udist, path_t.plen, (px_cord, py_cord), cost_arr



def dtw1d_GLS(x, y, distType=0, hasGlobalConst=1, globalType=0, bandwidth = 0.2, initCostMtx=1, reuseCostMtx=0, delStep=1, moveStep=1, diagStep=1, initFirstCol=0, isSubsequence=0):
    """Modified version of standard DTW as described in [Muller07] and [Keogh01],
    Modified code provided in mlpy.dtw
    This version is a subsequence version of the code with local constraints and global band constraint:

    :Parameters:
    x : 1d numpy array (length N) first sequence
    y : 1d numpy array (length M) second sequence

    :Returns: a list of same entities specified in the configuration parameter for 'Output'
    udist : (float) unnormalized minimum-distance of warp path between sequences
    plength : (float) path length
    path : tuple of two 1d numpy array (path_x, path_y) warp path
    cost : 2d numpy array (N,M) containing accumulated cost matrix


    References
    .. [Muller07] M Muller. Information Retrieval for Music and Motion. Springer, 2007.
    .. [Keogh01] E J Keogh, M J Pazzani. Derivative Dynamic Time Warping. In FirsWt SIAM International Conference on Data Mining, 2001.
    """

    cdef np.ndarray[np.float_t, ndim=1] x_arr
    cdef np.ndarray[np.float_t, ndim=1] y_arr
    cdef np.ndarray[np.float_t, ndim=2] cost_arr
    cdef np.ndarray[np.int_t, ndim=1] px_cord
    cdef np.ndarray[np.int_t, ndim=1] py_cord

    cdef double udist
    cdef DTW_path path_t
    cdef dtwParams_t myDtwParama



    x_arr = np.ascontiguousarray(x, dtype=np.float)
    y_arr = np.ascontiguousarray(y, dtype=np.float)

    cost_arr = np.ones((x_arr.shape[0], y_arr.shape[0]), dtype=np.float)

    myDtwParama.distType = distType;
    myDtwParama.hasGlobalConst = hasGlobalConst;
    myDtwParama.globalType = globalType;
    myDtwParama.bandwidth = np.round(np.min((x_arr.shape[0],y_arr.shape[0]))*bandwidth).astype(np.int);
    myDtwParama.initCostMtx = initCostMtx;
    myDtwParama.reuseCostMtx = reuseCostMtx;
    myDtwParama.delStep = delStep;
    myDtwParama.moveStep = moveStep;
    myDtwParama.diagStep = diagStep;
    myDtwParama.initFirstCol = initFirstCol;
    myDtwParama.isSubsequence = isSubsequence;


    udist = dtw_GLS(<double *>x_arr.data, <double*>y_arr.data, x_arr.shape[0],y_arr.shape[0], <double*>cost_arr.data, myDtwParama)

    if myDtwParama.isSubsequence == 0:
        x_cord = x_arr.shape[0]-1
        y_cord = y_arr.shape[0]-1
    elif myDtwParama.isSubsequence == 1:
        y_cord = np.argmin(cost_arr[x_arr.shape[0]-1, :])
        x_cord = x_arr.shape[0]-1
        min_val = cost_arr[x_cord, y_cord]
        if np.min(cost_arr[:,y_arr.shape[0]-1]) < min_val:
            x_cord = np.argmin(cost_arr[:,y_arr.shape[0]-1])
            y_cord = y_arr.shape[0]-1
    else:
        print "Please select a valid option for isSubsequence parameter"



    pathLocal(<double*>cost_arr.data, cost_arr.shape[0], cost_arr.shape[1], x_cord, y_cord, &path_t, myDtwParama)
    px_cord = np.empty(path_t.plen, dtype=np.int)
    py_cord = np.empty(path_t.plen, dtype=np.int)
    for i in range(path_t.plen):
        px_cord[i] = path_t.px[i]
        py_cord[i] = path_t.py[i]
    free (path_t.px)
    free (path_t.py)

    return udist, path_t.plen, (px_cord, py_cord), cost_arr
