#include "dtw.h"

//########################################## Similarity Measures #####################################
double distEuclidean(double a, double b)
{
    return fabs(a-b); //for one dimensional single value equclidean is essentially this
}

double distSqEuclidean(double a, double b)
{
    double diff;
    diff = a-b;
    return (diff*diff);
}
double L2Norm(double* a, double* b, int size)
{
    double cost = 0;
    for(int i=0; i<size;++i)
    {
        cost+=distSqEuclidean(a[i], b[i]);
    }
    return sqrt(cost);
}
double min3(double a, double b, double c)
{
    if (b < a)
        a = b;
    if (c < a)
        return c;
    return a;
}

distMeasures myDistMeasures[N_SIM_MEASURES] = {distEuclidean, distSqEuclidean};




// Compute the warp path starting at cost[startx, starty]
// If startx = -1 -> startx = n-1; if starty = -1 -> starty = m-1
int path(double *cost, int n, int m, int startx, int starty, DTW_path *p)
{
    int i, j, k, z1, z2;
    int *px;
    int *py;
    double min_cost;

    if ((startx >= n) || (starty >= m))
        return 0;

    if (startx < 0)
        startx = n - 1;

    if (starty < 0)
        starty = m - 1;

    i = startx;
    j = starty;
    k = 1;

    // allocate path for the worst case
    px = (int *) malloc ((startx+1) * (starty+1) * sizeof(int));
    py = (int *) malloc ((startx+1) * (starty+1) * sizeof(int));

    px[0] = i;
    py[0] = j;

    while ((i > 0) || (j > 0))
    {
        if (i == 0)
            j--;
        else if (j == 0)
            i--;
        else
        {
            min_cost = min3(cost[(i-1)*m+j],
                            cost[(i-1)*m+(j-1)],
                            cost[i*m+(j-1)]);

            if (cost[(i-1)*m+(j-1)] == min_cost)
            {
                i--;
                j--;
            }
            else if (cost[i*m+(j-1)] == min_cost)
                j--;
            else
                i--;
        }

        px[k] = i;
        py[k] = j;
        k++;
    }

    p->px = (int *) malloc (k * sizeof(int));
    p->py = (int *) malloc (k * sizeof(int));
    for (z1=0, z2=k-1; z1<k; z1++, z2--)
        {
            p->px[z1] = px[z2];
            p->py[z1] = py[z2];
        }
    p->plen = k;

    free(px);
    free(py);

    return 1;
}

//This is a basic (standard) DTW measure without any constraint or fansy stuff. This is for one dimensional time series.
double dtw1d_std(double *x, double*y, int x_len, int y_len, double*cost, int dist_type)
{
        // declarations of variables
        int i,j;

        //Initializing the row and columns of cost matrix
        cost[0]= (*myDistMeasures[dist_type])(x[0],y[0]);
        for (i=1;i<x_len;i++)
        {
            cost[i*y_len]=(*myDistMeasures[dist_type])(x[i],y[0]) + cost[(i-1)*y_len];
        }
        for (j=1;j<y_len;j++)
        {
            cost[j]=(*myDistMeasures[dist_type])(x[0],y[j]) + cost[(j-1)];
        }

        //filling in all the cumulative cost matrix
        for (i=1;i<x_len;i++)
        {
            for (j=1;j<y_len;j++)
            {
                cost[(i*y_len)+ j] = (*myDistMeasures[dist_type])(x[i],y[j]) +
                                        min3(cost[(i-1)*y_len+j], cost[((i-1)*y_len)+(j-1)], cost[(i*y_len)+(j-1)]);
            }

        }

        return cost[(x_len*y_len)-1];
}

double dtw_vect_std(double *x, double *y, int size, int x_len, int y_len, double* cost)
{
    int i,j;   
    
    //Initializing the row and columns of cost matrix
    cost[0] = L2Norm(x,y, size);
    for (i=1;i<x_len;++i)
    {
        cost[i*y_len]=L2Norm(x+(i*size),y, size) + cost[(i-1)*y_len];
    }
    for (j=1;j<y_len;++j)
    {
        cost[j]=L2Norm(x,y+(j*size), size) + cost[(j-1)];
    }

    //filling in all the cumulative cost matrix
    for (i=1;i<x_len;i++)
    {
        for (j=1;j<y_len;j++)
        {
            cost[(i*y_len)+ j] = L2Norm(x+(i*size),y+(j*size), size) +
                                    min3(cost[(i-1)*y_len+j], cost[((i-1)*y_len)+(j-1)], cost[(i*y_len)+(j-1)]);
        }
    }

    return cost[(x_len*y_len)-1];
}


/*
 * All the DTW variants for one dimensional time series
 *
 * NOTE: this first category of dtw functions are quite generic and can be used in C/Cython/Python/Mex/MATLAB. They do not require any specific consideration from the calling function.
 * Whereas there are some specific dtw versions I wrote specifically for my usage which incorporate lower bounding, early abandoning and do not initialize cost matrix for faster processing. (find them after this category)
 *
 * We should Ideally be keeping input param structure same for all variants so that they can be later indexed by a function pointer
*   # ==    Global  Local   Subsequence
*           No      No      No
*           Yes     No      No
*           No      Yes     No
*           Yes     Yes     No
*           No      Yes     Yes (with sub version it always make sense to have local alignment)
*           Yes     Yes     Yes (with sub version it always make sense to have local alignment)
 */

double dtw_GLS(double *x, double*y, int x_len, int y_len, double*cost, dtwParams_t params)
{
        // declarations of variables
        int i,j, bandwidth;
        float min_vals, factor, max_del, bound;

        max_del = max(params.delStep, params.moveStep);
        bandwidth = params.bandwidth;
//
//        if (params.globalType==0)
//        {
//            /*This is along 45 degree, the usual way to do in dtw*/
//            factor=1;
//            /*Since this is subsequence dtw, even if bandwidth is smaller than abs(x_len-y_len) its not a problem*/
//        }
//        else if (params.globalType==1)
//        {
//            factor = (float)y_len/(float)x_len;
//        }

        if (params.initCostMtx==1)
        {
            //putting infi in all cost mtx
            for (i=0;i<x_len;i++)
            {
                for (j=0;j<y_len;j++)
                {
                    cost[(i*y_len)+ j] = FLT_MAX;
                }

            }
        }
        if (params.hasGlobalConst==0)
        {
            bandwidth = max(x_len, y_len);
        }

        if (params.reuseCostMtx==0)
        {


            //Initializing the row and columns of cost matrix
            cost[0]= (*myDistMeasures[params.distType])(x[0],y[0]);
            if (params.isSubsequence==1){

                for (i=1;i<min(bandwidth+1, x_len);i++)
                    {
                        if (params.initFirstCol==1)
                        {
                            cost[i*y_len]=(*myDistMeasures[params.distType])(x[i],y[0]) + cost[((i-1)*y_len)];
                        }
                        else{
                            cost[i*y_len]=(*myDistMeasures[params.distType])(x[i],y[0]);
                        }

                    }
                for (j=1;j< min(bandwidth+1, y_len);j++)
                    {
                        cost[j]=(*myDistMeasures[params.distType])(x[0],y[j]);
                    }

            }
            else{
                for (i=1;i<min(bandwidth+1, x_len);i++)
                {
                    cost[i*y_len]=(*myDistMeasures[params.distType])(x[i],y[0]) + cost[((i-1)*y_len)];
                }
                for (j=1;j< min(bandwidth+1, y_len);j++)
                {
                    cost[j]=(*myDistMeasures[params.distType])(x[0],y[j]) + cost[j-1];
                }

            }


            /*Initializing other rows and colms till we can support out local constraints to be applied*/
            for (i=1;i<=min(bandwidth+1, x_len-1);i++)
            {
                for (j=1;j<max_del;j++)
                {
                    cost[(i*y_len)+ j] = (*myDistMeasures[params.distType])(x[i],y[j]) + min3(cost[((i-1)*y_len)+ j], cost[((i-1)*y_len)+(j-1)], cost[((i)*y_len)+(j-1)]);
                }
            }
            for (j=1;j<= min(bandwidth+1,y_len-1);j++)
            {
                for (i=1;i<max_del;i++)
                {
                    cost[(i*y_len)+ j] = (*myDistMeasures[params.distType])(x[i],y[j]) + min3(cost[((i-1)*y_len)+(j)], cost[((i-1)*y_len)+(j-1)], cost[((i)*y_len)+(j-1)]);
                }
            }

            //filling in all the cumulative cost matrix
            for (i=max_del;i<x_len;i++)
            {
                for (j=max(max_del, i-bandwidth);j<=min(y_len-1, i+bandwidth);j++)
                {
                    cost[(i*y_len)+ j] = (*myDistMeasures[params.distType])(x[i],y[j]) +
                                            min3(cost[(i-params.moveStep)*y_len+(j-params.delStep)], cost[((i-params.diagStep)*y_len)+(j-params.diagStep)], cost[((i-params.delStep)*y_len)+(j-params.moveStep)]);
                }

            }
        }

        if (params.isSubsequence==1)
        {
            min_vals = FLT_MAX;
            for (i=x_len-1;i>=max(x_len-1-bandwidth, 0);i--)
            {
                j = y_len -1;
                if(cost[(i*y_len)+ j] < min_vals)
                {
                    min_vals = cost[(i*y_len)+ j];
                }
            }
            for (j=y_len-1;j>=max(y_len-1-bandwidth,0);j--)
            {
                i = x_len -1;
                if(cost[(i*y_len)+ j] < min_vals)
                {
                    min_vals = cost[(i*y_len)+ j];
                }
            }
        }
        else
        {
            min_vals = cost[x_len*y_len -1];
        }
        return min_vals;

}

/*
This implementation is a variant of dtw_GLS with special processing needed for Riyaz context. THe differences are in terms
 of handling the silence regions.

  In this case we do not aggregate any cost for those regions, local cost for silence regionsis taken to be 0.
  Since we assume that silence regions maintain timing (an artist would be in time even though he takes a pause or the pitch
  detection method says unvoiced segment) we consider small transition cost to make sure that during the silence regions
  we always go diagonal (i.e. no insertion or deletion)

*/
//double dtw_GLS_riyaz(double *x, double*y, int x_len, int y_len, double*cost, dtwParams_t params, double sil_val)
//{
//        // declarations of variables
//        int i,j, bandwidth;
//        float min_vals, factor, max_del, bound;
//
//        max_del = max(params.delStep, params.moveStep);
//        bandwidth = params.bandwidth;
////
////        if (params.globalType==0)
////        {
////            /*This is along 45 degree, the usual way to do in dtw*/
////            factor=1;
////            /*Since this is subsequence dtw, even if bandwidth is smaller than abs(x_len-y_len) its not a problem*/
////        }
////        else if (params.globalType==1)
////        {
////            factor = (float)y_len/(float)x_len;
////        }
//
//        if (params.initCostMtx==1)
//        {
//            //putting infi in all cost mtx
//            for (i=0;i<x_len;i++)
//            {
//                for (j=0;j<y_len;j++)
//                {
//                    cost[(i*y_len)+ j] = FLT_MAX;
//                }
//
//            }
//        }
//        if (params.hasGlobalConst==0)
//        {
//            bandwidth = max(x_len, y_len);
//        }
//
//        if (params.reuseCostMtx==0)
//        {
//
//
//            //Initializing the row and columns of cost matrix
//            cost[0]= (*myDistMeasures[params.distType])(x[0],y[0]);
//            if (params.isSubsequence==1){
//
//                for (i=1;i<min(bandwidth+1, x_len);i++)
//                    {
//                        if (params.initFirstCol==1)
//                        {
//                            cost[i*y_len]=(*myDistMeasures[params.distType])(x[i],y[0]) + cost[((i-1)*y_len)];
//                        }
//                        else{
//                            cost[i*y_len]=(*myDistMeasures[params.distType])(x[i],y[0]);
//                        }
//
//                    }
//                for (j=1;j< min(bandwidth+1, y_len);j++)
//                    {
//                        cost[j]=(*myDistMeasures[params.distType])(x[0],y[j]);
//                    }
//
//            }
//            else{
//                for (i=1;i<min(bandwidth+1, x_len);i++)
//                {
//                    cost[i*y_len]=(*myDistMeasures[params.distType])(x[i],y[0]) + cost[((i-1)*y_len)];
//                }
//                for (j=1;j< min(bandwidth+1, y_len);j++)
//                {
//                    cost[j]=(*myDistMeasures[params.distType])(x[0],y[j]) + cost[j-1];
//                }
//
//            }
//
//
//            /*Initializing other rows and colms till we can support out local constraints to be applied*/
//            for (i=1;i<=min(bandwidth+1, x_len-1);i++)
//            {
//                for (j=1;j<max_del;j++)
//                {
//                    cost[(i*y_len)+ j] = (*myDistMeasures[params.distType])(x[i],y[j]) + min3(((i-1)*y_len)+ j, cost[((i-1)*y_len)+(j-1)], cost[((i)*y_len)+(j-1)]);
//                }
//            }
//            for (j=1;j<= min(bandwidth+1,y_len-1);j++)
//            {
//                for (i=1;i<max_del;i++)
//                {
//                    cost[(i*y_len)+ j] = (*myDistMeasures[params.distType])(x[i],y[j]) + min3(cost[((i-1)*y_len)+(j)], cost[((i-1)*y_len)+(j-1)], cost[((i)*y_len)+(j-1)]);
//                }
//            }
//
//            //filling in all the cumulative cost matrix
//            for (i=max_del;i<x_len;i++)
//            {
//                for (j=max(max_del, i-bandwidth);j<=min(y_len-1, i+bandwidth);j++)
//                {
//                    cost[(i*y_len)+ j] = (*myDistMeasures[params.distType])(x[i],y[j]) +
//                                            min3(cost[(i-params.moveStep)*y_len+(j-params.delStep)], cost[((i-params.diagStep)*y_len)+(j-params.diagStep)], cost[((i-params.delStep)*y_len)+(j-params.moveStep)]);
//                }
//
//            }
//        }
//
//        if (params.isSubsequence==1)
//        {
//            min_vals = FLT_MAX;
//            for (i=x_len-1;i>=max(x_len-1-bandwidth, 0);i--)
//            {
//                j = y_len -1;
//                if(cost[(i*y_len)+ j] < min_vals)
//                {
//                    min_vals = cost[(i*y_len)+ j];
//                }
//            }
//            for (j=y_len-1;j>=max(y_len-1-bandwidth,0);j--)
//            {
//                i = x_len -1;
//                if(cost[(i*y_len)+ j] < min_vals)
//                {
//                    min_vals = cost[(i*y_len)+ j];
//                }
//            }
//        }
//        else
//        {
//            min_vals = cost[x_len*y_len -1];
//        }
//        return min_vals;
//
//}


int pathLocal(double *cost, int n, int m, int startx, int starty, DTW_path *p, dtwParams_t params) 
{
    //params.delStep, params.moveStep and params.diagStep
    int i, j, k, z1, z2, a;
    int *px;
    int *py;
    double min_cost;

    // params which store the
    float diag, moveDel, delMove;

    if ((startx >= n) || (starty >= m))
        return 0;

    if (startx < 0) 
    {
        startx = n - 1;
    }

    if (starty < 0) 
    {
        starty = m - 1;
    }

    i = startx;
    j = starty;
    k = 1;

    // allocate path for the worst case
    px = (int *) malloc ((startx+1) * (starty+1) * sizeof(int));
    py = (int *) malloc ((startx+1) * (starty+1) * sizeof(int));

    px[0] = i;
    py[0] = j;

    while ((i-params.delStep >= 0 && j-params.moveStep >= 0) || (i-params.moveStep >= 0 && j-params.delStep >= 0) || i-params.diagStep >= 0) 
    {
        if (i == 0) 
        {
            j--;
        }
        else if (j == 0) {
            i--;
        }
        else 
        {
            if (i-params.delStep < 0 || j-params.moveStep < 0) {
                delMove = FLT_MAX;
            } 
            else 
            {
                delMove = cost[(i-params.delStep)*m+(j-params.moveStep)];
            }

            if (i-params.moveStep < 0 || j-params.delStep < 0) {
                moveDel = FLT_MAX;
            } 
            else 
            {
                moveDel = cost[(i-params.moveStep)*m+(j-params.delStep)];
            }

            if (i-params.diagStep < 0) {
                diag = FLT_MAX;
            } 
            else 
            {
                diag = cost[(i-params.diagStep)*m+(j-params.diagStep)];
            }

            min_cost = min3(delMove, moveDel, diag);

            if (diag == min_cost) {
                i -= params.diagStep;
                j -= params.diagStep;
            }
            else if (moveDel == min_cost) {
                i -= params.moveStep;
                j -= params.delStep;
            }
            else if (delMove == min_cost) {
                i -= params.delStep;
                j -= params.moveStep;
            }

        }


        px[k] = i;
        py[k] = j;
        k++;
    }


    p->px = (int *) malloc (k * sizeof(int));
    p->py = (int *) malloc (k * sizeof(int));

    for (z1=0, z2=k-1; z1<k; z1++, z2--)
    {

        p->px[z1] = px[z2];
        p->py[z1] = py[z2];

    }

    p->plen = k;

    free(px);
    free(py);

    return 1;
}