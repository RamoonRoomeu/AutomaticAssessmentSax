
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <float.h>

#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define N_SIM_MEASURES 10

enum
{
Euclidean=0,
SqEuclidean,
CityBlock,
ShiftCityBlock,
TMM_CityBlock,
ShiftLinExp
};

typedef struct dtwParams
{

    int distType;
    int hasGlobalConst;
    int globalType;
    int bandwidth;
    int initCostMtx;
    int reuseCostMtx;
    int delStep;
    int moveStep;
    int diagStep;
    int initFirstCol;
    int isSubsequence;

}dtwParams_t;

extern const char*SimMeasureNames[N_SIM_MEASURES];


typedef struct DTW_path
{
  int plen;
  int *px;
  int *py;
}DTW_path;

typedef double (*distMeasures)(double, double);


double dtw1d_std(double *x, double*y, int x_len, int y_len, double*cost, int dist_type);
double dtw_GLS(double *x, double*y, int x_len, int y_len, double*cost, dtwParams_t params);
double dtw_vect_std(double *x, double *y, int size, int x_len, int y_len, double* cost);

int path(double *cost, int n, int m, int startx, int starty, DTW_path *p);
int pathLocal(double *cost, int n, int m, int startx, int starty, DTW_path *p, dtwParams_t params);

