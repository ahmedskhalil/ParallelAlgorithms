/*
 * -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: 4; -*-
 * Author        : Ahmed Khalil
 * Created       : 12.04.20
 * 
 * 
 */





#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>
#include <sys/time.h>
#include <pthread.h>



#define NUM         10000000
#define TH_LVL      10

void swap(double lst[], int x, int y);
int partitioner(double lst[], int first, int last);

void helper(double lst[], int first, int last);
void qsort(double lst[], int size);
int isSorted(double lst[], int size);

void  pqsort(double lst[], int size, int thlvl);
void* phelper(void* tharg) __attribute__ ((noreturn));



struct th_data {
    double* dlst; int fst; int lst; int lv;
};

int doublesComparer(const void* x, const void* b);


int main(int argc, char* argv[]){
    struct timeval start, end;
    double dur;

    srand(time(NULL));

    int NUM0 = NUM;
    if (argc == 2){
        NUM0 = atoi(argv[1]);
    } 

    double* dlstonhold = (double*) malloc(NUM0 * sizeof(double));
    double* dlst       = (double*) malloc(NUM0 * sizeof(double));

    // gen random dlst
    for (int i = 0; i < NUM0; i++){
        dlstonhold[i] = 1.0 * rand() / RAND_MAX;
    }
    memcpy(dlst, dlstonhold, NUM0 * sizeof(double));

    // qsort benchmarking
    gettimeofday(&start, NULL);
    qsort(dlst, NUM0);
    gettimeofday(&end, NULL);

    if (!isSorted(dlst, NUM0)) printf("ALERT: not sorted!\n");

    dur = (  (end.tv_sec * 1000000   + end.tv_usec)
           - (start.tv_sec * 1000000 + start.tv_usec)
          )/ 1000000.0;
    printf("Ordinary quicksort processed in: %lf sec.\n", dur);


    // parallel qsort benchmarking
    memcpy(dlst, dlstonhold, NUM0 * sizeof(double));

    gettimeofday(&start, NULL);
    pqsort(dlst, NUM0, TH_LVL);
    gettimeofday(&end, NULL);

    if (!isSorted(dlst, NUM0)) printf("ALERT: not sorted!\n");

    dur = (  (end.tv_sec * 1000000   + end.tv_usec)
           - (start.tv_sec * 1000000 + start.tv_usec)
          )/ 1000000.0;
    printf("Parallelised quicksort processed in: %lf sec.\n", dur);

    return 0;
}



void swap(double lst[], int x, int y){
    double temp = lst[x];
    lst[x] = lst[y];
    lst[y] = temp;
}

int partitioner(double lst[], int first, int last){
    int b = first;
    int r = (int)(first + (last-first) * (1.0 * rand() / RAND_MAX));
    double pivot = lst[r];
    swap(lst,r,last);
    for (int i = first; i<last; i++){
        if(lst[i] < pivot){
            swap(lst, i, b);
            b++;
        }
    }
    swap(lst, last, b);
    return b;
}

void helper(double lst[], int first, int last){
    if (first >= last) return;
    int b = partitioner(lst, first, last);
    helper(lst, first, b-1);
    helper(lst, b+1, last);
}

void qsort(double lst[], int size){
    helper(lst, 0, size-1);
}

int isSorted(double lst[], int size){
    for (int i = 0; i< size; i++){
        if (lst[i] < lst[i-1]){
            printf("isSorted -> check at %d : %e < %e\n", 
                    i, lst[i], lst[i-1]);
                    return 0;
        }
    }
    return 1; // ascendingly sorted
}

void  pqsort(double dlst[], int size, int thlvl){
    int cval;
    void* status;

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    struct th_data thd;
    thd.dlst = dlst;
    thd.fst  = 0;
    thd.lst  = size - 1;
    thd.lv   = thlvl;

    pthread_t toplvl_th;
    cval = pthread_create(&toplvl_th, &attr, phelper, (void*)&thd);
    if(cval) { printf("ERROR: failed to create thread\n"); exit(-1); }

    pthread_attr_destroy(&attr);
    cval = pthread_join(toplvl_th, &status);
    if(cval) { printf("ERROR: failed to join thread\n"); exit(-1); }
}


void* phelper(void* tharg){
    int t, cval, mid;
    void* status;

    struct th_data* thdata;
    thdata = (struct th_data*) tharg;

    if (thdata->lv <= 0 || thdata->fst == thdata->lst+1){
        helper(thdata->dlst, thdata->fst, thdata->lst);
        pthread_exit(NULL);
    }

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    mid = partitioner(thdata->dlst, thdata->fst, thdata->lst);

    struct th_data th_data_array[2];

    for (t=0; t<2; t++){
        th_data_array[t].dlst = thdata->dlst;
        th_data_array[t].lv   = thdata->lv - 1;
    }

    th_data_array[0].fst = thdata->fst;
    th_data_array[0].lst = mid - 1;
    th_data_array[1].fst = mid + 1;
    th_data_array[1].lst = thdata->lst;

    pthread_t threads[2];
    for (t=0; t<2; t++){
        cval = pthread_create(&threads[t], &attr, phelper, 
                                (void*) &th_data_array[t]);

        if(cval) { printf("ERROR: failed to create thread\n"); exit(-1); }
    }

    pthread_attr_destroy(&attr);
    // joining left and right
    for (t=0; t<2; t++){
        cval = pthread_join(threads[t], &status);
        if(cval) { printf("ERROR: failed to join thread\n"); exit(-1); }
    }
    pthread_exit(NULL);
}